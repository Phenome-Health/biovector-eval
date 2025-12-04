"""Embedding strategies for vector database construction.

This module defines different approaches for converting metabolite data
into vector embeddings:

1. SinglePrimaryStrategy: One vector per metabolite (primary name only)
2. SinglePooledStrategy: One vector per metabolite (average of name + synonyms)
3. MultiVectorStrategy: Multiple vectors per metabolite (name + synonyms + variations)
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class StrategyType(Enum):
    """Enumeration of available embedding strategies."""

    SINGLE_PRIMARY = "single-primary"
    SINGLE_POOLED = "single-pooled"
    MULTI_VECTOR = "multi-vector"


@dataclass
class VectorRecord:
    """A single vector record with metadata."""

    text: str
    embedding: np.ndarray
    hmdb_id: str
    tier: str
    weight: float


@dataclass
class EmbeddingStrategy(ABC):
    """Abstract base class for embedding strategies."""

    name: str = field(init=False)

    @abstractmethod
    def prepare_vectors(
        self,
        metabolite: dict[str, Any],
        model: SentenceTransformer,
    ) -> list[dict[str, Any]]:
        """Prepare vector(s) for a single metabolite.

        Args:
            metabolite: Dictionary with keys 'hmdb_id', 'name', 'synonyms'
            model: SentenceTransformer model for encoding

        Returns:
            List of dicts with keys: text, embedding, hmdb_id, tier, weight
        """
        pass

    @abstractmethod
    def vectors_per_metabolite(self) -> int | str:
        """Return the number of vectors per metabolite.

        Returns:
            int for fixed count, 'variable' for strategies that vary
        """
        pass


@dataclass
class SinglePrimaryStrategy(EmbeddingStrategy):
    """Strategy: One vector per metabolite from primary name only.

    This is the simplest approach - each metabolite gets exactly one
    embedding vector from its primary name.

    Pros:
        - Minimal storage (1 vector per metabolite)
        - Fast indexing and search
        - Simple to implement

    Cons:
        - Cannot match on synonyms
        - May miss fuzzy matches
    """

    name: str = field(default="single-primary", init=False)

    def prepare_vectors(
        self,
        metabolite: dict[str, Any],
        model: SentenceTransformer,
    ) -> list[dict[str, Any]]:
        """Generate single vector from primary name."""
        text = metabolite["name"]
        embedding = model.encode(text, normalize_embeddings=True)

        return [
            {
                "text": text,
                "embedding": embedding,
                "hmdb_id": metabolite["hmdb_id"],
                "tier": "primary",
                "weight": 1.0,
            }
        ]

    def vectors_per_metabolite(self) -> int:
        """Returns 1 (one vector per metabolite)."""
        return 1


@dataclass
class SinglePooledStrategy(EmbeddingStrategy):
    """Strategy: One vector per metabolite from averaged name + synonyms.

    This approach creates embeddings for the primary name and each synonym,
    then averages them into a single vector. This creates a "semantic centroid"
    that represents all the ways this metabolite can be referred to.

    Pros:
        - Same storage as SinglePrimary (1 vector per metabolite)
        - Better synonym matching than SinglePrimary
        - Captures semantic breadth of the metabolite

    Cons:
        - May dilute specificity of exact matches
        - Averaging may not preserve all semantic nuances
    """

    name: str = field(default="single-pooled", init=False)
    max_synonyms: int = 15

    def prepare_vectors(
        self,
        metabolite: dict[str, Any],
        model: SentenceTransformer,
    ) -> list[dict[str, Any]]:
        """Generate single vector from averaged name + synonyms."""
        texts = [metabolite["name"]]
        synonyms = metabolite.get("synonyms", [])[: self.max_synonyms]
        texts.extend(synonyms)

        # Get embeddings for all texts (not normalized yet)
        embeddings = model.encode(texts, normalize_embeddings=False)

        # Average and normalize
        mean_embedding = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(mean_embedding)
        if norm > 0:
            mean_embedding = mean_embedding / norm

        # For display, note what was pooled
        pooled_text = metabolite["name"]
        if synonyms:
            pooled_text += f" (+{len(synonyms)} synonyms)"

        return [
            {
                "text": pooled_text,
                "embedding": mean_embedding.astype(np.float32),
                "hmdb_id": metabolite["hmdb_id"],
                "tier": "pooled",
                "weight": 1.0,
            }
        ]

    def vectors_per_metabolite(self) -> int:
        """Returns 1 (one vector per metabolite)."""
        return 1


@dataclass
class MultiVectorStrategy(EmbeddingStrategy):
    """Strategy: Multiple vectors per metabolite with tiered weights.

    This approach creates separate vectors for:
    - Primary name (weight 1.0)
    - Synonyms (weight 0.9)
    - Text variations (weight 0.7)

    Variations include:
    - Greek letter conversions (alpha- ↔ α-)
    - Punctuation variations (hyphens, spaces)
    - Prefix removal (N-, O-, L-, D-, etc.)

    Pros:
        - Highest recall (matches on any variation)
        - Weighted scoring allows prioritization
        - Can match exact and fuzzy queries

    Cons:
        - Large storage (~36 vectors per metabolite)
        - Requires deduplication at search time
        - More complex indexing
    """

    name: str = field(default="multi-vector", init=False)
    max_synonyms: int = 15
    max_variations: int = 20
    primary_weight: float = 1.0
    synonym_weight: float = 0.9
    variation_weight: float = 0.7

    def prepare_vectors(
        self,
        metabolite: dict[str, Any],
        model: SentenceTransformer,
    ) -> list[dict[str, Any]]:
        """Generate multiple vectors with tiered weights."""
        hmdb_id = metabolite["hmdb_id"]
        name = metabolite["name"]
        synonyms = metabolite.get("synonyms", [])[: self.max_synonyms]

        vectors: list[dict[str, Any]] = []
        all_texts: list[str] = []
        text_metadata: list[tuple[str, float]] = []  # (tier, weight)

        # Primary name
        all_texts.append(name)
        text_metadata.append(("primary", self.primary_weight))

        # Synonyms
        for syn in synonyms:
            if syn.lower() != name.lower():  # Skip duplicates
                all_texts.append(syn)
                text_metadata.append(("synonym", self.synonym_weight))

        # Generate variations from name and synonyms
        variations = self._generate_variations(name, synonyms)
        existing_lower = {t.lower() for t in all_texts}

        for var in variations[: self.max_variations]:
            if var.lower() not in existing_lower:
                all_texts.append(var)
                text_metadata.append(("variation", self.variation_weight))
                existing_lower.add(var.lower())

        # Batch encode all texts
        embeddings = model.encode(all_texts, normalize_embeddings=True)

        # Build vector records
        for i, (text, embedding) in enumerate(zip(all_texts, embeddings, strict=True)):
            tier, weight = text_metadata[i]
            vectors.append(
                {
                    "text": text,
                    "embedding": embedding,
                    "hmdb_id": hmdb_id,
                    "tier": tier,
                    "weight": weight,
                }
            )

        return vectors

    def _generate_variations(
        self,
        name: str,
        synonyms: list[str],
    ) -> list[str]:
        """Generate text variations for better matching.

        Variations include:
        - Greek letter conversions
        - Punctuation variations
        - Prefix removal
        """
        variations: set[str] = set()
        source_texts = [name] + synonyms[:10]

        for text in source_texts:
            # Greek letter conversions
            greek_map = {
                "alpha-": "α-",
                "beta-": "β-",
                "gamma-": "γ-",
                "delta-": "δ-",
                "omega-": "ω-",
            }
            for eng, greek in greek_map.items():
                if eng in text.lower():
                    variations.add(text.lower().replace(eng, greek))
                if greek in text:
                    variations.add(text.replace(greek, eng))

            # Punctuation variations
            variations.add(text.replace("-", " "))
            variations.add(text.replace("-", ""))
            variations.add(text.replace("_", " "))
            variations.add(text.replace("_", "-"))

            # Prefix removal
            prefixes = ["N-", "O-", "L-", "D-", "S-", "R-", "1-", "2-", "3-"]
            for prefix in prefixes:
                if text.startswith(prefix):
                    variations.add(text[len(prefix) :])

            # Numeric prefix removal (e.g., "1,2-" or "3-")
            numeric_match = re.match(r"^\d+[,-]", text)
            if numeric_match:
                variations.add(text[numeric_match.end() :])

            # Parenthetical removal
            if "(" in text and ")" in text:
                no_parens = re.sub(r"\s*\([^)]*\)", "", text).strip()
                if no_parens:
                    variations.add(no_parens)

        # Remove empty and original texts
        variations.discard("")
        variations.discard(name)
        for syn in synonyms:
            variations.discard(syn)

        return list(variations)

    def vectors_per_metabolite(self) -> str:
        """Returns 'variable' (depends on synonym count)."""
        return "variable"


def get_strategy(strategy_type: StrategyType | str) -> EmbeddingStrategy:
    """Factory function to get strategy instance by type.

    Args:
        strategy_type: StrategyType enum or string name

    Returns:
        EmbeddingStrategy instance

    Raises:
        ValueError: If unknown strategy type
    """
    if isinstance(strategy_type, str):
        strategy_type = StrategyType(strategy_type)

    strategies = {
        StrategyType.SINGLE_PRIMARY: SinglePrimaryStrategy,
        StrategyType.SINGLE_POOLED: SinglePooledStrategy,
        StrategyType.MULTI_VECTOR: MultiVectorStrategy,
    }

    if strategy_type not in strategies:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

    return strategies[strategy_type]()
