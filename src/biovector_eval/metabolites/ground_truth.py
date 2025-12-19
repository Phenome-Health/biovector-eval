"""Ground truth generation for HMDB metabolite evaluation."""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class GroundTruthQuery:
    """A single ground truth query."""

    query: str
    expected: list[str]
    category: str
    difficulty: str = "medium"
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "expected": self.expected,
            "category": self.category,
            "difficulty": self.difficulty,
            "notes": self.notes,
        }


@dataclass
class GroundTruthDataset:
    """Collection of ground truth queries."""

    queries: list[GroundTruthQuery] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add(self, query: GroundTruthQuery) -> None:
        """Add a query to the dataset."""
        self.queries.append(query)

    def filter_by_category(self, category: str) -> list[GroundTruthQuery]:
        """Get queries of a specific category."""
        return [q for q in self.queries if q.category == category]

    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            "metadata": {
                **self.metadata,
                "total_queries": len(self.queries),
                "categories": list({q.category for q in self.queries}),
            },
            "queries": [q.to_dict() for q in self.queries],
        }

    def save(self, path: Path | str) -> None:
        """Save to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> GroundTruthDataset:
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)
        dataset = cls(metadata=data.get("metadata", {}))
        for q in data.get("queries", []):
            dataset.add(GroundTruthQuery(**q))
        return dataset


class HMDBGroundTruthGenerator:
    """Generate ground truth queries from HMDB metabolite data."""

    GREEK_MAPPINGS = {
        "α": "alpha",
        "β": "beta",
        "γ": "gamma",
        "δ": "delta",
        "ω": "omega",
    }

    # Keyboard proximity map for realistic substitution typos
    KEYBOARD_NEIGHBORS = {
        "a": "sqwz",
        "b": "vghn",
        "c": "xdfv",
        "d": "erfcxs",
        "e": "wrsdf",
        "f": "rtgvcd",
        "g": "tyhbvf",
        "h": "yujnbg",
        "i": "uojkl",
        "j": "uikmnh",
        "k": "iojlm",
        "l": "opk",
        "m": "njk",
        "n": "bhjm",
        "o": "iplk",
        "p": "ol",
        "q": "wa",
        "r": "edft",
        "s": "wedxza",
        "t": "rfgy",
        "u": "yhji",
        "v": "cfgb",
        "w": "qeas",
        "x": "zsdc",
        "y": "tghu",
        "z": "asx",
    }

    def __init__(self, metabolites: list[dict], seed: int = 42):
        """Initialize with metabolite data.

        Args:
            metabolites: List of dicts with 'hmdb_id', 'name', 'synonyms' keys.
            seed: Random seed for reproducibility.
        """
        self.metabolites = metabolites
        self.rng = random.Random(seed)

    def _apply_substitution(self, text: str, pos: int) -> str:
        """Replace character at pos with nearby keyboard key."""
        char = text[pos].lower()
        neighbors = self.KEYBOARD_NEIGHBORS.get(char, "aeiou")
        replacement = self.rng.choice(list(neighbors))
        if text[pos].isupper():
            replacement = replacement.upper()
        return text[:pos] + replacement + text[pos + 1 :]

    def _apply_transposition(self, text: str, pos: int) -> str:
        """Swap character at pos with next character."""
        if pos >= len(text) - 1:
            pos = len(text) - 2
        return text[:pos] + text[pos + 1] + text[pos] + text[pos + 2 :]

    def _apply_deletion(self, text: str, pos: int) -> str:
        """Delete character at pos."""
        return text[:pos] + text[pos + 1 :]

    def _apply_insertion(self, text: str, pos: int) -> str:
        """Insert duplicate or random character at pos."""
        if self.rng.random() < 0.7:
            # Duplicate adjacent character
            char = text[pos] if pos < len(text) else text[-1]
        else:
            # Random vowel
            char = self.rng.choice("aeiou")
        return text[:pos] + char + text[pos:]

    def _generate_typo(self, name: str) -> tuple[str, str]:
        """Generate a typo'd version of a name.

        Returns:
            Tuple of (typo_string, description of changes)
        """
        # Calculate number of typos (~10% of length, min 1, max 3)
        num_typos = max(1, min(3, len(name) // 10))

        result = name
        changes = []
        typo_funcs = [
            (self._apply_substitution, "sub"),
            (self._apply_transposition, "trans"),
            (self._apply_deletion, "del"),
            (self._apply_insertion, "ins"),
        ]

        for _ in range(num_typos):
            if len(result) < 3:
                break
            func, change_type = self.rng.choice(typo_funcs)
            # Avoid first/last char to keep recognizable
            pos = self.rng.randint(1, len(result) - 2)
            result = func(result, pos)
            changes.append(change_type)

        return result, "+".join(changes)

    def generate_exact_matches(self, n: int = 150) -> Iterator[GroundTruthQuery]:
        """Generate exact name match queries."""
        sampled = self.rng.sample(self.metabolites, min(n, len(self.metabolites)))
        for m in sampled:
            yield GroundTruthQuery(
                query=m["name"],
                expected=[m["hmdb_id"]],
                category="exact_match",
                difficulty="easy",
            )

    def generate_synonym_matches(self, n: int = 150) -> Iterator[GroundTruthQuery]:
        """Generate synonym match queries."""
        with_synonyms = [m for m in self.metabolites if m.get("synonyms")]
        sampled = self.rng.sample(with_synonyms, min(n, len(with_synonyms)))
        for m in sampled:
            synonym = self.rng.choice(m["synonyms"])
            yield GroundTruthQuery(
                query=synonym,
                expected=[m["hmdb_id"]],
                category="synonym_match",
                difficulty="medium",
                notes=f"Synonym for '{m['name']}'",
            )

    def generate_fuzzy_matches(self, n: int = 100) -> Iterator[GroundTruthQuery]:
        """Generate typo-based fuzzy match queries.

        Creates synthetic typos including:
        - Character substitutions (keyboard proximity)
        - Character transpositions (adjacent swap)
        - Character deletions
        - Character insertions/duplications

        Number of typos is ~10% of name length (min 1, max 3).
        """
        # Filter to names long enough for meaningful typos
        valid_names = [m for m in self.metabolites if len(m["name"]) >= 5]
        sampled = self.rng.sample(valid_names, min(n, len(valid_names)))

        for m in sampled:
            typo_name, changes = self._generate_typo(m["name"])
            yield GroundTruthQuery(
                query=typo_name,
                expected=[m["hmdb_id"]],
                category="fuzzy_match",
                difficulty="hard",
                notes=f"Typo ({changes}) of '{m['name']}'",
            )

    def generate_edge_cases(
        self, greek: int = 100, numeric: int = 100, special: int = 100
    ) -> Iterator[GroundTruthQuery]:
        """Generate edge case queries (Greek letters, numeric prefixes, special prefixes).

        Args:
            greek: Number of greek letter queries (default 100)
            numeric: Number of numeric prefix queries (default 100)
            special: Number of special prefix queries (default 100)
        """
        # Greek letters (alpha, beta, gamma, delta, omega)
        greek_metabolites = []
        for m in self.metabolites:
            name_lower = m["name"].lower()
            for greek_char, english in self.GREEK_MAPPINGS.items():
                if greek_char in name_lower or english in name_lower:
                    greek_metabolites.append(m)
                    break

        sampled_greek = self.rng.sample(
            greek_metabolites, min(greek, len(greek_metabolites))
        )
        for m in sampled_greek:
            yield GroundTruthQuery(
                query=m["name"],
                expected=[m["hmdb_id"]],
                category="greek_letter",
                difficulty="hard",
            )

        # Numeric prefixes (e.g., "3-hydroxybutyrate")
        numeric_pattern = re.compile(r"^\d+-")
        numeric_metabolites = [
            m for m in self.metabolites if numeric_pattern.match(m["name"])
        ]

        sampled_numeric = self.rng.sample(
            numeric_metabolites, min(numeric, len(numeric_metabolites))
        )
        for m in sampled_numeric:
            yield GroundTruthQuery(
                query=m["name"],
                expected=[m["hmdb_id"]],
                category="numeric_prefix",
                difficulty="medium",
            )

        # Special prefixes (e.g., "N-acetyl", "L-carnitine")
        special_pattern = re.compile(r"^[A-Z]-[a-z]")
        special_metabolites = [
            m for m in self.metabolites if special_pattern.match(m["name"])
        ]

        sampled_special = self.rng.sample(
            special_metabolites, min(special, len(special_metabolites))
        )
        for m in sampled_special:
            yield GroundTruthQuery(
                query=m["name"],
                expected=[m["hmdb_id"]],
                category="special_prefix",
                difficulty="medium",
            )

    def generate_dataset(
        self,
        exact: int = 150,
        synonym: int = 150,
        fuzzy: int = 100,
        greek: int = 100,
        numeric: int = 100,
        special: int = 100,
    ) -> GroundTruthDataset:
        """Generate complete ground truth dataset.

        Args:
            exact: Number of exact match queries.
            synonym: Number of synonym match queries.
            fuzzy: Number of fuzzy match queries.
            greek: Number of greek letter edge case queries.
            numeric: Number of numeric prefix edge case queries.
            special: Number of special prefix edge case queries.

        Returns:
            GroundTruthDataset with all queries.
        """
        dataset = GroundTruthDataset(
            metadata={
                "source": "HMDB",
                "generated": datetime.now().isoformat(),
                "version": "1.0",
            }
        )

        for q in self.generate_exact_matches(exact):
            dataset.add(q)
        for q in self.generate_synonym_matches(synonym):
            dataset.add(q)
        for q in self.generate_fuzzy_matches(fuzzy):
            dataset.add(q)
        for q in self.generate_edge_cases(greek, numeric, special):
            dataset.add(q)

        return dataset
