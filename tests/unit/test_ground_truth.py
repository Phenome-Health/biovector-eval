"""Tests for ground truth generation."""

import pytest

from biovector_eval.metabolites.ground_truth import (
    GroundTruthDataset,
    GroundTruthQuery,
    HMDBGroundTruthGenerator,
)


@pytest.fixture
def sample_metabolites() -> list[dict]:
    """Sample metabolite data for testing."""
    return [
        {
            "hmdb_id": "HMDB0000001",
            "name": "1-Methylhistidine",
            "synonyms": ["1-MHis", "pi-methylhistidine"],
        },
        {
            "hmdb_id": "HMDB0000122",
            "name": "Glucose",
            "synonyms": ["Dextrose", "Blood sugar", "D-Glucose"],
        },
        {
            "hmdb_id": "HMDB0001893",
            "name": "alpha-Tocopherol",
            "synonyms": ["Vitamin E", "a-Tocopherol"],
        },
        {
            "hmdb_id": "HMDB0000357",
            "name": "3-Hydroxybutyric acid",
            "synonyms": ["Beta-hydroxybutyrate", "BHB"],
        },
        {
            "hmdb_id": "HMDB0000574",
            "name": "N-Acetylcysteine",
            "synonyms": ["NAC", "Acetylcysteine"],
        },
        {
            "hmdb_id": "HMDB0000562",
            "name": "Creatinine",
            "synonyms": ["Creatine anhydride"],
        },
        {
            "hmdb_id": "HMDB0000123",
            "name": "Glycine",
            "synonyms": ["Aminoacetic acid", "Gly"],
        },
        {
            "hmdb_id": "HMDB0000161",
            "name": "L-Alanine",
            "synonyms": ["Alanine", "Ala"],
        },
    ]


class TestGroundTruthQuery:
    """Tests for GroundTruthQuery dataclass."""

    def test_to_dict(self) -> None:
        """Query converts to dictionary correctly."""
        query = GroundTruthQuery(
            query="glucose",
            expected=["HMDB0000122"],
            category="exact_match",
            difficulty="easy",
        )
        d = query.to_dict()
        assert d["query"] == "glucose"
        assert d["expected"] == ["HMDB0000122"]
        assert d["category"] == "exact_match"
        assert d["difficulty"] == "easy"

    def test_default_values(self) -> None:
        """Default values are set correctly."""
        query = GroundTruthQuery(
            query="test",
            expected=["ID1"],
            category="test_category",
        )
        assert query.difficulty == "medium"
        assert query.notes == ""


class TestGroundTruthDataset:
    """Tests for GroundTruthDataset."""

    def test_add_and_filter(self) -> None:
        """Can add queries and filter by category."""
        dataset = GroundTruthDataset()
        dataset.add(
            GroundTruthQuery(
                query="glucose",
                expected=["HMDB0000122"],
                category="exact_match",
            )
        )
        dataset.add(
            GroundTruthQuery(
                query="dextrose",
                expected=["HMDB0000122"],
                category="synonym_match",
            )
        )

        exact = dataset.filter_by_category("exact_match")
        assert len(exact) == 1
        assert exact[0].query == "glucose"

        synonym = dataset.filter_by_category("synonym_match")
        assert len(synonym) == 1
        assert synonym[0].query == "dextrose"

    def test_to_dict(self) -> None:
        """Dataset converts to dictionary correctly."""
        dataset = GroundTruthDataset(metadata={"version": "1.0"})
        dataset.add(
            GroundTruthQuery(
                query="glucose",
                expected=["HMDB0000122"],
                category="exact_match",
            )
        )

        d = dataset.to_dict()
        assert d["metadata"]["version"] == "1.0"
        assert d["metadata"]["total_queries"] == 1
        assert "exact_match" in d["metadata"]["categories"]
        assert len(d["queries"]) == 1

    def test_save_and_load(self, tmp_path) -> None:
        """Dataset can be saved and loaded."""
        dataset = GroundTruthDataset(metadata={"version": "1.0"})
        dataset.add(
            GroundTruthQuery(
                query="glucose",
                expected=["HMDB0000122"],
                category="exact_match",
            )
        )

        path = tmp_path / "ground_truth.json"
        dataset.save(path)

        loaded = GroundTruthDataset.load(path)
        assert len(loaded.queries) == 1
        assert loaded.queries[0].query == "glucose"
        assert loaded.metadata["version"] == "1.0"


class TestHMDBGroundTruthGenerator:
    """Tests for HMDB ground truth generation."""

    def test_exact_matches(self, sample_metabolites: list[dict]) -> None:
        """Generates exact match queries."""
        generator = HMDBGroundTruthGenerator(sample_metabolites, seed=42)
        queries = list(generator.generate_exact_matches(n=3))

        assert len(queries) == 3
        for q in queries:
            assert q.category == "exact_match"
            assert len(q.expected) == 1
            assert q.difficulty == "easy"

    def test_synonym_matches(self, sample_metabolites: list[dict]) -> None:
        """Generates synonym match queries."""
        generator = HMDBGroundTruthGenerator(sample_metabolites, seed=42)
        queries = list(generator.generate_synonym_matches(n=3))

        assert len(queries) == 3
        for q in queries:
            assert q.category == "synonym_match"
            assert len(q.expected) == 1
            assert q.difficulty == "medium"

    def test_fuzzy_matches(self, sample_metabolites: list[dict]) -> None:
        """Generates fuzzy match queries with synthetic typos."""
        generator = HMDBGroundTruthGenerator(sample_metabolites, seed=42)
        queries = list(generator.generate_fuzzy_matches(n=3))

        assert len(queries) == 3
        for q in queries:
            assert q.category == "fuzzy_match"
            assert len(q.expected) == 1
            assert q.difficulty == "hard"
            # Query should have typo notes
            assert "Typo" in q.notes
            # Extract original name from notes: "Typo (x) of 'Name'"
            original = q.notes.split("'")[1]
            # Query should be similar length to original (not a prefix)
            assert abs(len(q.query) - len(original)) <= 3  # At most 3 char diff

    def test_typo_generation(self, sample_metabolites: list[dict]) -> None:
        """Typo generation creates valid mutations."""
        generator = HMDBGroundTruthGenerator(sample_metabolites, seed=42)

        # Test typo generation on a known name
        test_name = "Glucose"
        typo, changes = generator._generate_typo(test_name)

        assert typo != test_name  # Should be different
        assert len(typo) >= len(test_name) - 3  # Not too short
        assert len(typo) <= len(test_name) + 3  # Not too long
        assert changes  # Should have recorded changes

    def test_edge_cases(self, sample_metabolites: list[dict]) -> None:
        """Generates edge case queries for each category."""
        generator = HMDBGroundTruthGenerator(sample_metabolites, seed=42)
        queries = list(generator.generate_edge_cases(greek=5, numeric=5, special=5))

        # Should find alpha-Tocopherol (Greek), 3-Hydroxybutyric acid (numeric),
        # N-Acetylcysteine and L-Alanine (special prefix)
        assert len(queries) > 0
        categories = {q.category for q in queries}
        # Should have at least some of these categories
        assert len(categories) > 0

    def test_full_dataset(self, sample_metabolites: list[dict]) -> None:
        """Generates complete dataset with all categories."""
        generator = HMDBGroundTruthGenerator(sample_metabolites, seed=42)
        dataset = generator.generate_dataset(
            exact=2, synonym=2, fuzzy=2, greek=2, numeric=2, special=2
        )

        assert len(dataset.queries) > 0
        categories = {q.category for q in dataset.queries}
        assert "exact_match" in categories
        assert "synonym_match" in categories
        assert "fuzzy_match" in categories

    def test_reproducibility(self, sample_metabolites: list[dict]) -> None:
        """Same seed produces same results."""
        gen1 = HMDBGroundTruthGenerator(sample_metabolites, seed=42)
        gen2 = HMDBGroundTruthGenerator(sample_metabolites, seed=42)

        queries1 = list(gen1.generate_exact_matches(n=3))
        queries2 = list(gen2.generate_exact_matches(n=3))

        assert len(queries1) == len(queries2)
        for q1, q2 in zip(queries1, queries2, strict=True):
            assert q1.query == q2.query
            assert q1.expected == q2.expected
