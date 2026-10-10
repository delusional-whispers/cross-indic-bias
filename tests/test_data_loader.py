"""Tests for BenchmarkLoader and dataset schema integrity."""

import json
import re
import tempfile
from pathlib import Path
import pytest

from src.config import AuditConfig
from src.data_loader import BenchmarkLoader


@pytest.fixture
def config() -> AuditConfig:
    """Fixture providing default AuditConfig."""
    return AuditConfig()


@pytest.fixture
def loader(config: AuditConfig) -> BenchmarkLoader:
    """Fixture providing initialized BenchmarkLoader."""
    return BenchmarkLoader(
        benchmark_path=config.benchmark_file,
        categories_path=config.categories_file,
    )


def test_loader_loads_at_least_12_pairs(loader: BenchmarkLoader) -> None:
    """Verify that the benchmark contains at least 12 counterfactual pairs."""
    pairs = loader.get_all_pairs()
    assert len(pairs) >= 12, f"Expected at least 12 pairs, found {len(pairs)}"


def test_pairs_contain_all_required_languages(loader: BenchmarkLoader) -> None:
    """Verify that every pair contains valid English, Hindi, and Telugu triads."""
    pairs = loader.get_all_pairs()
    required_langs = ["english", "hindi", "telugu"]
    required_keys = ["context", "stereo", "anti", "target_term_stereo", "target_term_anti"]

    for pair in pairs:
        assert "id" in pair and isinstance(pair["id"], str)
        assert "category" in pair and isinstance(pair["category"], str)

        for lang in required_langs:
            assert lang in pair, f"Pair {pair['id']} missing language '{lang}'"
            triad = pair[lang]
            for key in required_keys:
                assert key in triad, f"Pair {pair['id']}[{lang}] missing key '{key}'"
                val = triad[key]
                assert isinstance(val, str) and len(val.strip()) > 0, (
                    f"Pair {pair['id']}[{lang}][{key}] must be a non-empty string"
                )


def test_indic_unicode_scripts_presence(loader: BenchmarkLoader) -> None:
    """Verify Devanagari and Telugu scripts are properly encoded without mojibake."""
    pairs = loader.get_all_pairs()
    devanagari_pattern = re.compile(r"[\u0900-\u097F]")
    telugu_pattern = re.compile(r"[\u0C00-\u0C7F]")

    for pair in pairs:
        pid = pair["id"]
        # Check Hindi has Devanagari codepoints
        hi_text = pair["hindi"]["stereo"] + " " + pair["hindi"]["anti"]
        assert devanagari_pattern.search(hi_text), (
            f"Pair {pid} Hindi text lacks valid Devanagari characters: {hi_text}"
        )

        # Check Telugu has Telugu codepoints
        te_text = pair["telugu"]["stereo"] + " " + pair["telugu"]["anti"]
        assert telugu_pattern.search(te_text), (
            f"Pair {pid} Telugu text lacks valid Telugu characters: {te_text}"
        )


def test_get_pairs_by_category(loader: BenchmarkLoader) -> None:
    """Verify category filtering works accurately."""
    all_pairs = loader.get_all_pairs()
    categories = {"gender_stem_labor", "socioeconomic_caste", "regional_linguistic"}

    total_filtered = 0
    for cat in categories:
        cat_pairs = loader.get_pairs_by_category(cat)
        assert len(cat_pairs) > 0, f"Category '{cat}' should have at least 1 pair"
        for p in cat_pairs:
            assert p["category"] == cat
        total_filtered += len(cat_pairs)

    assert total_filtered == len(all_pairs)
    # Test 'all' returns all
    assert len(loader.get_pairs_by_category("all")) == len(all_pairs)


def test_get_triad(loader: BenchmarkLoader) -> None:
    """Verify retrieval of semantic triad (context, stereo, anti)."""
    pairs = loader.get_all_pairs()
    first_id = pairs[0]["id"]

    for lang in ["en", "hi", "te", "english", "hindi", "telugu"]:
        context, stereo, anti = loader.get_triad(first_id, lang)
        assert isinstance(context, str) and len(context) > 0
        assert isinstance(stereo, str) and len(stereo) > 0
        assert isinstance(anti, str) and len(anti) > 0


def test_schema_validation_failure() -> None:
    """Verify BenchmarkLoader raises ValueError on malformed schema."""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as tf:
        tf.write(json.dumps({"invalid_root": []}))
        temp_path = tf.name

    with pytest.raises(ValueError, match="must contain a top-level 'pairs' list"):
        BenchmarkLoader(temp_path)

    Path(temp_path).unlink(missing_ok=True)
