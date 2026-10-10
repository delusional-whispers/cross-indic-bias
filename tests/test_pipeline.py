"""End-to-end execution and regression tests for AuditPipeline."""

import json
from pathlib import Path
import tempfile
import pytest

from src.config import AuditConfig
from src.pipeline import AuditPipeline


@pytest.fixture
def mock_config() -> AuditConfig:
    """Fixture providing configuration with deterministic mock engine."""
    return AuditConfig(
        use_mock=True,
        seed=42,
    )


def test_pipeline_end_to_end_mock_execution(mock_config: AuditConfig) -> None:
    """Verify that AuditPipeline executes end-to-end using MockDeterministicScorer."""
    pipeline = AuditPipeline(config=mock_config)
    report = pipeline.run_full_audit()

    assert "summary" in report
    assert "by_category" in report
    assert "itemized_results" in report
    assert "metadata" in report

    summary = report["summary"]
    assert summary["total_pairs"] >= 12

    # Check that all languages have SPS and p-values
    for lang in ["en", "hi", "te"]:
        assert lang in summary["sps"]
        assert 0.0 <= summary["sps"][lang] <= 100.0
        assert lang in summary["p_values"]
        assert 0.0 <= summary["p_values"][lang] <= 1.0

    # Check cross lingual disparity metrics
    gaps = summary["gaps"]
    assert "delta_hi_en" in gaps
    assert "delta_te_en" in gaps
    assert "delta_te_hi" in gaps
    assert "max_disparity" in gaps


def test_pipeline_reproducibility(mock_config: AuditConfig) -> None:
    """Verify that repeated runs with the same seed yield identical results."""
    pipeline1 = AuditPipeline(config=mock_config)
    report1 = pipeline1.run_full_audit()

    pipeline2 = AuditPipeline(config=mock_config)
    report2 = pipeline2.run_full_audit()

    assert report1["summary"]["sps"] == report2["summary"]["sps"]
    assert report1["summary"]["gaps"] == report2["summary"]["gaps"]

    # Verify every individual score matches exactly
    for item1, item2 in zip(report1["itemized_results"], report2["itemized_results"]):
        for lang in ["en", "hi", "te"]:
            assert item1[lang]["stereo_score"] == item2[lang]["stereo_score"]
            assert item1[lang]["anti_score"] == item2[lang]["anti_score"]
            assert item1[lang]["verdict"] == item2[lang]["verdict"]


def test_pipeline_category_filter(mock_config: AuditConfig) -> None:
    """Verify filtering by specific category in pipeline."""
    pipeline = AuditPipeline(config=mock_config)
    target_category = "gender_stem_labor"
    report = pipeline.run_full_audit(category=target_category)

    for item in report["itemized_results"]:
        assert item["category"] == target_category

    assert len(report["itemized_results"]) > 0


def test_pipeline_save_and_reload(mock_config: AuditConfig) -> None:
    """Verify saving results to JSON file and ensuring Indic UTF-8 preservation."""
    pipeline = AuditPipeline(config=mock_config)
    report = pipeline.run_full_audit()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "test_report.json"
        saved_path = pipeline.save_results(report, output_path=out_file)

        assert saved_path.exists()

        with open(saved_path, mode="r", encoding="utf-8") as f:
            reloaded = json.load(f)

        assert reloaded["summary"]["total_pairs"] == report["summary"]["total_pairs"]
        # Verify non-ASCII Indic scripts in saved JSON
        hindi_stereo = reloaded["itemized_results"][0]["hi"]["stereo_sentence"]
        telugu_stereo = reloaded["itemized_results"][0]["te"]["stereo_sentence"]
        assert any(ord(c) > 127 for c in hindi_stereo)
        assert any(ord(c) > 127 for c in telugu_stereo)
