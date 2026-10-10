"""Tests for mathematical and statistical metrics in IndicFair-Audit."""

import pytest
from src.metrics import (
    aggregate_report,
    compute_cross_lingual_gap,
    compute_pair_verdict,
    compute_p_value,
    compute_sps,
)


def test_compute_pair_verdict() -> None:
    """Verify pair verdict logic (+1 for stereo, -1 for anti, 0 for tie)."""
    assert compute_pair_verdict(-1.5, -2.5) == 1
    assert compute_pair_verdict(-2.5, -1.5) == -1
    assert compute_pair_verdict(-1.50000001, -1.50000002, tolerance=1e-6) == 0


def test_sps_synthetic_edge_cases() -> None:
    """Verify SPS on synthetic edge cases (100%, 0%, 50%, empty)."""
    # 100% stereotypical alignment
    all_stereo = [1, 1, 1, 1, 1]
    assert compute_sps(all_stereo) == 100.0

    # 0% stereotypical alignment (100% anti)
    all_anti = [-1, -1, -1, -1]
    assert compute_sps(all_anti) == 0.0

    # Balanced 50% unbiased null
    balanced = [1, -1, 1, -1]
    assert compute_sps(balanced) == 50.0

    # Empty list falls back to unbiased 50%
    assert compute_sps([]) == 50.0

    # Intermediate ratio: 3 stereo, 1 anti = 75.0%
    mixed = [1, 1, 1, -1]
    assert compute_sps(mixed) == 75.0


def test_compute_p_value_binomial_properties() -> None:
    """Verify statistical properties of two-sided Binomial test p-value."""
    # Balanced case at n=10, k=5 -> p-value must be 1.0
    p_balanced = compute_p_value(stereo_count=5, total_count=10)
    assert p_balanced == 1.0

    # Extreme skew: 10 out of 10 -> highly significant (p < 0.01)
    p_extreme = compute_p_value(stereo_count=10, total_count=10)
    assert p_extreme < 0.01

    # Symmetry under null p=0.5: k=2 and k=8 should yield identical two-sided p-values
    p_k2 = compute_p_value(stereo_count=2, total_count=10)
    p_k8 = compute_p_value(stereo_count=8, total_count=10)
    assert pytest.approx(p_k2, rel=1e-5) == p_k8

    # Edge cases
    assert compute_p_value(0, 0) == 1.0


def test_compute_cross_lingual_gap() -> None:
    """Verify calculation of cross-lingual disparity gaps."""
    sps_dict = {
        "en": 50.0,
        "hi": 75.0,
        "te": 60.0,
    }
    gaps = compute_cross_lingual_gap(sps_dict)

    assert gaps["delta_hi_en"] == 25.0
    assert gaps["delta_te_en"] == 10.0
    assert gaps["delta_te_hi"] == -15.0
    assert gaps["max_disparity"] == 25.0


def test_aggregate_report_structure() -> None:
    """Verify aggregate_report produces complete summary and category structures."""
    raw_results = [
        {
            "id": "pair_01",
            "category": "gender_stem_labor",
            "en": {"verdict": 1, "stereo_score": -1.2, "anti_score": -2.0},
            "hi": {"verdict": 1, "stereo_score": -1.1, "anti_score": -2.3},
            "te": {"verdict": -1, "stereo_score": -2.4, "anti_score": -1.5},
        },
        {
            "id": "pair_02",
            "category": "gender_stem_labor",
            "en": {"verdict": -1, "stereo_score": -2.2, "anti_score": -1.8},
            "hi": {"verdict": 1, "stereo_score": -1.0, "anti_score": -2.5},
            "te": {"verdict": 1, "stereo_score": -1.3, "anti_score": -2.1},
        },
    ]

    report = aggregate_report(raw_results)

    assert "summary" in report
    assert "by_category" in report
    assert "itemized_results" in report

    summary = report["summary"]
    assert summary["total_pairs"] == 2
    assert summary["sps"]["en"] == 50.0
    assert summary["sps"]["hi"] == 100.0
    assert summary["sps"]["te"] == 50.0

    assert "gender_stem_labor" in report["by_category"]
    cat_summary = report["by_category"]["gender_stem_labor"]
    assert cat_summary["num_pairs"] == 2
    assert cat_summary["sps"]["hi"] == 100.0
