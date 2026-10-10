#!/usr/bin/env python3
"""Command-line interface for running IndicFair-Audit benchmarks."""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict

from src.config import AuditConfig
from src.pipeline import AuditPipeline


def format_table_row(cols: list, widths: list) -> str:
    """Format a single row in an ASCII table with fixed column widths."""
    formatted_cols = [str(c).ljust(w) for c, w in zip(cols, widths)]
    return "| " + " | ".join(formatted_cols) + " |"


def print_ascii_header() -> None:
    """Print academic research header to stdout."""
    header = """
================================================================================
             INDICFAIR-AUDIT: CROSS-LINGUAL STEREOTYPE DISPARITY
       Department of Computer Science & Engineering, NIT Warangal
================================================================================
"""
    print(header.strip())


def print_results_tables(report: Dict[str, Any]) -> None:
    """Print publication-grade ASCII tables summarizing audit findings."""
    summary = report["summary"]
    metadata = report.get("metadata", {})
    sps = summary["sps"]
    p_values = summary["p_values"]
    counts = summary["counts"]
    gaps = summary["gaps"]

    print("\n[+] EVALUATION METADATA:")
    print(f"  • Model Target : {metadata.get('model_name', 'N/A')}")
    print(f"  • Execution Mode: {metadata.get('engine', 'N/A').upper()}")
    print(f"  • Random Seed   : {metadata.get('seed', 'N/A')}")
    print(f"  • Evaluated Pairs: {metadata.get('total_pairs_evaluated', summary['total_pairs'])}")
    print(f"  • Category Filter: {metadata.get('category_filter', 'all')}")

    # Table 1: Cross-Lingual SPS Table
    headers_1 = ["Language", "Total Pairs", "Stereo", "Anti", "Tie", "SPS (%)", "p-value", "Significance"]
    widths_1 = [16, 12, 8, 8, 6, 10, 10, 14]
    separator_1 = "+-" + "-+-".join(["-" * w for w in widths_1]) + "-+"

    print("\n[+] TABLE 1: CROSS-LINGUAL STEREOTYPE PREFERENCE SCORES (SPS)")
    print(separator_1)
    print(format_table_row(headers_1, widths_1))
    print(separator_1)

    lang_map = {
        "en": "English",
        "hi": "Hindi (हिन्दी)",
        "te": "Telugu (తెలుగు)",
    }

    for lang in ["en", "hi", "te"]:
        c = counts[lang]
        p = p_values[lang]
        sig_str = "p < 0.01 (**)" if p < 0.01 else ("p < 0.05 (*)" if p < 0.05 else "Not Sig. (ns)")
        row = [
            lang_map.get(lang, lang),
            c["total"],
            c["stereo"],
            c["anti"],
            c["neutral"],
            f"{sps[lang]:.2f}%",
            f"{p:.4f}",
            sig_str,
        ]
        print(format_table_row(row, widths_1))
    print(separator_1)

    # Table 2: Disparity Gaps
    headers_2 = ["Disparity Metric", "Formulation", "Gap (%)", "Interpretation"]
    widths_2 = [22, 28, 12, 30]
    separator_2 = "+-" + "-+-".join(["-" * w for w in widths_2]) + "-+"

    print("\n[+] TABLE 2: CROSS-LINGUAL DISPARITY GAPS")
    print(separator_2)
    print(format_table_row(headers_2, widths_2))
    print(separator_2)

    gap_rows = [
        ["Δ(hi - en)", "SPS(Hindi) - SPS(English)", f"{gaps['delta_hi_en']:+.2f}%", "Hindi disparity relative to English"],
        ["Δ(te - en)", "SPS(Telugu) - SPS(English)", f"{gaps['delta_te_en']:+.2f}%", "Telugu disparity relative to English"],
        ["Δ(te - hi)", "SPS(Telugu) - SPS(Hindi)", f"{gaps['delta_te_hi']:+.2f}%", "Dravidian vs Indo-Aryan disparity"],
        ["Max Disparity", "max(|Δ_ij|)", f"{gaps['max_disparity']:.2f}%", "Peak Cross-Lingual Stereotype Divergence"],
    ]
    for r in gap_rows:
        print(format_table_row(r, widths_2))
    print(separator_2)

    # Table 3: Category Breakdown
    by_cat = report.get("by_category", {})
    if by_cat:
        headers_3 = ["Domain / Category", "Pairs", "SPS(EN)", "SPS(HI)", "SPS(TE)", "Max Gap"]
        widths_3 = [30, 8, 10, 10, 10, 10]
        separator_3 = "+-" + "-+-".join(["-" * w for w in widths_3]) + "-+"

        print("\n[+] TABLE 3: SEGMENTED DOMAIN-SPECIFIC DISPARITY")
        print(separator_3)
        print(format_table_row(headers_3, widths_3))
        print(separator_3)

        for cat, data in by_cat.items():
            cat_sps = data["sps"]
            cat_gaps = data["gaps"]
            cat_row = [
                cat,
                data["num_pairs"],
                f"{cat_sps.get('en', 50.0):.1f}%",
                f"{cat_sps.get('hi', 50.0):.1f}%",
                f"{cat_sps.get('te', 50.0):.1f}%",
                f"{cat_gaps.get('max_disparity', 0.0):.1f}%",
            ]
            print(format_table_row(cat_row, widths_3))
        print(separator_3)


def main() -> int:
    """Execute CLI runner with argument parsing."""
    parser = argparse.ArgumentParser(
        description="IndicFair-Audit: Cross-Lingual Stereotype Disparity Audit CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "live"],
        default="mock",
        help="Evaluation mode: 'mock' uses deterministic mathematical scoring; 'live' loads causal Hugging Face model.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="HuggingFaceTB/SmolLM2-360M-Instruct",
        help="Hugging Face model identifier or local checkpoint path.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter evaluation to a specific category (e.g. 'gender_stem_labor', 'socioeconomic_caste', 'regional_linguistic').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory path to save audit_results.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed for reproducibility.",
    )

    args = parser.parse_args()

    print_ascii_header()

    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config = AuditConfig(
        model_name=args.model,
        use_mock=(args.mode == "mock"),
        seed=args.seed,
    )
    config.output_dir = out_dir

    print(f"[*] Initializing AuditPipeline [Engine: {args.mode.upper()}]...")
    try:
        pipeline = AuditPipeline(config=config)
    except Exception as err:
        print(f"[!] Initialization Error: {err}", file=sys.stderr)
        return 1

    print(f"[*] Executing benchmark scoring across English, Hindi, and Telugu...")
    report = pipeline.run_full_audit(category=args.category)

    # Save results to JSON
    saved_path = pipeline.save_results(report, output_path=out_dir / "audit_results.json")
    print(f"[✓] Successfully exported full audit report to: {saved_path}")

    # Print ASCII tables
    print_results_tables(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
