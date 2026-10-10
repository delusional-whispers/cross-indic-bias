"""IndicFair-Audit: Auditing Cross-Lingual Stereotype Disparity in LLMs.

A research framework for measuring and analyzing stereotypical bias
disparities across English, Hindi, and Telugu representations in Large Language Models.
"""

from src.config import AuditConfig
from src.data_loader import BenchmarkLoader
from src.model_interface import BaseScorer, HuggingFaceScorer, MockDeterministicScorer, get_scorer
from src.metrics import (
    compute_pair_verdict,
    compute_sps,
    compute_p_value,
    compute_cross_lingual_gap,
    aggregate_report,
)
from src.pipeline import AuditPipeline

__version__ = "1.0.0"
__all__ = [
    "AuditConfig",
    "BenchmarkLoader",
    "BaseScorer",
    "HuggingFaceScorer",
    "MockDeterministicScorer",
    "get_scorer",
    "compute_pair_verdict",
    "compute_sps",
    "compute_p_value",
    "compute_cross_lingual_gap",
    "aggregate_report",
    "AuditPipeline",
]
