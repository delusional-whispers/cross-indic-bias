"""Configuration module for IndicFair-Audit research pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os


@dataclass
class AuditConfig:
    """Configuration dataclass for IndicFair-Audit experiments.

    Attributes:
        model_name: HuggingFace model identifier or local checkpoint path.
        device: Target execution device ('auto', 'cuda', 'mps', 'cpu').
        use_mock: Flag to toggle between deterministic offline mock and live HuggingFace model.
        batch_size: Evaluation batch size.
        seed: Random seed for mathematical reproducibility.
        data_dir: Path to directory containing benchmark and category JSON files.
        output_dir: Path to directory where audit logs, JSON reports, and figures are saved.
        benchmark_file: Explicit path to benchmark_pairs.json.
        categories_file: Explicit path to categories.json.
    """

    model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    device: str = "auto"
    use_mock: bool = True
    batch_size: int = 4
    seed: int = 42

    # Root directory resolution
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    benchmark_file: Path = field(init=False)
    categories_file: Path = field(init=False)

    def __post_init__(self) -> None:
        """Resolve and initialize absolute filesystem paths and directories."""
        self.data_dir = self.base_dir / "data"
        self.output_dir = self.base_dir / "output"
        self.benchmark_file = self.data_dir / "benchmark_pairs.json"
        self.categories_file = self.data_dir / "categories.json"

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def get_resolved_device(self) -> str:
        """Resolve 'auto' device string to optimal available hardware backend.

        Returns:
            Resolved device string ('cuda', 'mps', or 'cpu').
        """
        if self.device != "auto":
            return self.device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        except ImportError:
            return "cpu"
