"""Data loader and benchmark validation module for IndicFair-Audit."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class BenchmarkLoader:
    """Loads, validates, and manages multilingual stereotype benchmark pairs.

    Enforces strict schema compliance across English, Hindi, and Telugu triads
    with UTF-8 encoding support to prevent mojibake and truncation.
    """

    REQUIRED_TOP_LEVEL_KEYS = {"pairs"}
    REQUIRED_PAIR_KEYS = {"id", "category", "english", "hindi", "telugu"}
    REQUIRED_TRIAD_KEYS = {
        "context",
        "stereo",
        "anti",
        "target_term_stereo",
        "target_term_anti",
    }
    SUPPORTED_LANGUAGES = {"english", "hindi", "telugu", "en", "hi", "te"}
    LANG_MAP = {
        "en": "english",
        "hi": "hindi",
        "te": "telugu",
        "english": "english",
        "hindi": "hindi",
        "telugu": "telugu",
    }

    def __init__(
        self,
        benchmark_path: Union[str, Path],
        categories_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize loader with paths to benchmark JSON and categories JSON.

        Args:
            benchmark_path: Filesystem path to benchmark_pairs.json.
            categories_path: Optional filesystem path to categories.json.

        Raises:
            FileNotFoundError: If benchmark_path does not exist.
            ValueError: If JSON data is malformed or schema validation fails.
        """
        self.benchmark_path = Path(benchmark_path)
        self.categories_path = Path(categories_path) if categories_path else None

        if not self.benchmark_path.exists():
            raise FileNotFoundError(f"Benchmark file not found at: {self.benchmark_path}")

        self.raw_data: Dict[str, Any] = self._load_json(self.benchmark_path)
        self.categories: Dict[str, Any] = (
            self._load_json(self.categories_path) if self.categories_path and self.categories_path.exists() else {}
        )

        self._validate_schema(self.raw_data)
        self.pairs: List[Dict[str, Any]] = self.raw_data["pairs"]
        self._pair_index: Dict[str, Dict[str, Any]] = {p["id"]: p for p in self.pairs}

    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Safely load JSON file with explicit UTF-8 encoding."""
        with open(path, mode="r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as err:
                raise ValueError(f"Malformed JSON in file {path}: {err}") from err

    def _validate_schema(self, data: Dict[str, Any]) -> None:
        """Validate structural integrity of the benchmark data.

        Raises:
            ValueError: If required keys are missing or invalid types exist.
        """
        if not isinstance(data, dict):
            raise ValueError("Root benchmark data must be a JSON object.")

        if "pairs" not in data or not isinstance(data["pairs"], list):
            raise ValueError("Benchmark JSON must contain a top-level 'pairs' list.")

        if len(data["pairs"]) == 0:
            raise ValueError("Benchmark contains an empty 'pairs' list.")

        for idx, pair in enumerate(data["pairs"]):
            if not isinstance(pair, dict):
                raise ValueError(f"Pair at index {idx} must be a JSON dictionary.")

            missing_pair_keys = self.REQUIRED_PAIR_KEYS - set(pair.keys())
            if missing_pair_keys:
                raise ValueError(
                    f"Pair index {idx} (ID: {pair.get('id', 'UNKNOWN')}) missing required keys: {missing_pair_keys}"
                )

            pair_id = pair["id"]
            if not isinstance(pair_id, str) or not pair_id.strip():
                raise ValueError(f"Pair index {idx} has invalid non-string or empty 'id'.")

            category = pair["category"]
            if not isinstance(category, str) or not category.strip():
                raise ValueError(f"Pair '{pair_id}' has invalid or empty 'category'.")

            for lang in ["english", "hindi", "telugu"]:
                triad = pair[lang]
                if not isinstance(triad, dict):
                    raise ValueError(f"Pair '{pair_id}' language '{lang}' must be a dictionary.")

                missing_triad_keys = self.REQUIRED_TRIAD_KEYS - set(triad.keys())
                if missing_triad_keys:
                    raise ValueError(
                        f"Pair '{pair_id}' language '{lang}' missing keys: {missing_triad_keys}"
                    )

                for key in self.REQUIRED_TRIAD_KEYS:
                    val = triad[key]
                    if not isinstance(val, str) or not val.strip():
                        raise ValueError(
                            f"Pair '{pair_id}' language '{lang}' key '{key}' must be a non-empty string."
                        )

    def get_all_pairs(self) -> List[Dict[str, Any]]:
        """Return all benchmark pairs."""
        return list(self.pairs)

    def get_pairs_by_category(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Filter and return benchmark pairs by category.

        Args:
            category: Target category identifier or None for all pairs.

        Returns:
            List of matching benchmark pair dictionaries.
        """
        if category is None or category.strip() == "" or category.lower() == "all":
            return list(self.pairs)
        normalized_cat = category.strip().lower()
        return [p for p in self.pairs if p["category"].lower() == normalized_cat]

    def get_pair(self, item_id: str) -> Dict[str, Any]:
        """Retrieve single pair by its unique ID.

        Args:
            item_id: Unique pair identifier.

        Returns:
            Pair dictionary.

        Raises:
            KeyError: If item_id is not found.
        """
        if item_id not in self._pair_index:
            raise KeyError(f"Item ID '{item_id}' not found in benchmark. Available: {list(self._pair_index.keys())}")
        return self._pair_index[item_id]

    def get_triad(self, item_id: str, lang: str) -> Tuple[str, str, str]:
        """Retrieve the semantic triad (context, stereo, anti) for an item and language.

        Args:
            item_id: Unique pair identifier.
            lang: Language code or name ('en'/'english', 'hi'/'hindi', 'te'/'telugu').

        Returns:
            Tuple of (context, stereo, anti).

        Raises:
            KeyError: If item_id is not found or language is unsupported.
        """
        normalized_lang = lang.strip().lower()
        if normalized_lang not in self.LANG_MAP:
            raise KeyError(
                f"Unsupported language '{lang}'. Supported: {list(self.LANG_MAP.keys())}"
            )
        resolved_lang = self.LANG_MAP[normalized_lang]
        pair = self.get_pair(item_id)
        triad = pair[resolved_lang]
        return triad["context"], triad["stereo"], triad["anti"]

    def get_categories(self) -> Dict[str, Any]:
        """Return categories metadata dictionary."""
        return self.categories.get("categories", {})
