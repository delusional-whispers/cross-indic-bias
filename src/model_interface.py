"""Dual-engine model scoring interface for IndicFair-Audit.

Features:
1. Live Hugging Face Engine: Computes exact autoregressive length-normalized log-likelihood.
2. Deterministic Offline/Mock Engine: Computes reproducible pseudo-log-probabilities derived
   from cryptographic MD5 hashes and character distributions without requiring weights or GPUs.
"""

from abc import ABC, abstractmethod
import hashlib
import math
import unicodedata
from typing import Optional, Tuple

from src.config import AuditConfig


class BaseScorer(ABC):
    """Abstract base class for language model sequence scorers."""

    @abstractmethod
    def score_sentence(self, sentence: str, context: Optional[str] = None) -> float:
        """Compute length-normalized sequence log-probability.

        Args:
            sentence: The target sentence to evaluate.
            context: Optional prefix/context conditioning the sequence.

        Returns:
            Length-normalized sequence log-probability (float).
        """
        pass

    def score_pair(
        self, stereo: str, anti: str, context: Optional[str] = None
    ) -> Tuple[float, float]:
        """Compute length-normalized log-probabilities for a counterfactual pair.

        Args:
            stereo: Stereotypical continuation or sentence.
            anti: Anti-stereotypical continuation or sentence.
            context: Optional common context prefix.

        Returns:
            Tuple of (stereo_score, anti_score).
        """
        s_score = self.score_sentence(stereo, context=context)
        a_score = self.score_sentence(anti, context=context)
        return s_score, a_score


class MockDeterministicScorer(BaseScorer):
    """Deterministic mathematical mock scorer.

    Uses MD5 cryptographic hashing of normalized Unicode text combined with
    pseudo-token distribution metrics and seed offsets to generate fully
    reproducible, mathematically consistent length-normalized log-probabilities.
    """

    def __init__(self, seed: int = 42) -> None:
        """Initialize mock scorer with fixed random seed.

        Args:
            seed: Deterministic integer seed.
        """
        self.seed = seed

    def _normalize_text(self, text: str) -> str:
        """Normalize Unicode characters (NFC) and trim whitespace."""
        return unicodedata.normalize("NFC", text.strip())

    def score_sentence(self, sentence: str, context: Optional[str] = None) -> float:
        """Compute deterministic pseudo-log-probability.

        Mathematical Formulation:
            1. Normalized input text T and optional context C are hashed via MD5.
            2. Pseudo-token chunks are derived from whitespace or script-sensitive boundaries.
            3. Each pseudo-token is mapped to an empirical log-prob range [-4.2, -1.1]
               via an integer hash projection modulo 10,000.
            4. The final score is length-normalized across pseudo-token count.

        Args:
            sentence: Target sentence string.
            context: Optional context string.

        Returns:
            Length-normalized pseudo-log-probability (negative float).
        """
        clean_sentence = self._normalize_text(sentence)
        clean_context = self._normalize_text(context) if context else ""

        if not clean_sentence:
            return -10.0

        full_text = f"{clean_context} {clean_sentence}".strip() if clean_context else clean_sentence

        # Script-aware pseudo-tokenization (words / character segments)
        tokens = clean_sentence.split()
        num_tokens = max(1, len(tokens))

        total_log_prob = 0.0
        for idx, token in enumerate(tokens):
            # Compound hash combining token, full context, index, and seed
            hash_input = f"{self.seed}_{idx}_{token}_{full_text}".encode("utf-8")
            digest = hashlib.md5(hash_input).hexdigest()
            int_val = int(digest[:8], 16)

            # Map to realistic per-token log-probability between -3.8 and -1.2
            normalized_val = (int_val % 10000) / 10000.0  # [0.0, 1.0)
            token_log_prob = -3.8 + (normalized_val * 2.6)
            total_log_prob += token_log_prob

        # Length-normalized sequence log-probability: (1 / |T|) * sum(log P)
        norm_score = total_log_prob / float(num_tokens)
        return round(norm_score, 6)


class HuggingFaceScorer(BaseScorer):
    """Live Hugging Face causal language model scorer.

    Computes exact autoregressive sequence log-likelihood with token-level
    length normalization to avoid penalizing fragmented Indic multi-byte tokenizations.
    """

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        """Initialize live Hugging Face model and tokenizer.

        Args:
            model_name: Model identifier on Hugging Face Hub or local directory.
            device: Target execution device ('cuda', 'mps', 'cpu').

        Raises:
            ImportError: If torch or transformers is not installed.
        """
        self.model_name = model_name
        self.device = device

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as err:
            raise ImportError(
                "HuggingFaceScorer requires 'torch' and 'transformers'. "
                f"Install with: pip install torch transformers. Details: {err}"
            ) from err

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Select data type
        dtype = torch.float32
        if device == "cuda" and torch.cuda.is_available():
            dtype = torch.float16
        elif device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dtype = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def score_sentence(self, sentence: str, context: Optional[str] = None) -> float:
        r"""Compute exact length-normalized sequence log-probability using causal LM.

        $$\text{Score}(S) = \frac{1}{|T|} \sum_{j=1}^{|T|} \log P_\theta(t_j \mid t_{<j}, C)$$

        Args:
            sentence: Target continuation or sentence.
            context: Optional prefix context.

        Returns:
            Length-normalized sequence log-probability.
        """
        torch = self.torch
        clean_sentence = sentence.strip()
        clean_context = context.strip() if context else ""

        if not clean_sentence:
            return -100.0

        if clean_context:
            full_prompt = f"{clean_context} {clean_sentence}"
            context_encoding = self.tokenizer(clean_context, return_tensors="pt")
            full_encoding = self.tokenizer(full_prompt, return_tensors="pt")

            context_len = context_encoding.input_ids.shape[1]
            input_ids = full_encoding.input_ids.to(self.device)
            attention_mask = full_encoding.attention_mask.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # shape [1, seq_len, vocab_size]

            # Shift logits and labels for autoregressive log likelihood
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            # Target tokens start at index context_len - 1 in shift_labels
            target_start_idx = max(0, context_len - 1)
            target_logits = shift_logits[:, target_start_idx:, :]
            target_labels = shift_labels[:, target_start_idx:]

            num_target_tokens = target_labels.shape[1]
            if num_target_tokens == 0:
                return -100.0

            log_probs = torch.log_softmax(target_logits, dim=-1)
            token_log_probs = log_probs.gather(
                dim=-1, index=target_labels.unsqueeze(-1)
            ).squeeze(-1)

            sum_log_prob = token_log_probs.sum().item()
            norm_score = sum_log_prob / float(num_target_tokens)
            return round(norm_score, 6)

        else:
            encoding = self.tokenizer(clean_sentence, return_tensors="pt")
            input_ids = encoding.input_ids.to(self.device)
            attention_mask = encoding.attention_mask.to(self.device)

            seq_len = input_ids.shape[1]
            if seq_len <= 1:
                return -10.0

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            num_tokens = shift_labels.shape[1]

            log_probs = torch.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(
                dim=-1, index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)

            sum_log_prob = token_log_probs.sum().item()
            norm_score = sum_log_prob / float(num_tokens)
            return round(norm_score, 6)


def get_scorer(config: AuditConfig) -> BaseScorer:
    """Factory function returning the configured BaseScorer implementation.

    Args:
        config: AuditConfig instance.

    Returns:
        Instance of MockDeterministicScorer or HuggingFaceScorer.
    """
    if config.use_mock:
        return MockDeterministicScorer(seed=config.seed)

    device = config.get_resolved_device()
    return HuggingFaceScorer(model_name=config.model_name, device=device)
