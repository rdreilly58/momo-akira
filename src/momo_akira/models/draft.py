"""Draft token generator — smallest, fastest model in the hierarchy.

Generates l_D tokens autoregressively and returns both the token IDs and the
logit value each token received at its position (needed for divergence checks).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase  # type: ignore[import-untyped]

from momo_akira.config import ModelConfig


@dataclass
class DraftOutput:
    token_ids: list[int]          # the proposed token IDs, length == n_tokens
    logits: torch.Tensor           # shape (n_tokens, vocab_size) — raw pre-softmax logits
    past_key_values: tuple[tuple[torch.Tensor, ...], ...]  # KV cache after generation


class DraftModel:
    """Wraps a small causal LM to act as the draft proposal engine."""

    def __init__(self, cfg: ModelConfig, tokenizer: PreTrainedTokenizerBase) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer
        self._device = self._resolve_device(cfg.device)
        self._dtype = self._resolve_dtype(cfg.dtype)

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            torch_dtype=self._dtype,
            device_map=cfg.device if cfg.device != "auto" else "auto",
            trust_remote_code=True,
        )
        self.model.eval()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: torch.Tensor,
        n_tokens: int,
        past_key_values: tuple[tuple[torch.Tensor, ...], ...] | None = None,
        temperature: float = 1.0,
    ) -> DraftOutput:
        """Generate n_tokens sequentially from input_ids.

        Returns the proposed token IDs, their logits at each step, and the
        updated KV cache.  Each token is sampled greedily here; the divergence
        check in the engine decides whether the choice is accepted.
        """
        device = next(self.model.parameters()).device
        ids = input_ids.to(device)
        pkv = past_key_values

        collected_ids: list[int] = []
        all_logits: list[torch.Tensor] = []

        with torch.no_grad():
            for _ in range(n_tokens):
                out = self.model(
                    input_ids=ids,
                    past_key_values=pkv,
                    use_cache=True,
                )
                logits_step = out.logits[:, -1, :]  # (1, vocab_size)
                pkv = out.past_key_values

                if temperature != 1.0 and temperature > 0:
                    scaled = logits_step / temperature
                else:
                    scaled = logits_step

                next_token = int(torch.argmax(scaled, dim=-1).item())
                collected_ids.append(next_token)
                all_logits.append(logits_step.squeeze(0))  # (vocab_size,)

                # Feed the newly generated token back for the next step
                ids = torch.tensor([[next_token]], device=device)

        return DraftOutput(
            token_ids=collected_ids,
            logits=torch.stack(all_logits, dim=0),  # (n_tokens, vocab_size)
            past_key_values=pkv,  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------
    # Tokenizer helpers (shared by engine)
    # ------------------------------------------------------------------

    def encode(self, text: str) -> torch.Tensor:
        ids = self.tokenizer.encode(text, return_tensors="pt")
        return ids.to(next(self.model.parameters()).device)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _resolve_dtype(dtype: str) -> torch.dtype:
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return mapping.get(dtype, torch.float16)

    @classmethod
    def load(cls, cfg: ModelConfig, shared_tokenizer_id: str) -> "DraftModel":
        """Load from config, validating that the model uses the shared tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(shared_tokenizer_id, trust_remote_code=True)
        return cls(cfg, tokenizer)
