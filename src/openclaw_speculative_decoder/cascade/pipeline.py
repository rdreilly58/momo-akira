"""Main cascade pipeline: orchestrates 3-tier speculative decoding.

Implements both PSDA (Assisted) and PSDF (Fuzzy) variants from the
PyramidSD paper, adapted for API-based LLM inference.

API Adaptation of the paper's algorithm:
  Paper operates on token-level logit distributions (l_D draft tokens).
  API adaptation operates on full response-level:

  PSDF (Fuzzy):
    1. Draft generates full response
    2. Score confidence C_draft using composite divergence proxy
    3. If C_draft >= tau_Q → return draft response (accepted)
    4. Qualifier generates response (or re-scores draft)
    5. Score confidence C_qual
    6. If C_qual >= tau_T → return qualifier response
    7. Target generates final response

  PSDA (Assisted):
    Same as PSDF but at step 4: qualifier always generates a fresh
    response when the draft is rejected (quality floor = qualifier tier).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..config import AppConfig
from ..models.base import LLMProvider, LLMRequest, LLMResponse, ProviderFactory
from .confidence import ConfidenceScore, score_response
from .threshold import AdaptiveThresholdController


class AcceptedTier(str, Enum):
    DRAFT = "draft"
    QUALIFIER = "qualifier"
    TARGET = "target"


@dataclass
class CascadeResult:
    """Result of a cascade pipeline run."""

    # The final accepted response
    response: LLMResponse
    # Which tier produced the accepted response
    accepted_tier: AcceptedTier
    # Confidence scores at each stage
    draft_confidence: ConfidenceScore | None = None
    qualifier_confidence: ConfidenceScore | None = None
    # Whether each tier was invoked
    draft_invoked: bool = False
    qualifier_invoked: bool = False
    target_invoked: bool = False
    # Total cost (USD)
    total_cost: float = 0.0
    # Total wall-clock latency (ms)
    total_latency_ms: float = 0.0
    # Cascade variant used
    variant: str = "psda"

    def to_metadata(self) -> dict[str, Any]:
        """Serialize cascade info for inclusion in API response."""
        return {
            "accepted_tier": self.accepted_tier.value,
            "variant": self.variant,
            "tiers_invoked": {
                "draft": self.draft_invoked,
                "qualifier": self.qualifier_invoked,
                "target": self.target_invoked,
            },
            "confidence": {
                "draft": round(self.draft_confidence.score, 4) if self.draft_confidence else None,
                "qualifier": round(self.qualifier_confidence.score, 4) if self.qualifier_confidence else None,
            },
            "cost_usd": round(self.total_cost, 6),
            "latency_ms": round(self.total_latency_ms, 1),
        }


class CascadePipeline:
    """Orchestrates the 3-tier speculative decoding cascade."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._threshold_controller = AdaptiveThresholdController(
            config=config.adaptive_threshold,
            initial_tau_q=config.cascade.tau_q,
            initial_tau_t=config.cascade.tau_t,
        )
        # Build providers — imported here to ensure registration side-effects
        from ..models import anthropic, openai, openrouter  # noqa: F401

        self._providers: dict[str, LLMProvider] = {}
        for tier_name in ("draft", "qualifier", "target"):
            tier_cfg = getattr(config.models, tier_name)
            provider = ProviderFactory.create(
                provider_name=tier_cfg.provider,
                api_key=config.get_api_key(tier_cfg.provider),
                base_url=getattr(
                    getattr(config.providers, tier_cfg.provider, None),
                    "base_url",
                    None,
                ),
            )
            self._providers[tier_name] = provider

    async def run(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> CascadeResult:
        """Run the full cascade pipeline and return the result."""
        start_time = time.monotonic() * 1000
        variant = self._config.cascade.variant
        extra_params = extra_params or {}

        if variant == "psdf":
            result = await self._run_psdf(
                messages, max_tokens, temperature, system, extra_params
            )
        else:
            result = await self._run_psda(
                messages, max_tokens, temperature, system, extra_params
            )

        result.total_latency_ms = time.monotonic() * 1000 - start_time
        result.variant = variant
        return result

    async def _make_request(
        self,
        tier: str,
        messages: list[dict],
        max_tokens: int | None,
        temperature: float | None,
        system: str | None,
        extra_params: dict,
        request_logprobs: bool = False,
    ) -> LLMResponse:
        """Make a request to a specific tier with timeout."""
        tier_cfg = getattr(self._config.models, tier)
        provider = self._providers[tier]

        request = LLMRequest(
            model_id=tier_cfg.model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            extra_params=extra_params,
            request_logprobs=request_logprobs and self._config.confidence.use_logprobs,
        )

        try:
            response = await asyncio.wait_for(
                provider.complete(request),
                timeout=tier_cfg.timeout_s,
            )
        except asyncio.TimeoutError:
            response = LLMResponse(
                content="",
                model_id=tier_cfg.model_id,
                provider=tier_cfg.provider,
                success=False,
                error=f"Tier '{tier}' timed out after {tier_cfg.timeout_s}s",
            )
        return response

    def _compute_cost(self, response: LLMResponse, tier: str) -> float:
        tier_cfg = getattr(self._config.models, tier)
        return response.cost(tier_cfg.cost_per_1k_input, tier_cfg.cost_per_1k_output)

    def _score(self, tier: str, messages: list[dict], response: LLMResponse) -> ConfidenceScore:
        return score_response(
            tier=tier,
            messages=messages,
            content=response.content,
            logprobs=response.logprobs,
            config=self._config.confidence,
        )

    async def _run_psdf(
        self,
        messages: list[dict],
        max_tokens: int | None,
        temperature: float | None,
        system: str | None,
        extra_params: dict,
    ) -> CascadeResult:
        """PSDF (Fuzzy) variant: fuzzy acceptance at both stages.

        Highest peak performance but higher variance.
        """
        tau_q = self._threshold_controller.tau_q
        tau_t = self._threshold_controller.tau_t
        total_cost = 0.0

        # --- Stage 1: Draft ---
        draft_resp = await self._make_request(
            "draft", messages, max_tokens, temperature, system, extra_params,
            request_logprobs=True,
        )
        draft_cost = self._compute_cost(draft_resp, "draft")
        total_cost += draft_cost

        if not draft_resp.success:
            # Draft failed — skip to qualifier
            return await self._run_from_qualifier(
                messages, max_tokens, temperature, system, extra_params,
                tau_t, total_cost, draft_invoked=True, draft_confidence=None,
            )

        draft_conf = self._score("draft", messages, draft_resp)
        draft_accepted = draft_conf.accepts_at(tau_q)
        self._threshold_controller.record_draft(draft_accepted)

        if draft_accepted:
            return CascadeResult(
                response=draft_resp,
                accepted_tier=AcceptedTier.DRAFT,
                draft_confidence=draft_conf,
                draft_invoked=True,
                total_cost=total_cost,
            )

        # --- Stage 2: Qualifier ---
        qual_resp = await self._make_request(
            "qualifier", messages, max_tokens, temperature, system, extra_params,
            request_logprobs=True,
        )
        qual_cost = self._compute_cost(qual_resp, "qualifier")
        total_cost += qual_cost

        if not qual_resp.success:
            return await self._run_target_fallback(
                messages, max_tokens, temperature, system, extra_params,
                total_cost, draft_invoked=True, qualifier_invoked=True,
                draft_confidence=draft_conf, qualifier_confidence=None,
            )

        qual_conf = self._score("qualifier", messages, qual_resp)
        qual_accepted = qual_conf.accepts_at(tau_t)
        self._threshold_controller.record_qualifier(qual_accepted)

        if qual_accepted:
            return CascadeResult(
                response=qual_resp,
                accepted_tier=AcceptedTier.QUALIFIER,
                draft_confidence=draft_conf,
                qualifier_confidence=qual_conf,
                draft_invoked=True,
                qualifier_invoked=True,
                total_cost=total_cost,
            )

        # --- Stage 3: Target ---
        target_resp = await self._make_request(
            "target", messages, max_tokens, temperature, system, extra_params,
        )
        total_cost += self._compute_cost(target_resp, "target")

        return CascadeResult(
            response=target_resp if target_resp.success else qual_resp,
            accepted_tier=AcceptedTier.TARGET if target_resp.success else AcceptedTier.QUALIFIER,
            draft_confidence=draft_conf,
            qualifier_confidence=qual_conf,
            draft_invoked=True,
            qualifier_invoked=True,
            target_invoked=True,
            total_cost=total_cost,
        )

    async def _run_psda(
        self,
        messages: list[dict],
        max_tokens: int | None,
        temperature: float | None,
        system: str | None,
        extra_params: dict,
    ) -> CascadeResult:
        """PSDA (Assisted) variant: when draft rejected, qualifier always generates fresh.

        More stable than PSDF. Quality floor = qualifier tier.
        """
        tau_q = self._threshold_controller.tau_q
        tau_t = self._threshold_controller.tau_t
        total_cost = 0.0

        # --- Stage 1: Draft ---
        draft_resp = await self._make_request(
            "draft", messages, max_tokens, temperature, system, extra_params,
            request_logprobs=True,
        )
        draft_cost = self._compute_cost(draft_resp, "draft")
        total_cost += draft_cost

        if not draft_resp.success:
            return await self._run_from_qualifier(
                messages, max_tokens, temperature, system, extra_params,
                tau_t, total_cost, draft_invoked=True, draft_confidence=None,
            )

        draft_conf = self._score("draft", messages, draft_resp)
        draft_accepted = draft_conf.accepts_at(tau_q)
        self._threshold_controller.record_draft(draft_accepted)

        if draft_accepted:
            return CascadeResult(
                response=draft_resp,
                accepted_tier=AcceptedTier.DRAFT,
                draft_confidence=draft_conf,
                draft_invoked=True,
                total_cost=total_cost,
            )

        # PSDA key difference: qualifier generates a NEW response (assisted decoding)
        # rather than just evaluating the draft.
        return await self._run_from_qualifier(
            messages, max_tokens, temperature, system, extra_params,
            tau_t, total_cost, draft_invoked=True, draft_confidence=draft_conf,
        )

    async def _run_from_qualifier(
        self,
        messages: list[dict],
        max_tokens: int | None,
        temperature: float | None,
        system: str | None,
        extra_params: dict,
        tau_t: float,
        accumulated_cost: float,
        draft_invoked: bool,
        draft_confidence: ConfidenceScore | None,
    ) -> CascadeResult:
        """Run from qualifier stage onward."""
        qual_resp = await self._make_request(
            "qualifier", messages, max_tokens, temperature, system, extra_params,
            request_logprobs=True,
        )
        qual_cost = self._compute_cost(qual_resp, "qualifier")
        accumulated_cost += qual_cost

        if not qual_resp.success:
            return await self._run_target_fallback(
                messages, max_tokens, temperature, system, extra_params,
                accumulated_cost, draft_invoked=draft_invoked, qualifier_invoked=True,
                draft_confidence=draft_confidence, qualifier_confidence=None,
            )

        qual_conf = self._score("qualifier", messages, qual_resp)
        qual_accepted = qual_conf.accepts_at(tau_t)
        self._threshold_controller.record_qualifier(qual_accepted)

        if qual_accepted:
            return CascadeResult(
                response=qual_resp,
                accepted_tier=AcceptedTier.QUALIFIER,
                draft_confidence=draft_confidence,
                qualifier_confidence=qual_conf,
                draft_invoked=draft_invoked,
                qualifier_invoked=True,
                total_cost=accumulated_cost,
            )

        return await self._run_target_fallback(
            messages, max_tokens, temperature, system, extra_params,
            accumulated_cost, draft_invoked=draft_invoked, qualifier_invoked=True,
            draft_confidence=draft_confidence, qualifier_confidence=qual_conf,
        )

    async def _run_target_fallback(
        self,
        messages: list[dict],
        max_tokens: int | None,
        temperature: float | None,
        system: str | None,
        extra_params: dict,
        accumulated_cost: float,
        draft_invoked: bool,
        qualifier_invoked: bool,
        draft_confidence: ConfidenceScore | None,
        qualifier_confidence: ConfidenceScore | None,
    ) -> CascadeResult:
        """Run the target tier as final stage."""
        target_resp = await self._make_request(
            "target", messages, max_tokens, temperature, system, extra_params,
        )
        accumulated_cost += self._compute_cost(target_resp, "target")

        # If target also fails, return best available response
        if not target_resp.success:
            # Try to use qualifier response if we have one
            fallback_resp = target_resp  # best we have
            tier = AcceptedTier.TARGET
        else:
            fallback_resp = target_resp
            tier = AcceptedTier.TARGET

        return CascadeResult(
            response=fallback_resp,
            accepted_tier=tier,
            draft_confidence=draft_confidence,
            qualifier_confidence=qualifier_confidence,
            draft_invoked=draft_invoked,
            qualifier_invoked=qualifier_invoked,
            target_invoked=True,
            total_cost=accumulated_cost,
        )

    def get_threshold_stats(self) -> dict:
        return self._threshold_controller.get_stats()
