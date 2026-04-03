"""Integration tests for the FastAPI server endpoints.

Uses httpx test client — no real API calls.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from momo_akira.cascade.pipeline import AcceptedTier, CascadeResult
from momo_akira.config import AppConfig
from momo_akira.models.base import LLMResponse
from momo_akira.server import app


def _make_cascade_result(content: str = "Test response") -> CascadeResult:
    return CascadeResult(
        response=LLMResponse(
            content=content,
            model_id="claude-haiku-test",
            provider="anthropic",
            input_tokens=10,
            output_tokens=5,
            latency_ms=150.0,
            success=True,
        ),
        accepted_tier=AcceptedTier.DRAFT,
        draft_invoked=True,
        qualifier_invoked=False,
        target_invoked=False,
        total_cost=0.0001,
        total_latency_ms=200.0,
        variant="psda",
    )


@pytest.fixture
def mock_app_state(app_config: AppConfig):
    """Patch the app state with a mock pipeline."""
    mock_pipeline = MagicMock()
    mock_pipeline.run = AsyncMock(return_value=_make_cascade_result())
    mock_pipeline.get_threshold_stats.return_value = {
        "tau_q": 0.35,
        "tau_t": 0.50,
        "draft_acceptance_rate": 0.65,
        "qualifier_acceptance_rate": 0.70,
        "total_requests": 10,
        "total_threshold_updates": 2,
    }

    mock_logger = MagicMock()
    mock_logger.log_request = MagicMock()

    from momo_akira import server

    old_config = server._state.config
    old_pipeline = server._state.pipeline
    old_logger = server._state.logger
    old_metrics = server._state.metrics

    server._state.config = app_config
    server._state.pipeline = mock_pipeline
    server._state.logger = mock_logger
    from momo_akira.metrics import MetricsCollector

    server._state.metrics = MetricsCollector()

    yield mock_pipeline

    server._state.config = old_config
    server._state.pipeline = old_pipeline
    server._state.logger = old_logger
    server._state.metrics = old_metrics


@pytest.fixture
def client(mock_app_state):
    return TestClient(app, raise_server_exceptions=True)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "tiers" in data
        assert "thresholds" in data

    def test_health_contains_model_info(self, client: TestClient) -> None:
        response = client.get("/health")
        data = response.json()
        assert "draft" in data["tiers"]
        assert "qualifier" in data["tiers"]
        assert "target" in data["tiers"]


class TestModelsEndpoint:
    def test_models_returns_list(self, client: TestClient) -> None:
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 4  # cascade + 3 tiers

    def test_cascade_model_in_list(self, client: TestClient) -> None:
        response = client.get("/v1/models")
        data = response.json()
        model_ids = [m["id"] for m in data["data"]]
        assert "cascade" in model_ids


class TestMetricsEndpoint:
    def test_metrics_returns_summary(self, client: TestClient) -> None:
        response = client.get("/v1/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        assert "adaptive_thresholds" in data


class TestChatCompletions:
    def test_basic_completion(self, client: TestClient, mock_app_state: MagicMock) -> None:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "cascade",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "Test response"

    def test_response_has_cascade_info(self, client: TestClient) -> None:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "cascade",
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        data = response.json()
        assert "cascade_info" in data
        cascade = data["cascade_info"]
        assert "accepted_tier" in cascade
        assert "variant" in cascade
        assert "tiers_invoked" in cascade
        assert "cost_usd" in cascade
        assert "latency_ms" in cascade

    def test_response_has_usage(self, client: TestClient) -> None:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "cascade",
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        data = response.json()
        assert "usage" in data
        assert "prompt_tokens" in data["usage"]
        assert "completion_tokens" in data["usage"]

    def test_streaming_not_supported(self, client: TestClient) -> None:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "cascade",
                "messages": [{"role": "user", "content": "test"}],
                "stream": True,
            },
        )
        assert response.status_code == 400

    def test_system_message_handled(self, client: TestClient, mock_app_state: MagicMock) -> None:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "cascade",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello"},
                ],
            },
        )
        assert response.status_code == 200
        # Verify pipeline was called
        mock_app_state.run.assert_called_once()

    def test_openai_compat_id_format(self, client: TestClient) -> None:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "cascade",
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        data = response.json()
        assert data["id"].startswith("chatcmpl-")
        assert data["object"] == "chat.completion"
        assert "created" in data
        assert "model" in data

    def test_pipeline_called_with_correct_messages(
        self, client: TestClient, mock_app_state: MagicMock
    ) -> None:
        messages = [{"role": "user", "content": "What is 2+2?"}]
        client.post(
            "/v1/chat/completions",
            json={"model": "cascade", "messages": messages},
        )
        call_kwargs = mock_app_state.run.call_args
        assert call_kwargs is not None
        # Messages should be passed through
        assert any("What is 2+2?" in str(v) for v in call_kwargs[1].values() or [])
