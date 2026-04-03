"""JSONL request/response logger with daily rotation.

Each log entry is a single-line JSON object containing:
  - timestamp
  - request_id
  - messages (if include_content=True)
  - cascade metadata (tier used, confidence, cost)
  - response content (if include_content=True)
  - latency
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any

from ..cascade.pipeline import CascadeResult
from ..config import LoggingConfig


class RequestLogger:
    """Thread-safe JSONL logger with daily file rotation."""

    def __init__(self, config: LoggingConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._handler: logging.Handler | None = None
        self._logger: logging.Logger | None = None

        if config.enabled:
            self._setup()

    def _setup(self) -> None:
        log_dir = Path(self._config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        log_path = log_dir / "requests.jsonl"

        handler = TimedRotatingFileHandler(
            filename=str(log_path),
            when="midnight",
            backupCount=self._config.backup_count,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        # Override rotation naming to use .jsonl extension
        handler.suffix = "%Y-%m-%d.jsonl"

        logger = logging.getLogger("openclaw.requests")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.propagate = False

        self._handler = handler
        self._logger = logger

    def log_request(
        self,
        request_id: str,
        messages: list[dict],
        result: CascadeResult,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Log a completed cascade request."""
        if not self._config.enabled or self._logger is None:
            return

        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "accepted_tier": result.accepted_tier.value,
            "variant": result.variant,
            "cost_usd": round(result.total_cost, 6),
            "latency_ms": round(result.total_latency_ms, 1),
            "tiers_invoked": {
                "draft": result.draft_invoked,
                "qualifier": result.qualifier_invoked,
                "target": result.target_invoked,
            },
            "confidence": {
                "draft": round(result.draft_confidence.score, 4) if result.draft_confidence else None,
                "qualifier": round(result.qualifier_confidence.score, 4) if result.qualifier_confidence else None,
            },
            "response_model": result.response.model_id,
            "input_tokens": result.response.input_tokens,
            "output_tokens": result.response.output_tokens,
        }

        if self._config.include_content:
            entry["messages"] = messages
            entry["response_content"] = result.response.content

        if extra:
            entry.update(extra)

        with self._lock:
            self._logger.info(json.dumps(entry, ensure_ascii=False))

    @staticmethod
    def new_request_id() -> str:
        return str(uuid.uuid4())
