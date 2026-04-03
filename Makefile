.PHONY: install dev test lint fmt typecheck clean run

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	pre-commit install

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

test:
	pytest

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-cov:
	pytest --cov=src/openclaw_speculative_decoder --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------

lint:
	ruff check src/ tests/

fmt:
	ruff format src/ tests/

typecheck:
	mypy src/

check: lint typecheck

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

run:
	openclaw-decoder --config config.yaml

run-dev:
	openclaw-decoder --config config.yaml --reload --log-level debug

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
