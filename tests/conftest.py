"""Shared fixtures. FULLY IMPLEMENTED."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = PROJECT_ROOT / "driftwatch"


@pytest.fixture(autouse=True)
def _fake_backend_env(monkeypatch):
    """Settings has no default base_url/api_key by design, so tests supply fakes.

    Deliberately not a real URL: nothing in the test suite may touch a network.
    """
    monkeypatch.setenv("DRIFTWATCH_BASE_URL", "http://test.invalid/v1")
    monkeypatch.setenv("DRIFTWATCH_API_KEY", "test-key-not-real")
    from driftwatch.settings import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings():
    from driftwatch.settings import get_settings

    return get_settings()


@pytest.fixture
def sample_query():
    from driftwatch.corpus.schemas import Query

    return Query(
        query_id="gsm8k_00042",
        slice="gsm8k",
        prompt="Janet has 3 apples and buys 5 more. How many does she have?",
        ground_truth="8",
        label_space=None,
        split="calib",
    )


@pytest.fixture
def sample_completion():
    from driftwatch.corpus.schemas import Completion

    return Completion(
        query_id="gsm8k_00042",
        sample_idx=0,
        seed=123456,
        condition="healthy",
        model_id="test-model",
        digest="a" * 64,
        digest_source="backend",
        text="She has 8 apples.",
        finish_reason="stop",
        parse_ok=True,
        submitted_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        returned_at=datetime(2026, 9, 1, 12, tzinfo=timezone.utc),
    )


def python_files(root: Path) -> list[Path]:
    """Every .py file under `root`."""
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
