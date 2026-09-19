from __future__ import annotations

from datetime import UTC, datetime

import pytest


@pytest.fixture
def t0() -> datetime:
    return datetime(2026, 1, 3, 11, 0, 0, tzinfo=UTC)
