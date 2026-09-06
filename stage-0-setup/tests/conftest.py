from __future__ import annotations

import os
from pathlib import Path

import pytest

studio_root = os.environ.get("MOVIN_STUDIO_ROOT")


@pytest.fixture(scope="session")
def mock_root() -> Path:
    if studio_root is None:
        pytest.skip("MOVIN_STUDIO_ROOT is not set")
    return Path(studio_root) / "MOVIN_Studio_Unity/mock"
