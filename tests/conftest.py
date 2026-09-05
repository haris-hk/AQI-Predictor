import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


@pytest.fixture(scope="session")
def daily():
    """Two years of synthetic daily features -- enough for lag-14 and 30-day rolls."""
    from tests.synthetic import make_daily_features
    return make_daily_features(days=760, seed=11)
