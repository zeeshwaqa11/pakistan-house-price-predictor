import pandas as pd
import pytest

from scripts.make_sample_data import make_sample


@pytest.fixture(scope="session")
def synthetic_raw() -> pd.DataFrame:
    return make_sample(n_rows=3000, seed=11)


@pytest.fixture()
def tmp_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HOUSE_PRICES_ROOT", str(tmp_path))
    return tmp_path
