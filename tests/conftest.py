import os
import shutil
from contextlib import contextmanager

import pandas as pd
import pytest

from scripts.make_sample_data import make_sample


@contextmanager
def project_env(root):
    previous = os.environ.get("HOUSE_PRICES_ROOT")
    os.environ["HOUSE_PRICES_ROOT"] = str(root)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("HOUSE_PRICES_ROOT", None)
        else:
            os.environ["HOUSE_PRICES_ROOT"] = previous


@pytest.fixture(scope="session")
def synthetic_raw() -> pd.DataFrame:
    return make_sample(n_rows=3000, seed=11)


@pytest.fixture()
def tmp_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HOUSE_PRICES_ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture(scope="session")
def pipeline_root(tmp_path_factory, synthetic_raw):
    from house_prices import clean, config, evaluate, explain, intervals, split, train

    root = tmp_path_factory.mktemp("pipeline")
    raw_path = root / "data" / "raw" / config.RAW_FILENAME
    raw_path.parent.mkdir(parents=True)
    synthetic_raw.to_csv(raw_path, index=False)
    previous_fast = os.environ.get("HOUSE_PRICES_FAST")
    os.environ["HOUSE_PRICES_FAST"] = "1"
    try:
        with project_env(root):
            assert clean.main([]) == 0
            assert split.main([]) == 0
            assert train.main(["--fast"]) == 0
            assert evaluate.main([]) == 0
            assert intervals.main([]) == 0
            assert explain.main([]) == 0
    finally:
        if previous_fast is None:
            os.environ.pop("HOUSE_PRICES_FAST", None)
        else:
            os.environ["HOUSE_PRICES_FAST"] = previous_fast
    return root


@pytest.fixture()
def pipeline_env(pipeline_root, monkeypatch):
    monkeypatch.setenv("HOUSE_PRICES_ROOT", str(pipeline_root))
    return pipeline_root


@pytest.fixture()
def pipeline_copy(pipeline_root, tmp_path, monkeypatch):
    target = tmp_path / "copy"
    shutil.copytree(pipeline_root, target)
    monkeypatch.setenv("HOUSE_PRICES_ROOT", str(target))
    return target
