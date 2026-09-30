import inspect
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from house_prices import config, evaluate, features, split, train


def test_metrics_on_handmade_values():
    y_true = np.log([100.0, 200.0, 400.0])
    y_pred = np.log([110.0, 180.0, 400.0])
    m = evaluate.regression_metrics(y_true, y_pred)
    assert m["mae_pkr"] == pytest.approx((10 + 20 + 0) / 3)
    assert m["rmse_pkr"] == pytest.approx(np.sqrt((100 + 400 + 0) / 3))
    assert m["mape"] == pytest.approx((0.1 + 0.1 + 0.0) / 3)
    assert m["mdape"] == pytest.approx(0.1)
    assert m["r2_log"] < 1.0
    perfect = evaluate.regression_metrics(y_true, y_true)
    assert perfect["r2_log"] == pytest.approx(1.0)
    assert perfect["mae_pkr"] == 0.0


def test_median_baseline_predicts_training_median():
    y = np.log([1.0, 10.0, 100.0])
    model = evaluate.MedianBaseline().fit(pd.DataFrame({"a": [1, 2, 3]}), y)
    assert np.exp(model.predict(pd.DataFrame({"a": [1, 2]}))).tolist() == pytest.approx([10.0, 10.0])


def test_location_baseline_uses_location_then_city_then_overall():
    train_df = pd.DataFrame(
        {
            "city": ["Lahore"] * 6 + ["Karachi"] * 6,
            "location": ["A"] * 6 + ["B"] * 6,
            "area_sqft": [1000.0] * 12,
        }
    )
    price = np.array([10_000_000.0] * 6 + [4_000_000.0] * 6)
    model = evaluate.LocationPPSFBaseline(min_location_count=5).fit(train_df, np.log(price))
    query = pd.DataFrame(
        {
            "city": ["Lahore", "Lahore", "Karachi", "Multan"],
            "location": ["A", "Unknown", "B", "Z"],
            "area_sqft": [2000.0, 1000.0, 500.0, 1000.0],
        }
    )
    out = np.exp(model.predict(query))
    assert out[0] == pytest.approx(20_000_000)
    assert out[1] == pytest.approx(10_000_000)
    assert out[2] == pytest.approx(2_000_000)
    assert out[3] == pytest.approx(7_000_000)


def test_baseline_ignores_thin_locations():
    df = pd.DataFrame({"city": ["Lahore"] * 6, "location": ["A"] * 5 + ["B"], "area_sqft": [1000.0] * 6})
    price = np.array([10_000_000.0] * 5 + [90_000_000.0])
    model = evaluate.LocationPPSFBaseline(min_location_count=5).fit(df, np.log(price))
    assert ("Lahore", "B") not in model.location_ppsf_


def test_search_spaces_cover_every_model_and_serialise():
    assert set(train.SEARCH_SPACES) == set(train.MODEL_NAMES)
    json.dumps(train.SEARCH_SPACES)
    for spec in train.SEARCH_SPACES["lightgbm"].values():
        assert train.to_distribution(spec).rvs(random_state=0) is not None


def test_training_never_loads_the_test_split():
    source = inspect.getsource(train)
    assert "load_split" not in source
    assert "load_train" in source
    assert "test" not in [p for p in inspect.signature(train.tune_and_fit).parameters]


def test_train_pipeline_outputs_exist(pipeline_env):
    for path in (config.PATHS.model_file, config.PATHS.metadata_file, config.PATHS.location_index):
        assert path.exists()
    for setup in split.SETUPS:
        assert (config.PATHS.candidates_dir / f"{setup}_summary.json").exists()
        for name in train.MODEL_NAMES:
            assert (config.PATHS.candidates_dir / f"{setup}_{name}.joblib").exists()


def test_metadata_contents(pipeline_env):
    meta = json.loads(config.PATHS.metadata_file.read_text())
    for key in (
        "training_date",
        "libraries",
        "features",
        "parameters",
        "cv_scores",
        "data_rows_raw",
        "data_rows_clean",
        "data_file_sha256",
        "marla_sqft",
        "synthetic_data",
    ):
        assert key in meta
    assert meta["synthetic_data"] is True
    assert meta["model"] in train.TREE_MODELS
    assert len(meta["data_file_sha256"]) == 64
    assert meta["features"]["input_columns"] == features.INPUT_COLUMNS


def test_deployed_model_is_the_best_tree_model_by_cv(pipeline_env):
    payload = json.loads((config.PATHS.candidates_dir / "random_summary.json").read_text())
    summaries = payload["summaries"]
    best = min(train.TREE_MODELS, key=lambda n: summaries[n]["cv"]["rmse_log"]["mean"])
    assert json.loads(config.PATHS.metadata_file.read_text())["model"] == best


def test_summaries_include_baselines_and_ablation(pipeline_env):
    payload = json.loads((config.PATHS.candidates_dir / "random_summary.json").read_text())
    summaries = payload["summaries"]
    assert "baseline_median" in summaries and "baseline_location_ppsf" in summaries
    assert set(summaries["encoding_ablation"]) == {"target_encoding", "one_hot"}
    for name in train.MODEL_NAMES:
        cv = summaries[name]["cv"]
        assert cv["rmse_log"]["std"] >= 0
        assert summaries[name]["cv_folds"] == 3


def test_models_beat_the_median_baseline_in_cv(pipeline_env):
    payload = json.loads((config.PATHS.candidates_dir / "random_summary.json").read_text())
    summaries = payload["summaries"]
    base = summaries["baseline_median"]["cv"]["rmse_log"]["mean"]
    for name in train.MODEL_NAMES:
        assert summaries[name]["cv"]["rmse_log"]["mean"] < base


def test_changing_test_rows_does_not_change_the_trained_model(pipeline_copy):
    listings = pd.read_parquet(config.PATHS.listings)
    splits = pd.read_parquet(config.PATHS.splits)
    test_ids = splits.loc[splits["random_split"] == "test", "listing_id"]
    is_test = listings["listing_id"].isin(test_ids)
    reference = joblib.load(config.PATHS.candidates_dir / "random_ridge.joblib")
    sample = features.make_inputs(listings.loc[~is_test].head(50))
    before = reference.predict(sample)
    altered = listings.copy()
    altered.loc[is_test, "price"] = altered.loc[is_test, "price"] * 37
    altered.to_parquet(config.PATHS.listings, index=False)
    train.train_setup("random", fast=True, models=["ridge"], max_rows=None, deploy=False)
    after = joblib.load(config.PATHS.candidates_dir / "random_ridge.joblib").predict(sample)
    np.testing.assert_allclose(before, after)
