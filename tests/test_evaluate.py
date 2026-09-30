import json

import joblib
import numpy as np
import pandas as pd
import pytest

from house_prices import config, evaluate, features, split, train


@pytest.fixture()
def metrics(pipeline_env):
    return json.loads(config.PATHS.metrics_json.read_text())


def test_metrics_json_structure(metrics):
    assert set(metrics["setups"]) == {"random", "time"}
    assert metrics["meta"]["synthetic_data"] is True
    for payload in metrics["setups"].values():
        assert set(payload["models"]) == set(evaluate.MODEL_ORDER)
        assert payload["chosen_model"] in evaluate.TREE_MODELS
        for entry in payload["models"].values():
            for key in ("mae_pkr", "rmse_pkr", "mdape", "mape", "r2_log"):
                assert key in entry["test"]
            assert "rmse_log" in entry["cv"] and "std" in entry["cv"]["rmse_log"]


def test_time_split_test_period_is_after_training_period(metrics):
    payload = metrics["setups"]["time"]
    assert payload["train_date_range"][1] < payload["test_date_range"][0]


def test_models_beat_the_median_baseline_on_test(metrics):
    for payload in metrics["setups"].values():
        chosen = payload["models"][payload["chosen_model"]]["test"]
        assert chosen["mdape"] < payload["models"]["baseline_median"]["test"]["mdape"]
        assert "mdape" in payload["models"]["baseline_location_ppsf"]["test"]


def test_reported_numbers_can_be_recomputed_from_the_saved_model(metrics):
    for setup in split.SETUPS:
        train_df, test_df = split.load_split(setup)
        chosen = metrics["setups"][setup]["chosen_model"]
        model = joblib.load(config.PATHS.candidates_dir / f"{setup}_{chosen}.joblib")
        pred = model.predict(features.make_inputs(test_df))
        recomputed = evaluate.regression_metrics(features.make_target(test_df), pred)
        reported = metrics["setups"][setup]["models"][chosen]["test"]
        for key, value in recomputed.items():
            if key in reported:
                assert reported[key] == pytest.approx(value)


def test_error_groups_cover_the_whole_test_set(metrics):
    for payload in metrics["setups"].values():
        analysis = payload["error_analysis"]
        for key in ("by_city", "by_property_type", "by_price_decile", "by_location_listings"):
            assert sum(r["n"] for r in analysis[key]) == payload["test_rows"]
        assert len(analysis["by_price_decile"]) == 10
        assert analysis["weakest"]


def test_worst_predictions_are_sorted_and_explained(metrics):
    worst = metrics["setups"]["random"]["error_analysis"]["worst_10"]
    assert len(worst) == 10
    apes = [w["ape"] for w in worst]
    assert apes == sorted(apes, reverse=True)
    assert all(w["likely_reason"] and "guess" in w["likely_reason"] for w in worst)


def test_figures_written(pipeline_env):
    for name in (
        "model_comparison",
        "predicted_vs_actual_random",
        "predicted_vs_actual_time",
        "residuals_random",
        "error_by_group_random",
        "error_by_group_time",
    ):
        assert (config.PATHS.figures_dir / f"{name}.png").exists()


def test_group_errors_on_handmade_frame():
    frame = pd.DataFrame(
        {
            "city": ["A", "A", "B"],
            "ape": [0.1, 0.3, 0.5],
            "error": [10.0, -30.0, 50.0],
            "log_ratio": [np.log(1.1), np.log(0.7), np.log(1.5)],
        }
    )
    rows = {r["group"]: r for r in evaluate.group_errors(frame, "city")}
    assert rows["A"]["n"] == 2
    assert rows["A"]["mdape"] == pytest.approx(0.2)
    assert rows["A"]["mae_pkr"] == pytest.approx(20.0)
    assert rows["B"]["bias_pct"] == pytest.approx(0.5)


def _row(**kw):
    base = {
        "actual_ppsf": 10_000.0,
        "location_ppsf_train": 10_000.0,
        "area_sqft": 2000.0,
        "bedrooms": 4,
        "n_train": 100,
    }
    base.update(kw)
    return pd.Series(base)


def test_guess_cause_rules():
    assert "extra digit" in evaluate.guess_cause(_row(actual_ppsf=50_000.0))
    assert "missing digit" in evaluate.guess_cause(_row(actual_ppsf=2_000.0))
    assert "unit mix-up" in evaluate.guess_cause(_row(area_sqft=40_000.0))
    assert "unit mix-up" in evaluate.guess_cause(_row(area_sqft=300.0, bedrooms=5))
    assert "rare or unseen" in evaluate.guess_cause(_row(n_train=2))
    assert "no obvious data problem" in evaluate.guess_cause(_row()).lower()


def test_location_bands_are_ordered_and_cover_zero():
    bins, labels = evaluate.location_band_labels()
    assert len(bins) == len(labels) + 1
    assert labels[0] == "not in training data"


def test_update_metrics_merges(tmp_root):
    evaluate.update_metrics({"a": 1})
    evaluate.update_metrics({"b": 2})
    assert json.loads(config.PATHS.metrics_json.read_text()) == {"a": 1, "b": 2}
    assert evaluate.read_metrics() == {"a": 1, "b": 2}


def test_cv_summary_present_for_every_model_in_train_output(pipeline_env):
    payload = json.loads((config.PATHS.candidates_dir / "time_summary.json").read_text())
    assert set(train.MODEL_NAMES) <= set(payload["summaries"])
