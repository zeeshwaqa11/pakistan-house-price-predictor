import json

import joblib
import numpy as np
import pytest

from house_prices import config, explain, features, intervals, split


def test_interval_from_log_brackets_the_estimate():
    estimate, lower, upper = intervals.interval_from_log(
        np.log([100.0, 100.0]), np.log([120.0, 80.0]), np.log([150.0, 90.0])
    )
    assert estimate.tolist() == pytest.approx([100.0, 100.0])
    assert lower.tolist() == pytest.approx([100.0, 80.0])
    assert upper.tolist() == pytest.approx([150.0, 100.0])
    assert (lower <= estimate).all() and (estimate <= upper).all()


def test_coverage_stats_handmade():
    actual = np.array([10.0, 20.0, 30.0, 40.0])
    estimate = np.array([10.0, 20.0, 30.0, 40.0])
    lower = np.array([9.0, 25.0, 25.0, 10.0])
    upper = np.array([11.0, 30.0, 35.0, 39.0])
    stats = intervals.coverage_stats(actual, estimate, lower, upper)
    assert stats["coverage"] == pytest.approx(0.5)
    assert stats["mean_width_pkr"] == pytest.approx((2 + 5 + 10 + 29) / 4)
    assert stats["n"] == 4


def test_nominal_quantiles_match_configured_level():
    assert intervals.LOWER_QUANTILE == pytest.approx(0.1)
    assert intervals.UPPER_QUANTILE == pytest.approx(0.9)


def test_interval_metrics_reported_for_both_setups(pipeline_env):
    metrics = json.loads(config.PATHS.metrics_json.read_text())
    for setup in ("random", "time"):
        block = metrics["intervals"][setup]
        assert block["level"] == 0.8
        assert 0.0 < block["overall"]["coverage"] <= 1.0
        assert block["overall"]["mean_width_pkr"] > 0
        assert {r["city"] for r in block["by_city"]} <= set(config.CITY_CENTRES)
        assert sum(r["n"] for r in block["by_city"]) == block["overall"]["n"]
    assert config.PATHS.lower_file.exists() and config.PATHS.upper_file.exists()


def test_deployed_interval_models_bracket_test_estimates(pipeline_env):
    _, test_df = split.load_split(config.DEPLOY_SETUP)
    X = features.make_inputs(test_df.head(200))
    point = joblib.load(config.PATHS.model_file)
    lower = joblib.load(config.PATHS.lower_file)
    upper = joblib.load(config.PATHS.upper_file)
    estimate, lo, hi = intervals.predict_interval(point, lower, upper, X)
    assert (estimate > 0).all()
    assert (lo <= estimate).all() and (estimate <= hi).all()


def test_readable_names_hide_pipeline_prefixes(pipeline_env):
    pipeline = joblib.load(config.PATHS.model_file)
    names = explain.feature_names(pipeline)
    readable = [explain.readable_name(n) for n in names]
    assert not any("__" in r for r in readable)
    assert "Location (encoded price level)" in readable
    assert any(r.startswith("City: ") for r in readable)
    assert {explain.group_name(n) for n in names} >= {"Location", "Size", "City", "Property type"}


def test_shap_values_add_up_to_the_model_prediction(pipeline_env):
    pipeline = joblib.load(config.PATHS.model_file)
    _, test_df = split.load_split(config.DEPLOY_SETUP)
    X = features.make_inputs(test_df.head(25))
    explainer = explain.make_explainer(pipeline)
    _, values = explain.shap_matrix(pipeline, explainer, X)
    reconstructed = explain.base_value(explainer) + values.sum(axis=1)
    np.testing.assert_allclose(reconstructed, pipeline.predict(X), atol=1e-3)


def test_explain_prediction_returns_sentence_and_figure(pipeline_env):
    pipeline = joblib.load(config.PATHS.model_file)
    _, test_df = split.load_split(config.DEPLOY_SETUP)
    X = features.make_inputs(test_df.head(1))
    result = explain.explain_prediction(pipeline, explain.make_explainer(pipeline), X, labels={"Size": "1,200 sq ft"})
    assert result["figure"] is not None
    assert result["sentence"].endswith(".")
    assert result["prediction_log"] == pytest.approx(float(pipeline.predict(X)[0]), abs=1e-3)
    assert set(result["contributions"]) >= {"Location", "Size"}


def test_plain_english_sentences():
    up = explain.plain_english({"Location": 0.4, "Size": 0.2, "Bedrooms": 0.01})
    assert up.startswith("Location added the most to this price")
    assert "followed by size" in up
    down = explain.plain_english({"Size": -0.5, "Location": 0.3})
    assert down.startswith("Size pulled this price down the most")
    assert "while location pushed it up" in down
    assert explain.plain_english({}).startswith("No feature")


def test_group_contributions_sum_one_hots():
    names = ["cat__city_Lahore", "cat__city_Karachi", "num__area_sqft", "num__log_area_sqft"]
    grouped = explain.group_contributions(names, np.array([0.1, -0.05, 0.2, 0.3]))
    assert grouped["City"] == pytest.approx(0.05)
    assert grouped["Size"] == pytest.approx(0.5)


def test_shap_summary_written_to_metrics_and_figures(pipeline_env):
    metrics = json.loads(config.PATHS.metrics_json.read_text())
    assert metrics["shap"]["top_features"]
    assert metrics["shap"]["group_importance"][0]["mean_abs_shap"] > 0
    for name in ("shap_bar", "shap_beeswarm", "shap_waterfall_example"):
        assert (config.PATHS.figures_dir / f"{name}.png").exists()
