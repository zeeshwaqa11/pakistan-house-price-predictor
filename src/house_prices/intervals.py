import json
import sys

import joblib
import lightgbm
import numpy as np
import pandas as pd

from house_prices import config, features, split
from house_prices.evaluate import best_tree_model, update_metrics

LOWER_QUANTILE = (1 - config.INTERVAL_LEVEL) / 2
UPPER_QUANTILE = 1 - LOWER_QUANTILE


def make_quantile_pipeline(alpha: float, params: dict | None = None):
    model = lightgbm.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
        verbose=-1,
        subsample_freq=1,
    )
    pipeline = features.build_pipeline(model)
    if params:
        pipeline.set_params(**params)
    return pipeline


def fit_quantile_models(train_df: pd.DataFrame, params: dict | None = None):
    X, y = features.make_inputs(train_df), features.make_target(train_df)
    lower = make_quantile_pipeline(LOWER_QUANTILE, params).fit(X, y)
    upper = make_quantile_pipeline(UPPER_QUANTILE, params).fit(X, y)
    return lower, upper


def interval_from_log(point_log, lower_log, upper_log) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    estimate = np.exp(np.asarray(point_log, dtype=float))
    lower = np.minimum(np.exp(np.asarray(lower_log, dtype=float)), estimate)
    upper = np.maximum(np.exp(np.asarray(upper_log, dtype=float)), estimate)
    return estimate, lower, upper


def predict_interval(point_model, lower_model, upper_model, X: pd.DataFrame):
    return interval_from_log(point_model.predict(X), lower_model.predict(X), upper_model.predict(X))


def coverage_stats(actual, estimate, lower, upper) -> dict:
    actual = np.asarray(actual, dtype=float)
    inside = (actual >= lower) & (actual <= upper)
    width = np.asarray(upper) - np.asarray(lower)
    return {
        "n": int(len(actual)),
        "coverage": float(inside.mean()),
        "mean_width_pkr": float(width.mean()),
        "median_width_pkr": float(np.median(width)),
        "mean_relative_width": float((width / np.asarray(estimate)).mean()),
    }


def coverage_by_city(test_df: pd.DataFrame, estimate, lower, upper) -> list[dict]:
    rows = []
    frame = pd.DataFrame(
        {
            "city": test_df["city"].to_numpy(),
            "actual": test_df["price"].to_numpy(),
            "estimate": estimate,
            "lower": lower,
            "upper": upper,
        }
    )
    for city, sub in frame.groupby("city"):
        rows.append({"city": city, **coverage_stats(sub["actual"], sub["estimate"], sub["lower"], sub["upper"])})
    return rows


def main(argv: list[str] | None = None) -> int:
    if not (config.PATHS.candidates_dir / "random_summary.json").exists():
        print("Trained models not found. Run: python -m house_prices.train", file=sys.stderr)
        return 1
    results = {}
    for setup in split.SETUPS:
        train_df, test_df = split.load_split(setup)
        summaries = json.loads((config.PATHS.candidates_dir / f"{setup}_summary.json").read_text())["summaries"]
        params = summaries["lightgbm"]["best_params"]
        lower, upper = fit_quantile_models(train_df, params)
        point = joblib.load(config.PATHS.candidates_dir / f"{setup}_{best_tree_model(summaries)}.joblib")
        estimate, lo, hi = predict_interval(point, lower, upper, features.make_inputs(test_df))
        overall = coverage_stats(test_df["price"], estimate, lo, hi)
        results[setup] = {
            "level": config.INTERVAL_LEVEL,
            "method": f"LightGBM quantile regression ({LOWER_QUANTILE:.0%} and {UPPER_QUANTILE:.0%})",
            "overall": overall,
            "by_city": coverage_by_city(test_df, estimate, lo, hi),
        }
        print(
            f"[{setup}] {config.INTERVAL_LEVEL:.0%} interval: coverage {overall['coverage']:.1%}, "
            f"mean width PKR {overall['mean_width_pkr']:,.0f}"
        )
        if setup == config.DEPLOY_SETUP:
            joblib.dump(lower, config.PATHS.lower_file)
            joblib.dump(upper, config.PATHS.upper_file)
    update_metrics({"intervals": results})
    print(f"Wrote interval models and updated {config.PATHS.metrics_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
