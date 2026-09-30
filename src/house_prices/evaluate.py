import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score


def regression_metrics(y_true_log, y_pred_log) -> dict:
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    actual = np.exp(y_true_log)
    predicted = np.exp(y_pred_log)
    ape = np.abs(predicted - actual) / actual
    return {
        "mae_pkr": float(np.mean(np.abs(predicted - actual))),
        "rmse_pkr": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "mdape": float(np.median(ape)),
        "mape": float(np.mean(ape)),
        "r2_log": float(r2_score(y_true_log, y_pred_log)),
        "rmse_log": float(np.sqrt(np.mean((y_pred_log - y_true_log) ** 2))),
    }


def _mae_pkr(y_true_log, y_pred_log):
    return mean_absolute_error(np.exp(y_true_log), np.exp(y_pred_log))


def _mdape(y_true_log, y_pred_log):
    actual = np.exp(np.asarray(y_true_log, dtype=float))
    return float(np.median(np.abs(np.exp(np.asarray(y_pred_log, dtype=float)) - actual) / actual))


def make_scorers() -> dict:
    return {
        "rmse_log": "neg_root_mean_squared_error",
        "r2_log": "r2",
        "mae_pkr": make_scorer(_mae_pkr, greater_is_better=False),
        "mdape": make_scorer(_mdape, greater_is_better=False),
    }


class MedianBaseline(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        self.median_log_price_ = float(np.median(np.asarray(y, dtype=float)))
        return self

    def predict(self, X):
        return np.full(len(X), self.median_log_price_)


class LocationPPSFBaseline(BaseEstimator, RegressorMixin):
    def __init__(self, min_location_count: int = 5):
        self.min_location_count = min_location_count

    def fit(self, X, y):
        price = np.exp(np.asarray(y, dtype=float))
        frame = pd.DataFrame(
            {
                "city": X["city"].astype(str).to_numpy(),
                "location": X["location"].astype(str).to_numpy(),
                "ppsf": price / X["area_sqft"].astype(float).to_numpy(),
            }
        )
        grouped = frame.groupby(["city", "location"])["ppsf"].agg(["median", "size"])
        grouped = grouped[grouped["size"] >= self.min_location_count]
        self.location_ppsf_ = grouped["median"].to_dict()
        self.city_ppsf_ = frame.groupby("city")["ppsf"].median().to_dict()
        self.overall_ppsf_ = float(frame["ppsf"].median())
        return self

    def predict(self, X):
        cities = X["city"].astype(str).to_numpy()
        locations = X["location"].astype(str).to_numpy()
        area = X["area_sqft"].astype(float).to_numpy()
        ppsf = np.array(
            [
                self.location_ppsf_.get((c, loc), self.city_ppsf_.get(c, self.overall_ppsf_))
                for c, loc in zip(cities, locations)
            ]
        )
        return np.log(ppsf * area)


BASELINES = {
    "baseline_median": MedianBaseline,
    "baseline_location_ppsf": LocationPPSFBaseline,
}

BASELINE_LABELS = {
    "baseline_median": "Baseline: overall median price",
    "baseline_location_ppsf": "Baseline: location median price/sq ft x size",
}

MODEL_LABELS = {
    "linear_regression": "Linear Regression",
    "ridge": "Ridge",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    **BASELINE_LABELS,
}
