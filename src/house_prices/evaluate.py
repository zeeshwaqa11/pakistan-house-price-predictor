import json
import sys
from datetime import datetime, timezone

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score

from house_prices import config, features, split
from house_prices.formatting import format_pkr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

TREE_MODELS = ["random_forest", "xgboost", "lightgbm"]


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

MODEL_ORDER = list(BASELINES) + ["linear_regression", "ridge", "random_forest", "xgboost", "lightgbm"]


def update_metrics(updates: dict) -> dict:
    path = config.PATHS.metrics_json
    path.parent.mkdir(parents=True, exist_ok=True)
    current = json.loads(path.read_text()) if path.exists() else {}
    current.update(updates)
    path.write_text(json.dumps(current, indent=2))
    return current


def read_metrics() -> dict | None:
    path = config.PATHS.metrics_json
    return json.loads(path.read_text()) if path.exists() else None


def location_band_labels() -> tuple[list, list]:
    low = config.MIN_LOCATION_COUNT
    bins = [-1, 0, low - 1, 49, 199, np.inf]
    labels = [
        "not in training data",
        f"1-{low - 1} listings (grouped as Other)",
        f"{low}-49 listings",
        "50-199 listings",
        "200+ listings",
    ]
    return bins, labels


def build_error_frame(test_df: pd.DataFrame, train_df: pd.DataFrame, y_pred_log: np.ndarray) -> pd.DataFrame:
    counts = train_df.groupby(["city", "location"]).size().rename("n_train")
    frame = test_df.merge(counts.reset_index(), on=["city", "location"], how="left")
    frame["n_train"] = frame["n_train"].fillna(0).astype(int)
    frame["predicted"] = np.exp(y_pred_log)
    frame["actual"] = frame["price"].astype(float)
    frame["error"] = frame["predicted"] - frame["actual"]
    frame["ape"] = frame["error"].abs() / frame["actual"]
    frame["log_ratio"] = np.log(frame["predicted"] / frame["actual"])
    frame["price_decile"] = pd.qcut(frame["actual"], 10, labels=False, duplicates="drop") + 1
    bins, labels = location_band_labels()
    frame["location_band"] = pd.cut(frame["n_train"], bins=bins, labels=labels).astype(str)
    ppsf_train = (train_df["price"] / train_df["area_sqft"]).groupby([train_df["city"], train_df["location"]]).median()
    frame = frame.merge(ppsf_train.rename("location_ppsf_train").reset_index(), on=["city", "location"], how="left")
    frame["actual_ppsf"] = frame["actual"] / frame["area_sqft"]
    return frame


def group_errors(frame: pd.DataFrame, by: str, order: list | None = None) -> list[dict]:
    rows = []
    for key, sub in frame.groupby(by, observed=True):
        rows.append(
            {
                "group": str(key),
                "n": int(len(sub)),
                "mdape": float(sub["ape"].median()),
                "mape": float(sub["ape"].mean()),
                "mae_pkr": float(sub["error"].abs().mean()),
                "bias_pct": float(np.exp(sub["log_ratio"].median()) - 1),
            }
        )
    if order:
        rows.sort(key=lambda r: order.index(r["group"]) if r["group"] in order else len(order))
    return rows


def guess_cause(row: pd.Series) -> str:
    reasons = []
    ratio = row["actual_ppsf"] / row["location_ppsf_train"] if row["location_ppsf_train"] > 0 else np.nan
    if np.isfinite(ratio):
        if ratio > 3:
            reasons.append(
                f"asking price per sq ft is {ratio:.1f}x the usual for this location, possibly an extra digit "
                "or a premium listing"
            )
        elif ratio < 1 / 3:
            reasons.append(
                f"asking price per sq ft is {1 / ratio:.1f}x below the usual for this location, possibly a "
                "missing digit or a price quoted per unit"
            )
    per_bed = row["area_sqft"] / max(row["bedrooms"], 1)
    if per_bed > 2500:
        reasons.append("very large area per bedroom, possibly an area unit mix-up")
    elif per_bed < 150:
        reasons.append("very small area per bedroom, possibly an area unit mix-up or overstated bedrooms")
    if row["n_train"] < config.MIN_LOCATION_COUNT:
        reasons.append("location is rare or unseen in training data")
    if not reasons:
        reasons.append("no obvious data problem; likely genuine variation the inputs cannot capture")
    return "; ".join(reasons).capitalize() + " (an automatic guess)"


def worst_predictions(frame: pd.DataFrame, n: int = 10) -> list[dict]:
    worst = frame.sort_values("ape", ascending=False).head(n)
    out = []
    for _, row in worst.iterrows():
        out.append(
            {
                "city": row["city"],
                "location": row["location"],
                "property_type": row["property_type"],
                "area_sqft": float(row["area_sqft"]),
                "bedrooms": int(row["bedrooms"]),
                "baths": int(row["baths"]),
                "actual_price": float(row["actual"]),
                "predicted_price": float(row["predicted"]),
                "ape": float(row["ape"]),
                "actual_ppsf": float(row["actual_ppsf"]),
                "location_median_ppsf_train": None
                if pd.isna(row["location_ppsf_train"])
                else float(row["location_ppsf_train"]),
                "location_train_listings": int(row["n_train"]),
                "likely_reason": guess_cause(row),
            }
        )
    return out


def weakest_findings(analysis: dict, overall: dict) -> list[str]:
    findings = []
    labels = {
        "by_city": "city",
        "by_property_type": "property type",
        "by_price_decile": "price decile",
        "by_location_listings": "location coverage",
    }
    for key, label in labels.items():
        rows = [r for r in analysis[key] if r["n"] >= config.MIN_GROUP_FOR_ERROR_TABLE]
        if not rows:
            continue
        worst = max(rows, key=lambda r: r["mdape"])
        best = min(rows, key=lambda r: r["mdape"])
        findings.append(
            f"By {label}, the weakest group is '{worst['group']}' with a median error of {worst['mdape']:.1%} "
            f"(n={worst['n']}) against {overall['mdape']:.1%} overall; the strongest is '{best['group']}' at "
            f"{best['mdape']:.1%}."
        )
    return findings


def error_analysis(frame: pd.DataFrame, overall: dict) -> dict:
    bins, labels = location_band_labels()
    decile_order = [str(i) for i in range(1, 11)]
    analysis = {
        "by_city": group_errors(frame, "city"),
        "by_property_type": group_errors(frame, "property_type"),
        "by_price_decile": group_errors(
            frame.assign(price_decile=frame["price_decile"].astype(int).astype(str)),
            "price_decile",
            decile_order,
        ),
        "by_location_listings": group_errors(frame, "location_band", labels),
    }
    analysis["weakest"] = weakest_findings(analysis, overall)
    analysis["worst_10"] = worst_predictions(frame)
    return analysis


def _price_axis(ax, axis="x"):
    formatter = plt.FuncFormatter(lambda v, _: format_pkr(v).replace("PKR ", ""))
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(formatter)


def plot_predicted_vs_actual(frame, setup, label):
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.scatter(frame["actual"], frame["predicted"], s=6, alpha=0.3, color="#4c78a8")
    low, high = frame["actual"].min(), frame["actual"].max()
    ax.plot([low, high], [low, high], color="#e45756", lw=1.2, label="Perfect prediction")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Actual asking price (PKR, log scale)")
    ax.set_ylabel("Predicted price (PKR, log scale)")
    ax.set_title(f"Predicted vs actual, {label} ({setup} split, test set)")
    ax.legend()
    _price_axis(ax, "x")
    _price_axis(ax, "y")
    return fig


def plot_residuals(frame, setup, label):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(frame["predicted"], frame["log_ratio"], s=6, alpha=0.3, color="#4c78a8")
    ax.axhline(0, color="#e45756", lw=1.2)
    bins = pd.qcut(frame["predicted"], 15, duplicates="drop")
    trend = frame.groupby(bins, observed=True).agg(x=("predicted", "median"), y=("log_ratio", "median"))
    ax.plot(trend["x"], trend["y"], color="black", lw=1.8, label="Median residual by predicted price")
    ax.set_xscale("log")
    ax.set_xlabel("Predicted price (PKR, log scale)")
    ax.set_ylabel("log(predicted / actual)")
    ax.set_title(f"Residuals vs predicted, {label} ({setup} split)")
    ax.legend()
    _price_axis(ax, "x")
    return fig


def plot_model_comparison(setups: dict):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    rows = []
    for setup, payload in setups.items():
        for name in MODEL_ORDER:
            if name in payload["models"]:
                test = payload["models"][name]["test"]
                rows.append(
                    {
                        "model": MODEL_LABELS[name],
                        "setup": setup,
                        "mdape": test["mdape"] * 100,
                        "r2_log": test["r2_log"],
                    }
                )
    data = pd.DataFrame(rows)
    order = [MODEL_LABELS[n] for n in MODEL_ORDER if MODEL_LABELS[n] in set(data["model"])]
    sns.barplot(data=data, x="mdape", y="model", hue="setup", order=order, ax=axes[0])
    axes[0].set_xlabel("Median absolute percentage error (%), lower is better")
    axes[0].set_ylabel("")
    sns.barplot(data=data, x="r2_log", y="model", hue="setup", order=order, ax=axes[1])
    axes[1].set_xlabel("R² on log price, higher is better")
    axes[1].set_ylabel("")
    axes[1].set_yticklabels([])
    axes[1].set_xlim(min(0, data["r2_log"].min() - 0.05), 1)
    fig.suptitle("Test-set comparison, random vs time split")
    fig.tight_layout()
    return fig


def plot_error_by_group(analysis, setup, label):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    panels = [
        ("by_city", "City"),
        ("by_property_type", "Property type"),
        ("by_price_decile", "Price decile (1 = cheapest)"),
        ("by_location_listings", "Training listings in the location"),
    ]
    for ax, (key, title) in zip(axes.ravel(), panels):
        rows = pd.DataFrame(analysis[key])
        sns.barplot(data=rows, x="mdape", y="group", ax=ax, color="#4c78a8")
        for i, r in rows.iterrows():
            ax.text(r["mdape"], i, f"  n={r['n']}", va="center", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("Median absolute percentage error")
        ax.set_ylabel("")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:g}%"))
    fig.suptitle(f"Where the model is weakest: {label} ({setup} split)")
    fig.tight_layout()
    return fig


def best_tree_model(summaries: dict) -> str:
    return min((n for n in TREE_MODELS if n in summaries), key=lambda n: summaries[n]["cv"]["rmse_log"]["mean"])


def evaluate_setup(setup: str) -> tuple[dict, pd.DataFrame]:
    train_df, test_df = split.load_split(setup)
    summary_path = config.PATHS.candidates_dir / f"{setup}_summary.json"
    payload = json.loads(summary_path.read_text())
    summaries = payload["summaries"]
    X_train, y_train = features.make_inputs(train_df), features.make_target(train_df)
    X_test, y_test = features.make_inputs(test_df), features.make_target(test_df)
    predictions = {}
    models = {}
    for name in MODEL_ORDER:
        if name in BASELINES:
            estimator = BASELINES[name]().fit(X_train, y_train)
        else:
            estimator = joblib.load(config.PATHS.candidates_dir / f"{setup}_{name}.joblib")
        predictions[name] = estimator.predict(X_test)
        models[name] = {
            "label": MODEL_LABELS[name],
            "test": regression_metrics(y_test, predictions[name]),
            "cv": summaries[name]["cv"],
            "best_params": summaries[name].get("best_params", {}),
        }
    chosen = best_tree_model(summaries)
    frame = build_error_frame(test_df, train_df, predictions[chosen])
    analysis = error_analysis(frame, models[chosen]["test"])
    figs = config.PATHS.figures_dir
    figs.mkdir(parents=True, exist_ok=True)
    for name, fig in (
        (f"predicted_vs_actual_{setup}", plot_predicted_vs_actual(frame, setup, MODEL_LABELS[chosen])),
        (f"residuals_{setup}", plot_residuals(frame, setup, MODEL_LABELS[chosen])),
        (f"error_by_group_{setup}", plot_error_by_group(analysis, setup, MODEL_LABELS[chosen])),
    ):
        fig.savefig(figs / f"{name}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
    result = {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_date_range": [
            str(train_df["date_added"].min().date()),
            str(train_df["date_added"].max().date()),
        ],
        "test_date_range": [str(test_df["date_added"].min().date()), str(test_df["date_added"].max().date())],
        "search_rows": payload.get("search_max_rows"),
        "chosen_model": chosen,
        "models": models,
        "error_analysis": analysis,
    }
    return result, frame


def main(argv: list[str] | None = None) -> int:
    missing = [s for s in split.SETUPS if not (config.PATHS.candidates_dir / f"{s}_summary.json").exists()]
    if missing:
        print("Trained models not found. Run: python -m house_prices.train", file=sys.stderr)
        return 1
    setups = {}
    for setup in split.SETUPS:
        setups[setup], _ = evaluate_setup(setup)
        chosen = setups[setup]["chosen_model"]
        test = setups[setup]["models"][chosen]["test"]
        print(
            f"[{setup}] {MODEL_LABELS[chosen]}: MAE {format_pkr(test['mae_pkr'])}, MdAPE {test['mdape']:.1%}, "
            f"R2(log) {test['r2_log']:.3f}"
        )
    fig = plot_model_comparison(setups)
    fig.savefig(config.PATHS.figures_dir / "model_comparison.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    metadata = json.loads(config.PATHS.metadata_file.read_text()) if config.PATHS.metadata_file.exists() else {}
    ablation = json.loads((config.PATHS.candidates_dir / f"{config.DEPLOY_SETUP}_summary.json").read_text())[
        "summaries"
    ].get("encoding_ablation")
    update_metrics(
        {
            "meta": {
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "synthetic_data": bool(metadata.get("synthetic_data", False)),
                "deployed_setup": config.DEPLOY_SETUP,
                "deployed_model": metadata.get("model"),
                "data_rows_raw": metadata.get("data_rows_raw"),
                "data_rows_clean": metadata.get("data_rows_clean"),
                "marla_sqft": metadata.get("marla_sqft"),
                "test_size": config.TEST_SIZE,
                "cv_folds": metadata.get("cv_folds"),
            },
            "setups": setups,
            "encoding_ablation": ablation,
        }
    )
    print(f"Wrote {config.PATHS.metrics_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
