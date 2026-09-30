import argparse
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone

import joblib
import lightgbm
import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn
import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate
from xgboost import XGBRegressor

from house_prices import config, features, split
from house_prices.evaluate import BASELINES, MODEL_LABELS, make_scorers

MODEL_NAMES = ["linear_regression", "ridge", "random_forest", "xgboost", "lightgbm"]
TREE_MODELS = ["random_forest", "xgboost", "lightgbm"]
SEARCH_MAX_ROWS = 60_000

SEARCH_SPACES = {
    "linear_regression": {},
    "ridge": {"model__alpha": ("loguniform", 0.01, 100.0)},
    "random_forest": {
        "model__n_estimators": ("randint", 120, 260),
        "model__max_depth": ("randint", 10, 24),
        "model__min_samples_leaf": ("randint", 1, 8),
        "model__max_features": ("uniform", 0.4, 0.6),
    },
    "xgboost": {
        "model__n_estimators": ("randint", 300, 800),
        "model__learning_rate": ("loguniform", 0.03, 0.2),
        "model__max_depth": ("randint", 4, 10),
        "model__subsample": ("uniform", 0.7, 0.3),
        "model__colsample_bytree": ("uniform", 0.6, 0.4),
        "model__min_child_weight": ("randint", 1, 10),
        "model__reg_lambda": ("loguniform", 0.5, 10.0),
    },
    "lightgbm": {
        "model__n_estimators": ("randint", 300, 900),
        "model__learning_rate": ("loguniform", 0.03, 0.2),
        "model__num_leaves": ("randint", 31, 200),
        "model__min_child_samples": ("randint", 10, 60),
        "model__subsample": ("uniform", 0.7, 0.3),
        "model__colsample_bytree": ("uniform", 0.6, 0.4),
        "model__reg_lambda": ("loguniform", 0.5, 10.0),
    },
}

SEARCH_ITERATIONS = {"linear_regression": 0, "ridge": 8, "random_forest": 4, "xgboost": 10, "lightgbm": 10}

FAST_OVERRIDES = {
    "random_forest": {"model__n_estimators": ("randint", 15, 25), "model__max_depth": ("randint", 4, 8)},
    "xgboost": {"model__n_estimators": ("randint", 15, 30), "model__max_depth": ("randint", 3, 5)},
    "lightgbm": {"model__n_estimators": ("randint", 15, 30), "model__num_leaves": ("randint", 8, 16)},
}


def to_distribution(spec):
    kind, low, high = spec
    if kind == "loguniform":
        return stats.loguniform(low, high)
    if kind == "randint":
        return stats.randint(low, high)
    if kind == "uniform":
        return stats.uniform(low, high)
    raise ValueError(kind)


def search_space(name: str, fast: bool) -> dict:
    space = dict(SEARCH_SPACES[name])
    if fast:
        space.update(FAST_OVERRIDES.get(name, {}))
    return space


def make_pipeline(name: str, params: dict | None = None):
    seed = config.RANDOM_STATE
    scale = False
    if name == "linear_regression":
        model, scale = LinearRegression(), True
    elif name == "ridge":
        model, scale = Ridge(random_state=seed), True
    elif name == "random_forest":
        model = RandomForestRegressor(n_jobs=-1, random_state=seed, max_samples=0.5)
    elif name == "xgboost":
        model = XGBRegressor(tree_method="hist", n_jobs=-1, random_state=seed, verbosity=0)
    elif name == "lightgbm":
        model = lightgbm.LGBMRegressor(n_jobs=-1, random_state=seed, verbose=-1, subsample_freq=1)
    else:
        raise ValueError(f"Unknown model: {name}")
    pipeline = features.build_pipeline(model, scale=scale)
    if params:
        pipeline.set_params(**params)
    return pipeline


def _summarise_cv(scores: dict) -> dict:
    out = {}
    for metric in ("rmse_log", "r2_log", "mae_pkr", "mdape"):
        values = np.asarray(scores[metric], dtype=float)
        if metric != "r2_log":
            values = -values
        out[metric] = {"mean": float(values.mean()), "std": float(values.std())}
    return out


def _cv_summary_from_search(search: RandomizedSearchCV, best: int) -> dict:
    results = search.cv_results_
    out = {}
    for metric in ("rmse_log", "r2_log", "mae_pkr", "mdape"):
        mean = float(results[f"mean_test_{metric}"][best])
        std = float(results[f"std_test_{metric}"][best])
        out[metric] = {"mean": mean if metric == "r2_log" else -mean, "std": std}
    return out


def _jsonable(params: dict) -> dict:
    out = {}
    for key, value in params.items():
        if isinstance(value, (np.integer,)):
            value = int(value)
        elif isinstance(value, (np.floating,)):
            value = float(value)
        out[key] = value
    return out


def search_frame(train_df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows and len(train_df) > max_rows:
        return train_df.sample(max_rows, random_state=config.RANDOM_STATE)
    return train_df


def tune_and_fit(name, train_df, fast: bool, max_rows: int | None):
    cv = KFold(n_splits=3 if fast else config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    scorers = make_scorers()
    subset = search_frame(train_df, 2000 if fast else max_rows)
    X_sub, y_sub = features.make_inputs(subset), features.make_target(subset)
    X_full, y_full = features.make_inputs(train_df), features.make_target(train_df)
    started = time.time()
    n_iter = 2 if fast and SEARCH_ITERATIONS[name] else SEARCH_ITERATIONS[name]
    if n_iter == 0:
        scores = cross_validate(make_pipeline(name), X_sub, y_sub, cv=cv, scoring=scorers)
        best_params = {}
        cv_summary = _summarise_cv({m: scores[f"test_{m}"] for m in scorers})
    else:
        space = {k: to_distribution(v) for k, v in search_space(name, fast).items()}
        search = RandomizedSearchCV(
            make_pipeline(name),
            space,
            n_iter=n_iter,
            cv=cv,
            scoring=scorers,
            refit=False,
            random_state=config.RANDOM_STATE,
            n_jobs=1,
        )
        search.fit(X_sub, y_sub)
        best = int(np.argmin(search.cv_results_["rank_test_rmse_log"]))
        best_params = _jsonable(search.cv_results_["params"][best])
        cv_summary = _cv_summary_from_search(search, best)
    final = make_pipeline(name, best_params).fit(X_full, y_full)
    return final, {
        "name": name,
        "label": MODEL_LABELS[name],
        "best_params": best_params,
        "search_space": {k: list(v) for k, v in search_space(name, fast).items()},
        "n_iter": n_iter,
        "cv_folds": cv.get_n_splits(),
        "cv_rows": len(subset),
        "train_rows": len(train_df),
        "cv": cv_summary,
        "seconds": round(time.time() - started, 1),
    }


def baseline_cv(name, train_df, fast: bool, max_rows: int | None) -> dict:
    cv = KFold(n_splits=3 if fast else config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    subset = search_frame(train_df, 2000 if fast else max_rows)
    scores = cross_validate(
        BASELINES[name](),
        features.make_inputs(subset),
        features.make_target(subset),
        cv=cv,
        scoring=make_scorers(),
    )
    return {
        "name": name,
        "label": MODEL_LABELS[name],
        "cv_folds": cv.get_n_splits(),
        "cv_rows": len(subset),
        "cv": _summarise_cv({m: scores[f"test_{m}"] for m in make_scorers()}),
    }


def encoding_ablation(train_df, lgbm_params: dict, fast: bool, max_rows: int | None) -> dict:
    cv = KFold(n_splits=3 if fast else config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    subset = search_frame(train_df, 2000 if fast else max_rows)
    X, y = features.make_inputs(subset), features.make_target(subset)
    out = {}
    for label, encoding in (("target_encoding", "target"), ("one_hot", "onehot")):
        model = lightgbm.LGBMRegressor(
            n_jobs=-1, random_state=config.RANDOM_STATE, verbose=-1, subsample_freq=1
        )
        pipeline = features.build_pipeline(model, location_encoding=encoding)
        pipeline.set_params(**lgbm_params)
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=make_scorers())
        out[label] = _summarise_cv({m: scores[f"test_{m}"] for m in make_scorers()})
    return out


def build_location_index(train_df: pd.DataFrame) -> pd.DataFrame:
    frame = train_df.assign(ppsf=train_df["price"] / train_df["area_sqft"])
    index = (
        frame.groupby(["city", "location"])
        .agg(
            n_train=("price", "size"),
            median_lat=("latitude", "median"),
            median_lon=("longitude", "median"),
            median_ppsf=("ppsf", "median"),
        )
        .reset_index()
    )
    index["grouped_as_other"] = index["n_train"] < config.MIN_LOCATION_COUNT
    return index.sort_values(["city", "n_train"], ascending=[True, False]).reset_index(drop=True)


def library_versions() -> dict:
    import shap

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "lightgbm": lightgbm.__version__,
        "shap": shap.__version__,
        "joblib": joblib.__version__,
    }


def _read_cleaning_summary() -> dict:
    path = config.PATHS.cleaning_summary
    return json.loads(path.read_text()) if path.exists() else {}


def write_metadata(chosen: dict, setup: str, train_df: pd.DataFrame, summaries: dict) -> dict:
    cleaning = _read_cleaning_summary()
    metadata = {
        "training_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "libraries": library_versions(),
        "setup": setup,
        "model": chosen["name"],
        "model_label": chosen["label"],
        "parameters": chosen["best_params"],
        "search_space": chosen["search_space"],
        "cv_folds": chosen["cv_folds"],
        "cv_rows": chosen["cv_rows"],
        "cv_scores": chosen["cv"],
        "features": {
            "input_columns": features.INPUT_COLUMNS,
            "numeric": features.NUMERIC_FEATURES,
            "categorical": features.CATEGORICAL_FEATURES,
            "location": features.LOCATION_FEATURE,
        },
        "target": "log(price in PKR)",
        "min_location_count": config.MIN_LOCATION_COUNT,
        "rare_location_count": config.RARE_LOCATION_COUNT,
        "random_state": config.RANDOM_STATE,
        "train_rows": len(train_df),
        "train_date_range": [
            str(train_df["date_added"].min().date()),
            str(train_df["date_added"].max().date()),
        ],
        "data_rows_raw": cleaning.get("raw_rows"),
        "data_rows_clean": cleaning.get("clean_rows"),
        "data_file": cleaning.get("source_file"),
        "data_file_sha256": cleaning.get("file_sha256"),
        "marla_sqft": cleaning.get("marla_sqft", config.MARLA_SQFT_FALLBACK),
        "marla_source": cleaning.get("marla_source"),
        "synthetic_data": bool(cleaning.get("synthetic", False)),
        "model_comparison_cv": {name: {"label": s["label"], "cv": s["cv"]} for name, s in summaries.items()},
    }
    config.PATHS.metadata_file.write_text(json.dumps(metadata, indent=2))
    return metadata


def train_setup(setup: str, fast: bool, models: list[str], max_rows: int | None, deploy: bool) -> dict:
    train_df = split.load_train(setup)
    print(f"[{setup}] training rows: {len(train_df):,}")
    config.PATHS.candidates_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name in BASELINES:
        summaries[name] = baseline_cv(name, train_df, fast, max_rows)
        print(f"[{setup}] {MODEL_LABELS[name]}: CV RMSE(log) {summaries[name]['cv']['rmse_log']['mean']:.4f}")
    fitted = {}
    for name in models:
        estimator, summary = tune_and_fit(name, train_df, fast, max_rows)
        summaries[name] = summary
        fitted[name] = estimator
        joblib.dump(estimator, config.PATHS.candidates_dir / f"{setup}_{name}.joblib")
        print(
            f"[{setup}] {summary['label']}: CV RMSE(log) {summary['cv']['rmse_log']['mean']:.4f} "
            f"+/- {summary['cv']['rmse_log']['std']:.4f} in {summary['seconds']}s"
        )
    lgbm = summaries.get("lightgbm")
    if lgbm is not None and setup == config.DEPLOY_SETUP:
        summaries["encoding_ablation"] = encoding_ablation(train_df, lgbm["best_params"], fast, max_rows)
    payload = {
        "setup": setup,
        "train_rows": len(train_df),
        "train_date_range": [
            str(train_df["date_added"].min().date()),
            str(train_df["date_added"].max().date()),
        ],
        "search_max_rows": max_rows,
        "summaries": summaries,
    }
    (config.PATHS.candidates_dir / f"{setup}_summary.json").write_text(json.dumps(payload, indent=2))
    if deploy:
        contenders = {n: summaries[n] for n in models if n in summaries}
        deployable = [n for n in contenders if n in TREE_MODELS]
        if not deployable:
            raise SystemExit("Deployment needs at least one tree model (random_forest, xgboost or lightgbm).")
        best_name = min(deployable, key=lambda n: contenders[n]["cv"]["rmse_log"]["mean"])
        shutil.copyfile(config.PATHS.candidates_dir / f"{setup}_{best_name}.joblib", config.PATHS.model_file)
        build_location_index(train_df).to_csv(config.PATHS.location_index, index=False)
        write_metadata(contenders[best_name], setup, train_df, {**contenders})
        print(f"[{setup}] deployed model: {MODEL_LABELS[best_name]} -> {config.PATHS.model_file}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tune and train all models on the training split only.")
    parser.add_argument("--fast", action="store_true", help="tiny search for tests and CI")
    parser.add_argument("--setup", choices=["random", "time", "both"], default="both")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    parser.add_argument(
        "--search-rows",
        type=int,
        default=SEARCH_MAX_ROWS,
        help="rows used for the tuning search; 0 uses the whole training set",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if not config.PATHS.splits.exists():
        print(
            "Splits not found. Run: python -m house_prices.clean and python -m house_prices.split",
            file=sys.stderr,
        )
        return 1
    fast = args.fast or config.fast_mode()
    config.PATHS.models_dir.mkdir(parents=True, exist_ok=True)
    setups = ["random", "time"] if args.setup == "both" else [args.setup]
    max_rows = args.search_rows or None
    for setup in setups:
        train_setup(setup, fast, args.models, max_rows, deploy=setup == config.DEPLOY_SETUP)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
