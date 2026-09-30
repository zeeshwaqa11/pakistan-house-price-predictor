import re
import sys

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap

from house_prices import config, features, split
from house_prices.evaluate import update_metrics

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

READABLE = {
    "num__area_sqft": "Size (sq ft)",
    "num__log_area_sqft": "Size (log sq ft)",
    "num__bedrooms": "Bedrooms",
    "num__baths": "Bathrooms",
    "num__latitude": "Latitude",
    "num__longitude": "Longitude",
    "num__distance_to_centre_km": "Distance to city centre (km)",
    f"loc__{features.LOCATION_FEATURE}": "Location (encoded price level)",
}

GROUP_OF = {
    "num__area_sqft": "Size",
    "num__log_area_sqft": "Size",
    "num__bedrooms": "Bedrooms",
    "num__baths": "Bathrooms",
    "num__latitude": "Coordinates",
    "num__longitude": "Coordinates",
    "num__distance_to_centre_km": "Distance to city centre",
    f"loc__{features.LOCATION_FEATURE}": "Location",
}


def readable_name(feature_name: str) -> str:
    if feature_name in READABLE:
        return READABLE[feature_name]
    match = re.match(r"cat__(city|property_type)_(.*)", feature_name)
    if match:
        kind = "City" if match.group(1) == "city" else "Property type"
        return f"{kind}: {match.group(2)}"
    return feature_name


def group_name(feature_name: str) -> str:
    if feature_name in GROUP_OF:
        return GROUP_OF[feature_name]
    if feature_name.startswith("cat__city_"):
        return "City"
    if feature_name.startswith("cat__property_type_"):
        return "Property type"
    return feature_name


def preprocess(pipeline, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(pipeline[:-1].transform(X), dtype=float)


def feature_names(pipeline) -> list[str]:
    return list(pipeline.named_steps["prep"].get_feature_names_out())


def make_explainer(pipeline) -> shap.TreeExplainer:
    return shap.TreeExplainer(pipeline.named_steps["model"])


def shap_matrix(pipeline, explainer, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    transformed = preprocess(pipeline, X)
    values = np.asarray(explainer.shap_values(transformed))
    return transformed, values


def base_value(explainer) -> float:
    return float(np.ravel(explainer.expected_value)[0])


def group_contributions(names: list[str], values: np.ndarray) -> dict[str, float]:
    totals: dict[str, float] = {}
    for name, value in zip(names, values):
        group = group_name(name)
        totals[group] = totals.get(group, 0.0) + float(value)
    return totals


def _percent(value: float) -> str:
    change = np.exp(value) - 1
    return f"{change:+.0%}"


def plain_english(contributions: dict[str, float]) -> str:
    ranked = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    if not ranked:
        return "No feature moved this estimate away from the typical listing."
    first, first_value = ranked[0]
    if first_value >= 0:
        sentence = f"{first} added the most to this price ({_percent(first_value)})"
    else:
        sentence = f"{first} pulled this price down the most ({_percent(first_value)})"
    if len(ranked) > 1:
        second, second_value = ranked[1]
        second_verb = "followed by" if (second_value >= 0) == (first_value >= 0) else "while"
        tail = second.lower() if second != "City" else "city"
        if second_verb == "followed by":
            sentence += f", followed by {tail} ({_percent(second_value)})"
        else:
            direction = "pushed it up" if second_value >= 0 else "pulled it down"
            sentence += f", while {tail} {direction} ({_percent(second_value)})"
    return sentence + "."


def explain_prediction(pipeline, explainer, X_row: pd.DataFrame, labels: dict | None = None, draw: bool = True) -> dict:
    names = feature_names(pipeline)
    _, values = shap_matrix(pipeline, explainer, X_row)
    row_values = values[0]
    base = base_value(explainer)
    contributions = group_contributions(names, row_values)
    ordered = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    result = {
        "base_log": base,
        "base_price": float(np.exp(base)),
        "prediction_log": float(base + row_values.sum()),
        "contributions": dict(ordered),
        "sentence": plain_english(contributions),
        "figure": None,
    }
    if draw:
        labels = labels or {}
        display = [f"{g} ({labels[g]})" if g in labels else g for g, _ in ordered]
        explanation = shap.Explanation(
            values=np.array([v for _, v in ordered]),
            base_values=base,
            data=None,
            feature_names=display,
        )
        plt.close("all")
        shap.plots.waterfall(explanation, max_display=len(display), show=False)
        fig = plt.gcf()
        fig.set_size_inches(8, 0.5 * len(display) + 1.8)
        result["figure"] = fig
    return result


def global_importance(names: list[str], values: np.ndarray) -> tuple[list[dict], list[dict]]:
    mean_abs = np.abs(values).mean(axis=0)
    per_feature = sorted(
        ({"feature": readable_name(n), "mean_abs_shap": float(v)} for n, v in zip(names, mean_abs)),
        key=lambda r: r["mean_abs_shap"],
        reverse=True,
    )
    grouped: dict[str, np.ndarray] = {}
    for j, name in enumerate(names):
        group = group_name(name)
        grouped[group] = grouped.get(group, np.zeros(len(values))) + values[:, j]
    per_group = sorted(
        ({"group": g, "mean_abs_shap": float(np.abs(v).mean())} for g, v in grouped.items()),
        key=lambda r: r["mean_abs_shap"],
        reverse=True,
    )
    return per_feature, per_group


def plot_global(names: list[str], transformed: np.ndarray, values: np.ndarray):
    readable = [readable_name(n) for n in names]
    plt.close("all")
    shap.summary_plot(values, transformed, feature_names=readable, plot_type="bar", max_display=15, show=False)
    bar = plt.gcf()
    bar.set_size_inches(8, 5.5)
    plt.title("Mean |SHAP value| (average effect on log price)")
    bar.tight_layout()
    bar_path = config.PATHS.figures_dir / "shap_bar.png"
    bar.savefig(bar_path, dpi=140, bbox_inches="tight")
    plt.close(bar)
    shap.summary_plot(values, transformed, feature_names=readable, max_display=15, show=False)
    swarm = plt.gcf()
    swarm.set_size_inches(9, 6)
    plt.title("SHAP values by feature value (each dot is a test listing)")
    swarm.tight_layout()
    swarm.savefig(config.PATHS.figures_dir / "shap_beeswarm.png", dpi=140, bbox_inches="tight")
    plt.close(swarm)


def main(argv: list[str] | None = None) -> int:
    if not config.PATHS.model_file.exists():
        print("Trained model not found. Run: python -m house_prices.train", file=sys.stderr)
        return 1
    pipeline = joblib.load(config.PATHS.model_file)
    _, test_df = split.load_split(config.DEPLOY_SETUP)
    size = min(config.SHAP_SAMPLE_SIZE, len(test_df))
    sample = test_df.sample(size, random_state=config.RANDOM_STATE)
    X = features.make_inputs(sample)
    explainer = make_explainer(pipeline)
    names = feature_names(pipeline)
    transformed, values = shap_matrix(pipeline, explainer, X)
    config.PATHS.figures_dir.mkdir(parents=True, exist_ok=True)
    plot_global(names, transformed, values)
    per_feature, per_group = global_importance(names, values)
    nearest = int((sample["price"] - sample["price"].median()).abs().to_numpy().argmin())
    median_row = sample.iloc[[nearest]]
    example = explain_prediction(
        pipeline,
        explainer,
        features.make_inputs(median_row),
        labels={
            "Location": str(median_row["location"].iloc[0]),
            "City": str(median_row["city"].iloc[0]),
            "Property type": str(median_row["property_type"].iloc[0]),
            "Size": f"{median_row['area_sqft'].iloc[0]:,.0f} sq ft",
            "Bedrooms": str(int(median_row["bedrooms"].iloc[0])),
            "Bathrooms": str(int(median_row["baths"].iloc[0])),
        },
    )
    example["figure"].savefig(config.PATHS.figures_dir / "shap_waterfall_example.png", dpi=140, bbox_inches="tight")
    plt.close(example["figure"])
    update_metrics(
        {
            "shap": {
                "model": type(pipeline.named_steps["model"]).__name__,
                "sample_size": int(size),
                "top_features": per_feature[:15],
                "group_importance": per_group,
                "example_sentence": example["sentence"],
                "unit": "log price",
            }
        }
    )
    top = per_group[0]["group"] if per_group else "n/a"
    print(f"SHAP on {size:,} test listings; most influential group: {top}")
    print(f"Example: {example['sentence']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
