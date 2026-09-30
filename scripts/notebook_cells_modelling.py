from nbformat.v4 import new_notebook


def build(md, code, setup_cell):
    modelling = [
        md(
            """
# 03 Modelling

Reads what `python -m house_prices.train`, `evaluate` and `intervals` produced. Tuning and training happen in
those commands, not here, so the notebook stays quick and cannot touch the test set by accident. Two evaluation
setups are reported: a **random split** (shuffled 80/20, stratified by city) and a **time split** (train on older
listings, test on the newest 20%).
"""
        ),
        setup_cell,
        code(
            """
import json

import joblib
from IPython.display import display

from house_prices import evaluate, features, formatting, split, train

metrics = json.loads(config.PATHS.metrics_json.read_text())
summaries = {s: json.loads((config.PATHS.candidates_dir / f"{s}_summary.json").read_text()) for s in split.SETUPS}
print("Synthetic data:", metrics["meta"]["synthetic_data"])
print("Deployed:", metrics["meta"]["deployed_model"], "trained with the", metrics["meta"]["deployed_setup"], "split")
"""
        ),
        md(
            """
## The pipeline

Preprocessing sits inside one scikit-learn `Pipeline`, so cross-validation refits every step on each training
fold. The location target encoder is the important one: fitted on the full data it would leak the price of the
rows it later predicts.
"""
        ),
        code(
            """
from sklearn import set_config

set_config(display="text")
print(train.make_pipeline("lightgbm"))
print()
print("Inputs:", features.INPUT_COLUMNS)
print("Numeric:", features.NUMERIC_FEATURES)
print("Categorical (one-hot):", features.CATEGORICAL_FEATURES)
print(f"Location: grouped when fewer than {config.MIN_LOCATION_COUNT} training listings, then target-encoded (smoothed, 5-fold cross-fitted)")
"""
        ),
        md("## Tuning setup"),
        code(
            """
rows = []
for name in train.MODEL_NAMES:
    payload = summaries["random"]["summaries"][name]
    rows.append({"model": payload["label"], "search iterations": payload["n_iter"], "CV folds": payload["cv_folds"], "rows searched": payload["cv_rows"], "best parameters": payload["best_params"]})
display(pd.DataFrame(rows))
print("Search spaces (kind, low, high):")
for name in train.MODEL_NAMES:
    print(f"  {name}: {train.SEARCH_SPACES[name]}")
"""
        ),
        md("## Cross-validation on the training set only"),
        code(
            """
def cv_table(setup):
    rows = []
    for name in evaluate.MODEL_ORDER:
        cv = summaries[setup]["summaries"][name]["cv"]
        rows.append(
            {
                "model": evaluate.MODEL_LABELS[name],
                "RMSE (log) mean": cv["rmse_log"]["mean"],
                "RMSE (log) std": cv["rmse_log"]["std"],
                "R2 (log) mean": cv["r2_log"]["mean"],
                "MdAPE mean": cv["mdape"]["mean"],
                "MAE (PKR) mean": cv["mae_pkr"]["mean"],
            }
        )
    return pd.DataFrame(rows).round(4)


for setup in split.SETUPS:
    print(f"CV on the {setup} split's training set")
    display(cv_table(setup))
"""
        ),
        md(
            """
## Location encoding: target encoding against one-hot

LightGBM with the tuned parameters, cross-validated on the training set of the deployed split.
"""
        ),
        code(
            """
ablation = metrics.get("encoding_ablation")
if ablation:
    display(pd.DataFrame({k: {m: v[m]["mean"] for m in ("rmse_log", "r2_log", "mdape", "mae_pkr")} for k, v in ablation.items()}).T.round(4))
"""
        ),
        md(
            """
## Test-set results, both setups

The test rows were used for the first time in `python -m house_prices.evaluate`.
"""
        ),
        code(
            """
def test_table(setup):
    rows = []
    for name in evaluate.MODEL_ORDER:
        t = metrics["setups"][setup]["models"][name]["test"]
        rows.append(
            {
                "model": evaluate.MODEL_LABELS[name],
                "MAE": formatting.format_pkr(t["mae_pkr"]),
                "RMSE": formatting.format_pkr(t["rmse_pkr"]),
                "MdAPE": f"{t['mdape']:.1%}",
                "MAPE": f"{t['mape']:.1%}",
                "R2 (log)": round(t["r2_log"], 3),
            }
        )
    return pd.DataFrame(rows)


for setup in split.SETUPS:
    s = metrics["setups"][setup]
    print(f"{setup} split: train {s['train_rows']:,} ({s['train_date_range'][0]} to {s['train_date_range'][1]}), test {s['test_rows']:,} ({s['test_date_range'][0]} to {s['test_date_range'][1]}); chosen model {evaluate.MODEL_LABELS[s['chosen_model']]}")
    display(test_table(setup))
"""
        ),
        code(
            """
fig = plt.figure(figsize=(13, 4.8))
from PIL import Image

image = Image.open(config.PATHS.figures_dir / "model_comparison.png")
plt.imshow(image)
plt.axis("off")
plt.show()
"""
        ),
        md("## Random split against time split"),
        code(
            """
lines = []
for name in evaluate.MODEL_ORDER:
    r = metrics["setups"]["random"]["models"][name]["test"]
    t = metrics["setups"]["time"]["models"][name]["test"]
    lines.append(f"{evaluate.MODEL_LABELS[name]}: median error {r['mdape']:.1%} on the random split, {t['mdape']:.1%} on the time split ({t['mdape'] - r['mdape']:+.1%} points).")
print("\\n".join("- " + l for l in lines))
chosen_r = metrics["setups"]["random"]["chosen_model"]
chosen_t = metrics["setups"]["time"]["chosen_model"]
cv_r = metrics["setups"]["random"]["models"][chosen_r]["cv"]["rmse_log"]["mean"]
te_r = metrics["setups"]["random"]["models"][chosen_r]["test"]["rmse_log"]
te_t = metrics["setups"]["time"]["models"][chosen_t]["test"]["rmse_log"]
print()
print(f"Chosen model, RMSE on log price: CV {cv_r:.3f}, random-split test {te_r:.3f}, time-split test {te_t:.3f}.")
"""
        ),
        md(
            """
A shuffled split lets the model see listings from the same weeks as its test rows (the same agents, near
duplicates, the same market level). A time split asks it to price listings from a later period, which is how it
would be used. When the time-split error is larger, that gap is the honest number to quote.
"""
        ),
        md("## 80% prediction interval"),
        code(
            """
rows = []
for setup, block in metrics["intervals"].items():
    o = block["overall"]
    rows.append({"split": setup, "coverage": f"{o['coverage']:.1%}", "target": f"{block['level']:.0%}", "mean width": formatting.format_pkr(o["mean_width_pkr"]), "mean width / estimate": f"{o['mean_relative_width']:.0%}"})
display(pd.DataFrame(rows))
for setup, block in metrics["intervals"].items():
    print(f"Coverage by city, {setup} split")
    display(pd.DataFrame(block["by_city"]).assign(coverage=lambda d: (d["coverage"] * 100).round(1), mean_width_pkr=lambda d: d["mean_width_pkr"].round(0)))
"""
        ),
    ]
    errors = [
        md(
            """
# 04 Error analysis

Where does the chosen tree model go wrong? Everything is computed on the test set of each split from the fitted
candidates saved by `train`, and every sentence in the summary is generated from those numbers.
"""
        ),
        setup_cell,
        code(
            """
import json

import joblib
from IPython.display import display

from house_prices import evaluate, features, formatting, split

metrics = json.loads(config.PATHS.metrics_json.read_text())
frames = {}
for setup in split.SETUPS:
    train_df, test_df = split.load_split(setup)
    chosen = metrics["setups"][setup]["chosen_model"]
    model = joblib.load(config.PATHS.candidates_dir / f"{setup}_{chosen}.joblib")
    frames[setup] = evaluate.build_error_frame(test_df, train_df, model.predict(features.make_inputs(test_df)))
    print(setup, evaluate.MODEL_LABELS[chosen], f"{len(test_df):,} test listings")
"""
        ),
        md("## Predicted against actual, and residuals"),
        code(
            """
for setup, frame in frames.items():
    label = evaluate.MODEL_LABELS[metrics["setups"][setup]["chosen_model"]]
    fig = evaluate.plot_predicted_vs_actual(frame, setup, label)
    plt.show()
    fig = evaluate.plot_residuals(frame, setup, label)
    plt.show()
"""
        ),
        md("## Error by city, property type, price decile and location coverage"),
        code(
            """
for setup, frame in frames.items():
    analysis = metrics["setups"][setup]["error_analysis"]
    label = evaluate.MODEL_LABELS[metrics["setups"][setup]["chosen_model"]]
    fig = evaluate.plot_error_by_group(analysis, setup, label)
    plt.show()
    for key in ("by_city", "by_property_type", "by_price_decile", "by_location_listings"):
        table = pd.DataFrame(analysis[key])
        table["mdape"] = (table["mdape"] * 100).round(1)
        table["bias_pct"] = (table["bias_pct"] * 100).round(1)
        table["mae_pkr"] = table["mae_pkr"].round(0)
        print(f"{setup} split: {key}")
        display(table)
"""
        ),
        md(
            """
## Bias by price decile

Positive values mean the model over-predicts that band on average; negative means it under-predicts. Trees pull
extreme prices toward the middle, so the cheapest deciles tend to be over-predicted and the most expensive
under-predicted. The table shows whether that happens here.
"""
        ),
        code(
            """
for setup, frame in frames.items():
    decile = frame.groupby("price_decile").agg(listings=("ape", "size"), median_error=("ape", "median"), bias=("log_ratio", lambda v: np.exp(v.median()) - 1))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    sns.barplot(x=decile.index, y=decile["bias"] * 100, ax=ax, color="#4c78a8")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Price decile (1 = cheapest)")
    ax.set_ylabel("Median bias (%)")
    ax.set_title(f"Median over/under-prediction by price decile ({setup} split)")
    plt.show()
"""
        ),
        md("## The ten worst predictions"),
        code(
            """
for setup in split.SETUPS:
    worst = pd.DataFrame(metrics["setups"][setup]["error_analysis"]["worst_10"])
    show = worst.assign(actual=[formatting.format_pkr(v) for v in worst["actual_price"]], predicted=[formatting.format_pkr(v) for v in worst["predicted_price"]], error=(worst["ape"] * 100).round(0).astype(int).astype(str) + "%")
    print(f"{setup} split")
    display(show[["city", "location", "property_type", "area_sqft", "bedrooms", "baths", "actual", "predicted", "error", "location_train_listings", "likely_reason"]])
"""
        ),
        md(
            """
The reasons are automatic guesses from simple rules (price per sq ft against the location's usual level,
area per bedroom, and location coverage). They point at likely data entry problems; they are not verified.
"""
        ),
        md("## Summary of the weakest areas"),
        code(
            """
for setup in split.SETUPS:
    print(f"{setup} split")
    for line in metrics["setups"][setup]["error_analysis"]["weakest"]:
        print("-", line)
    print()
"""
        ),
        md("## What the model relies on (SHAP)"),
        code(
            """
from PIL import Image

for name in ("shap_bar", "shap_beeswarm", "shap_waterfall_example"):
    path = config.PATHS.figures_dir / f"{name}.png"
    if path.exists():
        plt.figure(figsize=(9, 5.5))
        plt.imshow(Image.open(path))
        plt.axis("off")
        plt.show()
shap_block = metrics.get("shap")
if shap_block:
    print(shap_block["example_sentence"])
    display(pd.DataFrame(shap_block["group_importance"]).round(4))
"""
        ),
    ]
    return new_notebook(cells=modelling), new_notebook(cells=errors)
