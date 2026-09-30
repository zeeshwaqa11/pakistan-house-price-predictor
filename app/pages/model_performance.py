import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402
from common import disclaimer, metrics_or_none  # noqa: E402

from house_prices import config, formatting  # noqa: E402
from house_prices.evaluate import MODEL_LABELS, MODEL_ORDER  # noqa: E402

st.title("Model performance")

metrics = metrics_or_none()
if metrics is None:
    st.error("No metrics file was found, so there is nothing to show yet.")
    st.markdown("Run these commands from the project folder, then reload:")
    st.code(
        "python -m house_prices.train\npython -m house_prices.evaluate\n"
        "python -m house_prices.intervals\npython -m house_prices.explain",
        language="bash",
    )
    disclaimer()
    st.stop()

if metrics["meta"].get("synthetic_data"):
    st.warning("These numbers come from SYNTHETIC test data. They are not real results.")

st.write(
    "Every number below is read from `reports/metrics.json`, which the evaluation code writes. "
    "The **random split** shuffles listings 80/20 stratified by city. The **time split** trains on older "
    "listings and tests on the newest 20%, which is closer to real use and is usually a harder test."
)

setups = metrics["setups"]
tabs = st.tabs(["Random split", "Time split"])


def cv_text(cv: dict) -> str:
    entry = cv["rmse_log"]
    return f"{entry['mean']:.3f} ± {entry['std']:.3f}"


for tab, setup in zip(tabs, ("random", "time")):
    payload = setups[setup]
    with tab:
        st.caption(
            f"Train {payload['train_rows']:,} listings ({payload['train_date_range'][0]} to "
            f"{payload['train_date_range'][1]}); test {payload['test_rows']:,} listings "
            f"({payload['test_date_range'][0]} to {payload['test_date_range'][1]}). "
            f"Chosen model: {MODEL_LABELS[payload['chosen_model']]}."
        )
        rows = []
        for name in MODEL_ORDER:
            entry = payload["models"][name]
            test = entry["test"]
            rows.append(
                {
                    "Model": entry["label"],
                    "MAE": formatting.format_pkr(test["mae_pkr"]),
                    "RMSE": formatting.format_pkr(test["rmse_pkr"]),
                    "Median % error": f"{test['mdape']:.1%}",
                    "Mean % error": f"{test['mape']:.1%}",
                    "R² (log price)": f"{test['r2_log']:.3f}",
                    "CV RMSE (log), mean ± std": cv_text(entry["cv"]),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        chart = pd.DataFrame(
            [
                {"Model": payload["models"][n]["label"], "Median % error": payload["models"][n]["test"]["mdape"] * 100}
                for n in MODEL_ORDER
            ]
        )
        st.plotly_chart(
            px.bar(chart, x="Median % error", y="Model", orientation="h", height=340).update_yaxes(
                autorange="reversed"
            ),
            width="stretch",
        )

        analysis = payload["error_analysis"]
        st.subheader("Where the model is weakest")
        for line in analysis["weakest"]:
            st.markdown(f"- {line}")
        group_choice = st.selectbox(
            "Break the error down by",
            ["by_city", "by_property_type", "by_price_decile", "by_location_listings"],
            format_func=lambda k: {
                "by_city": "City",
                "by_property_type": "Property type",
                "by_price_decile": "Price decile",
                "by_location_listings": "Training listings in the location",
            }[k],
            key=f"group_{setup}",
        )
        group_rows = pd.DataFrame(analysis[group_choice])
        st.dataframe(
            pd.DataFrame(
                {
                    "Group": group_rows["group"],
                    "Test listings": group_rows["n"],
                    "Median % error": (group_rows["mdape"] * 100).round(1),
                    "Mean absolute error": [formatting.format_pkr(v) for v in group_rows["mae_pkr"]],
                    "Bias (median)": (group_rows["bias_pct"] * 100).round(1).astype(str) + "%",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        with st.expander("Ten worst predictions"):
            worst = pd.DataFrame(analysis["worst_10"])
            st.dataframe(
                pd.DataFrame(
                    {
                        "City": worst["city"],
                        "Location": worst["location"],
                        "Type": worst["property_type"],
                        "Size (sq ft)": worst["area_sqft"].round(0),
                        "Beds": worst["bedrooms"],
                        "Actual": [formatting.format_pkr(v) for v in worst["actual_price"]],
                        "Predicted": [formatting.format_pkr(v) for v in worst["predicted_price"]],
                        "Error": (worst["ape"] * 100).round(0).astype(int).astype(str) + "%",
                        "Likely reason (a guess)": worst["likely_reason"],
                    }
                ),
                hide_index=True,
                width="stretch",
            )
        figures = config.PATHS.figures_dir
        for name, caption in (
            (f"predicted_vs_actual_{setup}", "Predicted vs actual"),
            (f"residuals_{setup}", "Residuals vs predicted"),
        ):
            path = figures / f"{name}.png"
            if path.exists():
                st.image(str(path), caption=caption)

st.subheader("80% prediction interval")
intervals = metrics.get("intervals")
if intervals:
    rows = []
    for setup, block in intervals.items():
        overall = block["overall"]
        rows.append(
            {
                "Split": setup,
                "Coverage (target 80%)": f"{overall['coverage']:.1%}",
                "Mean width": formatting.format_pkr(overall["mean_width_pkr"]),
                "Median width": formatting.format_pkr(overall["median_width_pkr"]),
                "Mean width / estimate": f"{overall['mean_relative_width']:.0%}",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    chosen_split = st.radio("Per-city coverage for", list(intervals), horizontal=True)
    city_rows = pd.DataFrame(intervals[chosen_split]["by_city"])
    st.dataframe(
        pd.DataFrame(
            {
                "City": city_rows["city"],
                "Test listings": city_rows["n"],
                "Coverage": (city_rows["coverage"] * 100).round(1).astype(str) + "%",
                "Mean width": [formatting.format_pkr(v) for v in city_rows["mean_width_pkr"]],
            }
        ),
        hide_index=True,
        width="stretch",
    )
else:
    st.info("Interval results are not available yet. Run `python -m house_prices.intervals`.")

st.subheader("What drives the estimate")
shap_block = metrics.get("shap")
if shap_block:
    groups = pd.DataFrame(shap_block["group_importance"])
    st.plotly_chart(
        px.bar(
            groups,
            x="mean_abs_shap",
            y="group",
            orientation="h",
            labels={"mean_abs_shap": "Mean |SHAP| (log price)", "group": ""},
            height=340,
        ).update_yaxes(autorange="reversed"),
        width="stretch",
    )
    beeswarm = config.PATHS.figures_dir / "shap_beeswarm.png"
    if beeswarm.exists():
        st.image(str(beeswarm), caption="SHAP beeswarm on test listings")
else:
    st.info("SHAP results are not available yet. Run `python -m house_prices.explain`.")

disclaimer()
