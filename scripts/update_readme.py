import json
import re
import sys

from house_prices import config, formatting
from house_prices.evaluate import MODEL_ORDER

SECTIONS = ["RESULTS", "INTERVALS", "ERRORS", "ABLATION", "SHAP", "FUNNEL"]

PLACEHOLDER = (
    "_Not generated yet. Run the pipeline on the real dataset (see [How to run](#how-to-run)) and then "
    "`python -m scripts.update_readme`._"
)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _banner(metrics: dict) -> str:
    if metrics["meta"].get("synthetic_data"):
        return "> **These numbers come from SYNTHETIC test data and are not real results.**\n\n"
    return ""


def render_results(metrics: dict) -> str:
    parts = [_banner(metrics)]
    for setup, title in (("random", "Random split"), ("time", "Time split")):
        payload = metrics["setups"][setup]
        parts.append(
            f"**{title}**: train {payload['train_rows']:,} listings "
            f"({payload['train_date_range'][0]} to {payload['train_date_range'][1]}), test {payload['test_rows']:,} "
            f"({payload['test_date_range'][0]} to {payload['test_date_range'][1]}).\n"
        )
        rows = []
        for name in MODEL_ORDER:
            entry = payload["models"][name]
            t, cv = entry["test"], entry["cv"]["rmse_log"]
            label = f"**{entry['label']}**" if name == payload["chosen_model"] else entry["label"]
            rows.append(
                [
                    label,
                    formatting.format_pkr(t["mae_pkr"]),
                    formatting.format_pkr(t["rmse_pkr"]),
                    f"{t['mdape']:.1%}",
                    f"{t['mape']:.1%}",
                    f"{t['r2_log']:.3f}",
                    f"{cv['mean']:.3f} ± {cv['std']:.3f}",
                ]
            )
        parts.append(
            _table(
                [
                    "Model",
                    "MAE",
                    "RMSE",
                    "Median % error",
                    "Mean % error",
                    "R² (log price)",
                    "CV RMSE (log), mean ± std",
                ],
                rows,
            )
        )
        parts.append("")
    parts.append(
        "Bold rows are the best tree model by cross-validation, chosen without looking at the test set. "
        "Cross-validation used up to "
        f"{metrics['setups']['random'].get('search_rows') or 'all'} training rows for the tuning search."
    )
    return "\n".join(parts)


def render_intervals(metrics: dict) -> str:
    block = metrics.get("intervals")
    if not block:
        return PLACEHOLDER
    rows = []
    for setup, payload in block.items():
        o = payload["overall"]
        rows.append(
            [
                setup,
                f"{o['coverage']:.1%}",
                formatting.format_pkr(o["mean_width_pkr"]),
                formatting.format_pkr(o["median_width_pkr"]),
                f"{o['mean_relative_width']:.0%}",
            ]
        )
    out = [_banner(metrics), f"Nominal coverage is {config.INTERVAL_LEVEL:.0%}.\n"]
    out.append(_table(["Split", "Empirical coverage", "Mean width", "Median width", "Mean width / estimate"], rows))
    for setup, payload in block.items():
        out.append("")
        out.append(f"Per city, {setup} split:\n")
        out.append(
            _table(
                ["City", "Test listings", "Coverage", "Mean width"],
                [
                    [r["city"], f"{r['n']:,}", f"{r['coverage']:.1%}", formatting.format_pkr(r["mean_width_pkr"])]
                    for r in payload["by_city"]
                ],
            )
        )
    return "\n".join(out)


def render_errors(metrics: dict) -> str:
    out = [_banner(metrics)]
    for setup in ("random", "time"):
        out.append(f"**{setup.capitalize()} split**\n")
        for line in metrics["setups"][setup]["error_analysis"]["weakest"]:
            out.append(f"- {line}")
        out.append("")
    return "\n".join(out)


def render_ablation(metrics: dict) -> str:
    ablation = metrics.get("encoding_ablation")
    if not ablation:
        return PLACEHOLDER
    labels = {"target_encoding": "Target encoding (smoothed, cross-fitted)", "one_hot": "One-hot"}
    rows = [
        [
            labels[k],
            f"{v['rmse_log']['mean']:.4f} ± {v['rmse_log']['std']:.4f}",
            f"{v['r2_log']['mean']:.4f}",
            f"{v['mdape']['mean']:.1%}",
        ]
        for k, v in ablation.items()
    ]
    return _banner(metrics) + _table(["Location encoding", "CV RMSE (log)", "CV R² (log)", "CV median % error"], rows)


def render_shap(metrics: dict) -> str:
    block = metrics.get("shap")
    if not block:
        return PLACEHOLDER
    rows = [[r["group"], f"{r['mean_abs_shap']:.4f}"] for r in block["group_importance"]]
    return (
        _banner(metrics)
        + f"Mean |SHAP| by input group for {block['model']} on {block['sample_size']:,} test listings "
        + "(log-price units).\n\n"
        + _table(["Input group", "Mean |SHAP|"], rows)
    )


def render_funnel() -> str:
    path = config.PATHS.data_quality_md
    if not path.exists():
        return PLACEHOLDER
    text = path.read_text(encoding="utf-8")
    match = re.search(r"## Cleaning funnel\s+(.*?)\n## ", text, re.S)
    banner = "> **Produced from SYNTHETIC data.**\n\n" if "SYNTHETIC" in text else ""
    return banner + (match.group(1).strip() if match else PLACEHOLDER)


def render_sections(metrics: dict | None) -> dict[str, str]:
    if metrics is None:
        return {name: PLACEHOLDER for name in SECTIONS if name != "FUNNEL"} | {"FUNNEL": render_funnel()}
    return {
        "RESULTS": render_results(metrics),
        "INTERVALS": render_intervals(metrics),
        "ERRORS": render_errors(metrics),
        "ABLATION": render_ablation(metrics),
        "SHAP": render_shap(metrics),
        "FUNNEL": render_funnel(),
    }


def inject(text: str, sections: dict[str, str]) -> str:
    for name, body in sections.items():
        pattern = re.compile(rf"(<!-- BEGIN:{name} -->)(.*?)(<!-- END:{name} -->)", re.S)
        if not pattern.search(text):
            continue
        text = pattern.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text)
    return text


def main(argv: list[str] | None = None) -> int:
    path = config.PATHS.metrics_json
    metrics = json.loads(path.read_text()) if path.exists() else None
    sections = render_sections(metrics)
    updated = []
    for target in (config.PATHS.readme, config.PATHS.model_card_md):
        if target.exists():
            target.write_text(inject(target.read_text(encoding="utf-8"), sections), encoding="utf-8")
            updated.append(str(target))
    if not updated:
        print("No README.md or reports/model_card.md found", file=sys.stderr)
        return 1
    print(
        "Updated tables in " + ", ".join(updated) + ("" if metrics else " (no metrics.json yet, placeholders written)")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
