import json
import shutil
from pathlib import Path

import pytest

from house_prices import config
from scripts import update_readme

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def docs_root(pipeline_copy):
    shutil.copy(REPO / "README.md", config.PATHS.readme)
    config.PATHS.model_card_md.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "reports" / "model_card.md", config.PATHS.model_card_md)
    return pipeline_copy


def test_committed_readme_and_card_have_every_marker():
    for path in (REPO / "README.md", REPO / "reports" / "model_card.md"):
        text = path.read_text(encoding="utf-8")
        assert "<!-- BEGIN:RESULTS -->" in text and "<!-- END:RESULTS -->" in text
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    for name in update_readme.SECTIONS:
        assert f"<!-- BEGIN:{name} -->" in readme


def test_tables_are_filled_from_metrics_json(docs_root):
    assert update_readme.main([]) == 0
    text = config.PATHS.readme.read_text(encoding="utf-8")
    metrics = json.loads(config.PATHS.metrics_json.read_text())
    assert "SYNTHETIC" in text
    for setup in ("random", "time"):
        payload = metrics["setups"][setup]
        chosen = payload["models"][payload["chosen_model"]]["test"]
        assert f"{chosen['mdape']:.1%}" in text
        assert f"{chosen['r2_log']:.3f}" in text
    coverage = metrics["intervals"]["random"]["overall"]["coverage"]
    assert f"{coverage:.1%}" in text
    assert "Baseline: overall median price" in text
    assert "| removed" not in text
    assert "for_sale" in text
    assert "Not generated yet" not in text


def test_model_card_tables_are_filled_too(docs_root):
    update_readme.main([])
    card = config.PATHS.model_card_md.read_text(encoding="utf-8")
    assert "Random forest" in card or "XGBoost" in card or "LightGBM" in card
    assert "Not generated yet" not in card


def test_update_is_idempotent(docs_root):
    update_readme.main([])
    first = config.PATHS.readme.read_text(encoding="utf-8")
    update_readme.main([])
    assert config.PATHS.readme.read_text(encoding="utf-8") == first


def test_text_outside_markers_is_untouched(docs_root):
    before = config.PATHS.readme.read_text(encoding="utf-8")
    update_readme.main([])
    after = config.PATHS.readme.read_text(encoding="utf-8")
    assert before.split("<!-- BEGIN:RESULTS -->")[0] == after.split("<!-- BEGIN:RESULTS -->")[0]
    assert before.split("<!-- END:FUNNEL -->")[1] == after.split("<!-- END:FUNNEL -->")[1]


def test_placeholders_when_no_metrics(tmp_root):
    shutil.copy(REPO / "README.md", config.PATHS.readme)
    assert update_readme.main([]) == 0
    assert "Not generated yet" in config.PATHS.readme.read_text(encoding="utf-8")


def test_every_number_in_the_results_table_comes_from_metrics(docs_root):
    metrics = json.loads(config.PATHS.metrics_json.read_text())
    section = update_readme.render_results(metrics)
    for setup in ("random", "time"):
        for entry in metrics["setups"][setup]["models"].values():
            assert f"{entry['test']['mdape']:.1%}" in section
