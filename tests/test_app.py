from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from house_prices import predict

APP_DIR = Path(__file__).resolve().parents[1] / "app"
TIMEOUT = 120


def run_page(name):
    at = AppTest.from_file(str(APP_DIR / "pages" / name), default_timeout=TIMEOUT)
    return at.run()


def test_predict_page_renders_an_estimate(pipeline_env):
    at = run_page("predict.py")
    assert not at.exception
    assert at.title[0].value == "Pakistan House Price Predictor"
    text = " ".join(m.value for m in at.markdown)
    assert "PKR" in text
    assert "80% range" in text
    assert any("Disclaimer" in c.value for c in at.caption)
    assert any("SYNTHETIC" in w.value for w in at.warning)
    assert len(at.dataframe) == 1
    assert len(at.button) >= 2


def test_predict_location_list_follows_city(pipeline_env):
    at = run_page("predict.py")
    cities = at.selectbox(key="city").options
    assert len(cities) >= 2
    other = cities[1]
    at.selectbox(key="city").select(other).run()
    assert not at.exception
    bundle = predict.load_bundle()
    assert set(at.selectbox(key="location").options) <= set(bundle.locations_for(other))


def test_predict_location_search_filters_options(pipeline_env):
    at = run_page("predict.py")
    all_options = at.selectbox(key="location").options
    fragment = all_options[0][-2:]
    at.text_input(key="location_search").input(fragment).run()
    filtered = at.selectbox(key="location").options
    assert all(fragment.lower() in o.lower() for o in filtered)
    assert len(filtered) <= len(all_options)


def test_predict_unit_selector_and_presets(pipeline_env):
    at = run_page("predict.py")
    at.radio(key="unit").set_value("Kanal").run()
    assert not at.exception
    assert at.number_input(key="size_Kanal").value >= 0.25
    at.button[0].click().run()
    assert not at.exception
    assert at.radio(key="unit").value in ("Marla", "Kanal", "Sq. Ft.")


def test_predict_page_without_model_shows_command(tmp_root):
    at = run_page("predict.py")
    assert not at.exception
    assert any("not found" in e.value for e in at.error)
    assert any("python -m house_prices.train" in c.value for c in at.code)


@pytest.mark.parametrize("page", ["market_explorer.py", "model_performance.py", "about.py"])
def test_other_pages_render_with_disclaimer(pipeline_env, page):
    at = run_page(page)
    assert not at.exception
    assert len(at.title) == 1
    assert any("Disclaimer" in c.value for c in at.caption)


def test_market_explorer_city_filter(pipeline_env):
    at = run_page("market_explorer.py")
    options = at.multiselect[0].options
    at.multiselect[0].set_value([options[0]]).run()
    assert not at.exception
    at.multiselect[0].set_value([]).run()
    assert any("at least one city" in i.value for i in at.info)


def test_model_performance_reads_metrics_json(pipeline_env):
    at = run_page("model_performance.py")
    assert len(at.tabs) == 2
    assert len(at.dataframe) >= 4
    assert any("SYNTHETIC" in w.value for w in at.warning)


@pytest.mark.parametrize("page", ["market_explorer.py", "model_performance.py"])
def test_pages_without_artifacts_are_friendly(tmp_root, page):
    at = run_page(page)
    assert not at.exception
    assert len(at.error) >= 1
    assert any("Disclaimer" in c.value for c in at.caption)


def test_entry_point_with_navigation(pipeline_env):
    at = AppTest.from_file(str(APP_DIR / "streamlit_app.py"), default_timeout=TIMEOUT).run()
    assert not at.exception
