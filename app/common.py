import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
SRC_DIR = APP_DIR.parent / "src"
for candidate in (str(SRC_DIR), str(APP_DIR.parent)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from house_prices import config, predict  # noqa: E402
from house_prices.evaluate import read_metrics  # noqa: E402


@st.cache_resource(show_spinner="Loading the model")
def get_bundle(root: str, stamp: float):
    return predict.load_bundle()


def bundle_or_stop():
    if not predict.model_files_exist():
        st.error("The trained model was not found, so predictions are not available yet.")
        st.markdown("Run these commands from the project folder (with the virtual environment active), then reload:")
        st.code(predict.TRAIN_COMMANDS, language="bash")
        st.caption("The raw data file must be at `data/raw/pakistan_property.csv` first. See the README.")
        disclaimer()
        st.stop()
    stamp = config.PATHS.model_file.stat().st_mtime
    return get_bundle(str(config.PATHS.root), stamp)


@st.cache_data(show_spinner="Loading listings")
def get_listings(root: str, stamp: float) -> pd.DataFrame:
    return pd.read_parquet(config.PATHS.listings)


def listings_or_none() -> pd.DataFrame | None:
    path = config.PATHS.listings
    if not path.exists():
        return None
    return get_listings(str(config.PATHS.root), path.stat().st_mtime)


def metrics_or_none() -> dict | None:
    return read_metrics()


def disclaimer():
    st.divider()
    st.caption("**Disclaimer.** " + config.DISCLAIMER)


def synthetic_banner(metadata: dict | None):
    if metadata and metadata.get("synthetic_data"):
        st.warning(
            "This model was trained on SYNTHETIC test data, not on real listings. Nothing shown here is a real result."
        )
