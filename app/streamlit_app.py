import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(page_title="Pakistan House Price Predictor", page_icon="🏠", layout="wide")

pages = [
    st.Page("pages/predict.py", title="Predict", icon="🏠", default=True),
    st.Page("pages/market_explorer.py", title="Market explorer", icon="🗺️"),
    st.Page("pages/model_performance.py", title="Model performance", icon="📊"),
    st.Page("pages/about.py", title="About & limitations", icon="ℹ️"),
]
st.navigation(pages).run()
