import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json  # noqa: E402

import streamlit as st  # noqa: E402
from common import disclaimer  # noqa: E402

from house_prices import config  # noqa: E402

st.title("About & limitations")

st.markdown(
    f"""
This project estimates the **asking price** of a property in Pakistan from its size, location, type, bedrooms and
bathrooms, and shows an 80% price range and the reasons behind the estimate.

**Data.** {config.DATASET_NAME} ({config.DATASET_SOURCE}). Only "For Sale" listings are used. Areas are converted to
square feet, prices are in PKR, and duplicates and implausible values are removed by documented rules
(`reports/data_quality.md`).

**Model.** Gradient-boosted or forest regressors predict log price. Location is target-encoded inside the
scikit-learn pipeline, so the encoding is fitted on training folds only. The range comes from LightGBM quantile
regression at the 10th and 90th percentiles.

### Limitations

- **Asking prices, not sale prices.** Listings usually ask for more than the final deal.
- **A limited time period.** The data covers a short window; prices since then have changed with inflation, so
  today's market can sit well above what the model has seen.
- **Uneven coverage.** Large cities and popular areas have many listings; small cities and rare locations have few,
  and estimates there are less reliable. The app warns when a location is rare.
- **Marla is not one fixed size.** The model uses the conversion stored with the trained model (shown on the Predict
  page). If the true local Marla differs, size-based conversions shift accordingly.
- **Missing details.** Age, condition, floor, furnishing, corner plots and road width are not in the data, so two
  listings with identical inputs can have very different prices.
- **Not advice.** Use it to learn how the modelling works, not to price a property.
"""
)

metadata_path = config.PATHS.metadata_file
if metadata_path.exists():
    metadata = json.loads(metadata_path.read_text())
    st.subheader("This model")
    st.markdown(
        f"- Model: **{metadata['model_label']}** ({metadata['setup']} split)\n"
        f"- Trained: {metadata['training_date'][:10]} on {metadata['train_rows']:,} listings\n"
        f"- Listings dated {metadata['train_date_range'][0]} to {metadata['train_date_range'][1]}\n"
        f"- Sq ft per Marla: {metadata['marla_sqft']:g}\n"
        f"- Data file SHA-256: `{metadata['data_file_sha256']}`"
    )
    if metadata.get("synthetic_data"):
        st.warning("This model was trained on SYNTHETIC test data and is not a real result.")

disclaimer()
