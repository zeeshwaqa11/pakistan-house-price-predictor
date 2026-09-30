import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from common import bundle_or_stop, disclaimer, listings_or_none, synthetic_banner  # noqa: E402

from house_prices import config, formatting, predict  # noqa: E402

SIZE_RULES = {
    "Marla": {"min": 1.0, "max": 400.0, "step": 0.5, "default": 5.0},
    "Kanal": {"min": 0.25, "max": 100.0, "step": 0.25, "default": 1.0},
    "Sq. Ft.": {"min": 100.0, "max": 200000.0, "step": 50.0, "default": 1200.0},
}


def apply_preset(preset: dict):
    st.session_state["city"] = preset["city"]
    st.session_state["location_search"] = ""
    st.session_state["location"] = preset["location"]
    st.session_state["ptype"] = preset["property_type"]
    st.session_state["unit"] = preset["unit"]
    st.session_state[f"size_{preset['unit']}"] = float(preset["size"])
    st.session_state["beds"] = int(preset["bedrooms"])
    st.session_state["baths"] = int(preset["baths"])


def init_state(bundle, presets):
    if "city" in st.session_state:
        return
    if presets:
        apply_preset(presets[0])
        return
    city = bundle.cities[0]
    st.session_state.update(
        {
            "city": city,
            "location_search": "",
            "location": bundle.locations_for(city)[0],
            "ptype": bundle.property_types[0],
            "unit": "Marla",
            "beds": 3,
            "baths": 2,
        }
    )


st.title("Pakistan House Price Predictor")
st.write("Estimate the asking price of a property from its size, location, type, bedrooms and bathrooms.")

bundle = bundle_or_stop()
synthetic_banner(bundle.metadata)
presets = predict.preset_examples(bundle)
init_state(bundle, presets)

if presets:
    st.caption("Try an example")
    preset_columns = st.columns(len(presets))
    for column, preset in zip(preset_columns, presets):
        column.button(
            preset["title"],
            key=f"preset_{preset['title']}",
            on_click=apply_preset,
            args=(preset,),
            width="stretch",
        )

left, right = st.columns([1, 1.4], gap="large")

with left:
    st.subheader("Property details")
    city = st.selectbox("City", bundle.cities, key="city")
    all_locations = bundle.locations_for(city)
    st.text_input("Search locations", key="location_search", placeholder="Type to narrow the list")
    query = st.session_state.get("location_search", "").strip().lower()
    options = [loc for loc in all_locations if query in loc.lower()] or all_locations
    if st.session_state.get("location") not in options:
        st.session_state["location"] = options[0]
    location = st.selectbox("Location", options, key="location")
    property_type = st.selectbox("Property type", bundle.property_types, key="ptype")
    unit = st.radio("Size unit", list(SIZE_RULES), horizontal=True, key="unit")
    rules = SIZE_RULES[unit]
    size_key = f"size_{unit}"
    if size_key not in st.session_state:
        st.session_state[size_key] = rules["default"]
    size = st.number_input(
        f"Size ({unit})", min_value=rules["min"], max_value=rules["max"], step=rules["step"], key=size_key
    )
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=15, step=1, key="beds")
    baths = st.number_input("Bathrooms", min_value=1, max_value=15, step=1, key="baths")
    st.caption(
        f"1 Marla = {bundle.marla_sqft:g} sq ft and 1 Kanal = {bundle.marla_sqft * config.KANAL_IN_MARLA:,g} sq ft "
        "in this model."
    )

with right:
    problems = predict.validate_inputs(bundle, city, location, property_type, size, unit, bedrooms, baths)
    if problems:
        for problem in problems:
            st.error(problem)
        disclaimer()
        st.stop()
    prediction = predict.predict(bundle, city, location, property_type, size, unit, bedrooms, baths)
    st.subheader("Estimated asking price")
    st.markdown(f"## {formatting.format_pkr(prediction.estimate)}")
    st.caption(formatting.format_exact(prediction.estimate))
    st.markdown(
        f"**80% range:** {formatting.format_range(prediction.lower, prediction.upper)}  \n"
        f"<span style='opacity:0.7'>{formatting.format_exact(prediction.lower)} to "
        f"{formatting.format_exact(prediction.upper)}</span>",
        unsafe_allow_html=True,
    )
    st.caption("About 8 in 10 similar listings had an asking price inside a range like this in testing.")
    metric_a, metric_b = st.columns(2)
    metric_a.metric("Price per sq ft", formatting.format_exact(prediction.ppsf))
    metric_b.metric("Size in sq ft", f"{prediction.area_sqft:,.0f}")
    if prediction.is_rare:
        if prediction.grouped_as_other:
            st.warning(
                f"Only {prediction.location_listings} listings in the training data for {location}. The model "
                "treats it as part of 'Other' locations in this city, so this estimate is less reliable."
            )
        else:
            st.warning(
                f"{location} has only {prediction.location_listings} training listings, so this estimate is "
                "less reliable than for well-covered areas."
            )

st.divider()
st.subheader("Why this estimate?")
area_label = formatting.format_area(prediction.area_sqft, bundle.marla_sqft)
explanation = predict.explain_inputs(bundle, prediction, area_label)
st.markdown(f"**{explanation['sentence']}**")
st.pyplot(explanation["figure"], clear_figure=True)
plt.close("all")
st.caption(
    "Each bar shows how much a group of inputs moved the estimate away from the average listing "
    f"(about {formatting.format_pkr(explanation['base_price'])}). Values are on the log-price scale: "
    "+0.1 is roughly +10%."
)

st.subheader("Comparable listings")
listings = listings_or_none()
if listings is None:
    st.info("Comparable listings need the cleaned data. Run `python -m house_prices.clean` first.")
else:
    comparables = predict.find_comparables(listings, prediction)
    if comparables.empty:
        st.info("No comparable listings were found.")
    else:
        table = pd.DataFrame(
            {
                "Location": comparables["location"],
                "Type": comparables["property_type"],
                "Size": [formatting.format_area(a, bundle.marla_sqft) for a in comparables["area_sqft"]],
                "Beds": comparables["bedrooms"],
                "Baths": comparables["baths"],
                "Asking price": [formatting.format_pkr(p) for p in comparables["price"]],
                "Per sq ft": [formatting.format_ppsf(p) for p in comparables["price_per_sqft"]],
                "Distance (km)": comparables["distance_km"].round(1),
                "Listed": pd.to_datetime(comparables["date_added"]).dt.strftime("%Y-%m-%d"),
            }
        )
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption("Closest listings by location, size and type from the cleaned data. These are asking prices.")

disclaimer()
