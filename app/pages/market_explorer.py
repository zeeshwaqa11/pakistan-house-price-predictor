import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402
from common import disclaimer, listings_or_none  # noqa: E402

from house_prices import config, formatting  # noqa: E402

st.title("Market explorer")
st.write("Explore the cleaned asking-price data behind the model.")

listings = listings_or_none()
if listings is None:
    st.error("The cleaned listings were not found.")
    st.code("python -m house_prices.clean", language="bash")
    disclaimer()
    st.stop()

listings = listings.assign(ppsf=listings["price"] / listings["area_sqft"])
all_cities = sorted(listings["city"].unique())
chosen = st.multiselect("Cities", all_cities, default=all_cities)
if not chosen:
    st.info("Pick at least one city.")
    disclaimer()
    st.stop()
data = listings[listings["city"].isin(chosen)]

c1, c2, c3 = st.columns(3)
c1.metric("Listings", f"{len(data):,}")
c2.metric("Median asking price", formatting.format_pkr(float(data["price"].median())))
c3.metric("Median price per sq ft", formatting.format_ppsf(float(data["ppsf"].median())))
st.caption(f"Listings added {data['date_added'].min().date()} to {data['date_added'].max().date()}.")

st.subheader("Median price per sq ft by location")
spots = (
    data.dropna(subset=["latitude", "longitude"])
    .groupby(["city", "location"])
    .agg(
        latitude=("latitude", "median"),
        longitude=("longitude", "median"),
        ppsf=("ppsf", "median"),
        listings=("ppsf", "size"),
    )
    .reset_index()
)
spots = spots[spots["listings"] >= 5]
if spots.empty:
    st.info("Not enough located listings to draw the map for this selection.")
else:
    figure = px.scatter_map(
        spots,
        lat="latitude",
        lon="longitude",
        color="ppsf",
        size="listings",
        size_max=20,
        hover_name="location",
        hover_data={"city": True, "ppsf": ":,.0f", "listings": True, "latitude": False, "longitude": False},
        color_continuous_scale="Viridis",
        zoom=4.5 if len(chosen) > 1 else 9,
        center={"lat": float(spots["latitude"].mean()), "lon": float(spots["longitude"].mean())},
        map_style="open-street-map",
        height=560,
        labels={"ppsf": "PKR per sq ft"},
    )
    figure.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
    st.plotly_chart(figure, width="stretch")

left, right = st.columns(2)
with left:
    st.subheader("Price per sq ft by city")
    fig_city = px.box(
        data,
        x="city",
        y="ppsf",
        log_y=True,
        labels={"ppsf": "PKR per sq ft (log scale)", "city": ""},
        color="city",
    )
    fig_city.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig_city, width="stretch")
with right:
    st.subheader("Price per sq ft by property type")
    fig_type = px.box(
        data,
        x="property_type",
        y="ppsf",
        log_y=True,
        labels={"ppsf": "PKR per sq ft (log scale)", "property_type": ""},
        color="property_type",
    )
    fig_type.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig_type, width="stretch")

st.subheader("Most and least expensive locations")
min_count = st.slider("Minimum listings per location", 5, 100, config.MIN_LOCATION_COUNT, step=5)
ranking = (
    data.groupby(["city", "location"])["ppsf"]
    .agg(["median", "size"])
    .reset_index()
    .rename(columns={"median": "median_ppsf", "size": "listings"})
)
ranking = ranking[ranking["listings"] >= min_count]
if ranking.empty:
    st.info("No location has that many listings in this selection.")
else:
    ranking["label"] = ranking["location"] + " (" + ranking["city"] + ")"
    top = ranking.nlargest(15, "median_ppsf").sort_values("median_ppsf")
    bottom = ranking.nsmallest(15, "median_ppsf").sort_values("median_ppsf", ascending=False)
    a, b = st.columns(2)
    a.plotly_chart(
        px.bar(
            top,
            x="median_ppsf",
            y="label",
            orientation="h",
            title="Highest",
            labels={"median_ppsf": "PKR per sq ft", "label": ""},
        ),
        width="stretch",
    )
    b.plotly_chart(
        px.bar(
            bottom,
            x="median_ppsf",
            y="label",
            orientation="h",
            title="Lowest",
            labels={"median_ppsf": "PKR per sq ft", "label": ""},
        ),
        width="stretch",
    )

st.subheader("Listings added over time")
monthly = (
    data.assign(month=data["date_added"].dt.to_period("M").dt.to_timestamp())
    .groupby(["month", "city"])
    .size()
    .reset_index(name="listings")
)
st.plotly_chart(px.bar(monthly, x="month", y="listings", color="city", labels={"month": ""}), width="stretch")
disclaimer()
