import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def md(text: str):
    return new_markdown_cell(text.strip("\n"))


def code(text: str):
    return new_code_cell(text.strip("\n"))


SETUP = code(
    """
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from house_prices import clean, config, load

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 200)
sns.set_theme(style="whitegrid", context="notebook")
config.PATHS.figures_dir.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name):
    fig.savefig(config.PATHS.figures_dir / f"{name}.png", dpi=140, bbox_inches="tight")
"""
)


def audit_notebook():
    cells = [
        md(
            """
# 01 Data audit

Load the raw file, print its real schema, check units, price formats, dates and missing values, then run the
documented cleaning rules and look at the funnel. Nothing here assumes column names: the mapping in
`config.COLUMN_ALIASES` is printed so it can be checked against the file.
"""
        ),
        SETUP,
        md("## Load and inspect the raw file"),
        code(
            """
raw = load.load_raw()
print(load.schema_report(raw))
print()
print(load.mapping_report(raw))
synthetic = load.is_synthetic(load.apply_schema(raw))
print()
print("Synthetic sample file:", synthetic)
"""
        ),
        md("## Canonical view of the columns the project uses"),
        code(
            """
canonical = load.apply_schema(raw)
canonical.head()
"""
        ),
        md(
            """
## Purpose, property types and cities

Only `For Sale` listings are modelled; rentals are a different problem with different price scales.
"""
        ),
        code(
            """
print(canonical["purpose"].value_counts(dropna=False).to_string())
print()
print(canonical["property_type"].value_counts(dropna=False).to_string())
print()
print(canonical["city"].value_counts(dropna=False).to_string())
"""
        ),
        md("## Area units and price formats"),
        code(
            """
print(canonical["area_unit"].value_counts(dropna=False).to_string())
print()
text_prices = clean.units.count_text_prices(canonical["price"])
print(f"Price column dtype: {canonical['price'].dtype}; rows stored as text: {text_prices:,}")
if text_prices:
    display(canonical.loc[pd.to_numeric(canonical["price"], errors="coerce").isna(), "price"].head(10))
"""
        ),
        md(
            """
## Dates

`date_added` is parsed month-first. If any value has a first component above 12 the assumption is wrong and
`clean.parse_dates` must be changed.
"""
        ),
        code(
            """
parsed_dates = clean.parse_dates(canonical["date_added"])
first_part = canonical["date_added"].astype(str).str.extract(r"^(\\d+)[/-]", expand=False).astype(float)
print("Unparseable dates:", int(parsed_dates.isna().sum()))
print("Earliest:", parsed_dates.min(), " Latest:", parsed_dates.max())
print("Rows whose first component exceeds 12 (would contradict month-first):", int((first_part > 12).sum()))
"""
        ),
        md("## Missing values in the modelling columns"),
        code(
            """
cols = ["property_type", "price", "location", "city", "latitude", "longitude", "baths", "bedrooms", "date_added"]
missing = canonical[cols].isna().mean().sort_values(ascending=False) * 100
fig, ax = plt.subplots(figsize=(7, 3.5))
sns.barplot(x=missing.values, y=missing.index, ax=ax, color="#4c78a8")
ax.set_xlabel("Missing (%)")
ax.set_title("Missing values by column (raw file)")
save_fig(fig, "audit_missing")
plt.show()
"""
        ),
        md("## Run the cleaning rules"),
        code(
            """
settings = config.CleaningSettings()
cleaned, funnel, info, parsed = clean.clean(canonical, settings)
funnel_df = pd.DataFrame(funnel)[["key", "removed", "modified", "remaining", "description"]]
display(funnel_df)
print()
print(f"Sq ft per Marla: {info['marla_sqft']:.2f} ({info['marla_source']})")
print(f"Raw rows {len(raw):,} -> cleaned rows {len(cleaned):,} ({len(cleaned) / len(raw):.1%})")
"""
        ),
        code(
            """
steps = funnel_df[funnel_df["removed"] > 0].sort_values("removed")
fig, ax = plt.subplots(figsize=(8, 0.45 * len(steps) + 1.5))
sns.barplot(x="removed", y="key", data=steps, ax=ax, color="#e45756")
ax.set_xlabel("Rows removed")
ax.set_ylabel("")
ax.set_title("Rows removed by each cleaning rule")
save_fig(fig, "audit_funnel")
plt.show()
"""
        ),
        md(
            """
## Price per sq ft before and after the outlier rule

The per-city IQR fence on log price per sq ft is applied last. The plot shows what it removes.
"""
        ),
        code(
            """
before = parsed[(parsed["purpose"] == config.PURPOSE_SALE) & (parsed["price"] > 0) & (parsed["area_sqft"] > 0)].copy()
before["ppsf"] = before["price"] / before["area_sqft"]
after = cleaned.assign(ppsf=cleaned["price"] / cleaned["area_sqft"])
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, frame, title in zip(axes, (before, after), ("Before outlier rule", "After cleaning")):
    sns.boxplot(data=frame, x="city", y="ppsf", ax=ax, color="#9ecae9", fliersize=1)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("")
axes[0].set_ylabel("PKR per sq ft (log scale)")
save_fig(fig, "audit_ppsf_before_after")
plt.show()
"""
        ),
        md("## Saved report"),
        code(
            """
report = config.PATHS.data_quality_md
print(report.read_text(encoding="utf-8")[:1500] if report.exists() else "Run `python -m house_prices.clean` first.")
"""
        ),
    ]
    return new_notebook(cells=cells)


def eda_notebook():
    cells = [
        md(
            """
# 02 Exploratory analysis

Reads the cleaned listings written by `python -m house_prices.clean`. Every statement in the final section is
computed from the data shown above it.
"""
        ),
        SETUP,
        code(
            """
import plotly.express as px

listings = pd.read_parquet(config.PATHS.listings)
listings["ppsf"] = listings["price"] / listings["area_sqft"]
listings["price_crore"] = listings["price"] / 1e7
print(f"{len(listings):,} cleaned listings, {listings['date_added'].min().date()} to {listings['date_added'].max().date()}")
listings.head()
"""
        ),
        md("## Price distribution: linear and log scale"),
        code(
            """
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
cap = listings["price_crore"].quantile(0.99)
sns.histplot(listings.loc[listings["price_crore"] <= cap, "price_crore"], bins=60, ax=axes[0], color="#4c78a8")
axes[0].set_xlabel("Price (PKR Crore), clipped at the 99th percentile")
axes[0].set_title("Linear scale")
sns.histplot(np.log10(listings["price"]), bins=60, ax=axes[1], color="#f58518")
axes[1].set_xlabel("log10(price in PKR)")
axes[1].set_title("Log scale")
save_fig(fig, "eda_price_distribution")
plt.show()
print(f"Skewness of price: {listings['price'].skew():.2f}; skewness of log price: {np.log(listings['price']).skew():.2f}")
"""
        ),
        md("## Price per sq ft by city and by property type"),
        code(
            """
order = listings.groupby("city")["ppsf"].median().sort_values(ascending=False).index
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
sns.boxplot(data=listings, x="city", y="ppsf", order=order, ax=axes[0], color="#9ecae9", fliersize=1)
axes[0].set_yscale("log")
axes[0].set_ylabel("PKR per sq ft (log scale)")
axes[0].set_xlabel("")
axes[0].set_title("By city")
type_order = listings.groupby("property_type")["ppsf"].median().sort_values(ascending=False).index
sns.boxplot(data=listings, x="property_type", y="ppsf", order=type_order, ax=axes[1], color="#a1d99b", fliersize=1)
axes[1].set_yscale("log")
axes[1].set_ylabel("")
axes[1].set_xlabel("")
axes[1].set_title("By property type")
plt.setp(axes[1].get_xticklabels(), rotation=25, ha="right")
save_fig(fig, "eda_ppsf_city_type")
plt.show()
display(listings.groupby("city")["ppsf"].agg(["count", "median"]).round(0).sort_values("median", ascending=False))
"""
        ),
        md(
            """
## Most and least expensive locations

Median price per sq ft for Karachi, Lahore and Islamabad, only for locations with at least
`MIN_LOCATION_COUNT` listings so a handful of listings cannot decide the ranking.
"""
        ),
        code(
            """
min_count = config.MIN_LOCATION_COUNT
focus = [c for c in ("Karachi", "Lahore", "Islamabad") if c in set(listings["city"])]
fig, axes = plt.subplots(len(focus), 2, figsize=(13, 4.6 * len(focus)), squeeze=False)
location_tables = {}
for row, city in enumerate(focus):
    sub = listings[listings["city"] == city]
    stats = sub.groupby("location")["ppsf"].agg(["count", "median"])
    stats = stats[stats["count"] >= min_count].sort_values("median", ascending=False)
    location_tables[city] = stats
    n = min(15, len(stats))
    top, bottom = stats.head(n), stats.tail(n).sort_values("median")
    sns.barplot(x=top["median"], y=top.index, ax=axes[row, 0], color="#e45756")
    axes[row, 0].set_title(f"{city}: top {n} by median PKR per sq ft")
    sns.barplot(x=bottom["median"], y=bottom.index, ax=axes[row, 1], color="#4c78a8")
    axes[row, 1].set_title(f"{city}: bottom {n}")
    for ax in axes[row]:
        ax.set_xlabel("Median PKR per sq ft")
        ax.set_ylabel("")
fig.tight_layout()
save_fig(fig, "eda_location_rankings")
plt.show()
print({city: len(t) for city, t in location_tables.items()}, "locations with enough listings")
"""
        ),
        md("## Price versus size and versus bedrooms"),
        code(
            """
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
sample = listings.sample(min(len(listings), 20000), random_state=config.RANDOM_STATE)
sns.scatterplot(data=sample, x="area_sqft", y="price", hue="city", s=8, alpha=0.4, ax=axes[0])
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("Size (sq ft, log scale)")
axes[0].set_ylabel("Price (PKR, log scale)")
axes[0].set_title("Price versus size")
bed = listings[listings["bedrooms"] <= 10]
sns.boxplot(data=bed, x="bedrooms", y="price", ax=axes[1], color="#9ecae9", fliersize=1)
axes[1].set_yscale("log")
axes[1].set_title("Price versus bedrooms")
axes[1].set_ylabel("Price (PKR, log scale)")
save_fig(fig, "eda_price_size_bedrooms")
plt.show()
corr = pd.Series(
    {
        "log price vs log size": np.corrcoef(np.log(listings["price"]), np.log(listings["area_sqft"]))[0, 1],
        "log price vs bedrooms": np.corrcoef(np.log(listings["price"]), listings["bedrooms"])[0, 1],
        "log price vs baths": np.corrcoef(np.log(listings["price"]), listings["baths"])[0, 1],
        "log size vs bedrooms": np.corrcoef(np.log(listings["area_sqft"]), listings["bedrooms"])[0, 1],
    }
)
spearman = listings[["price", "area_sqft", "bedrooms", "baths"]].corr(method="spearman")["price"]
display(corr.round(3).to_frame("Pearson"))
display(spearman.round(3).to_frame("Spearman with price"))
"""
        ),
        md(
            """
Bedrooms and size move together (see the last Pearson row), so bedrooms carry a good part of the size signal;
the modelling notebook checks how much each adds on its own.
"""
        ),
        md("## Map of median price per sq ft"),
        code(
            """
geo = listings.dropna(subset=["latitude", "longitude"])
spots = (
    geo.groupby(["city", "location"])
    .agg(latitude=("latitude", "median"), longitude=("longitude", "median"), ppsf=("ppsf", "median"), listings=("ppsf", "size"))
    .reset_index()
)
spots = spots[spots["listings"] >= 5]
print(f"{len(spots):,} locations plotted from {len(geo):,} listings with coordinates")
fig_map = px.scatter_map(
    spots,
    lat="latitude",
    lon="longitude",
    color="ppsf",
    size="listings",
    size_max=22,
    hover_name="location",
    hover_data={"city": True, "ppsf": ":,.0f", "listings": True, "latitude": False, "longitude": False},
    color_continuous_scale="Viridis",
    zoom=4.3,
    center={"lat": 30.3, "lon": 70.0},
    map_style="open-street-map",
    height=620,
    title="Median PKR per sq ft by location",
)
fig_map.update_layout(margin={"l": 0, "r": 0, "t": 40, "b": 0})
fig_map.write_html(config.PATHS.figures_dir / "price_map.html", include_plotlyjs="cdn")
fig_map.show()
"""
        ),
        code(
            """
fig, ax = plt.subplots(figsize=(6.5, 7.5))
scatter = ax.scatter(spots["longitude"], spots["latitude"], c=np.log10(spots["ppsf"]), s=np.sqrt(spots["listings"]) * 6, cmap="viridis", alpha=0.75)
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label("log10(median PKR per sq ft)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Median price per sq ft by location (marker size = listings)")
ax.set_aspect(1.15)
save_fig(fig, "price_map")
plt.show()
"""
        ),
        md("## Listing counts over time"),
        code(
            """
monthly = listings.assign(month=listings["date_added"].dt.to_period("M").dt.to_timestamp())
counts = monthly.groupby(["month", "city"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(11, 4))
counts.plot.bar(stacked=True, ax=ax, width=0.85)
ax.set_xticklabels([d.strftime("%Y-%m") for d in counts.index], rotation=45, ha="right")
ax.set_ylabel("Listings added")
ax.set_xlabel("")
ax.set_title("Cleaned listings by month added")
save_fig(fig, "eda_listings_over_time")
plt.show()
print(f"Data covers {listings['date_added'].min().date()} to {listings['date_added'].max().date()} "
      f"({(listings['date_added'].max() - listings['date_added'].min()).days} days)")
"""
        ),
        md("## What the data shows"),
        md(
            "Each sentence below is generated from the numbers plotted above, so it can only report what the charts contain."
        ),
        code(
            """
by_city = listings.groupby("city")["ppsf"].median().sort_values(ascending=False)
by_type = listings.groupby("property_type")["ppsf"].median().sort_values(ascending=False)
share = listings["city"].value_counts(normalize=True)
days = (listings["date_added"].max() - listings["date_added"].min()).days
lines = [
    f"The file has {len(listings):,} cleaned for-sale listings added between {listings['date_added'].min().date()} and {listings['date_added'].max().date()} ({days} days).",
    f"{share.index[0]} has the most listings ({share.iloc[0]:.0%}); {share.index[-1]} has the fewest ({share.iloc[-1]:.0%}).",
    f"Median price per sq ft is highest in {by_city.index[0]} (PKR {by_city.iloc[0]:,.0f}) and lowest in {by_city.index[-1]} (PKR {by_city.iloc[-1]:,.0f}), a ratio of {by_city.iloc[0] / by_city.iloc[-1]:.1f}x.",
    f"By property type, median price per sq ft is highest for {by_type.index[0]} (PKR {by_type.iloc[0]:,.0f}) and lowest for {by_type.index[-1]} (PKR {by_type.iloc[-1]:,.0f}).",
    f"Price is right-skewed (skewness {listings['price'].skew():.1f}); on a log scale the skewness is {np.log(listings['price']).skew():.1f}, which is why the model predicts log price.",
    f"Log price and log size have a Pearson correlation of {corr['log price vs log size']:.2f}; log price and bedrooms {corr['log price vs bedrooms']:.2f}.",
]
for city, stats in location_tables.items():
    if len(stats) >= 2:
        lines.append(
            f"Within {city}, the priciest well-covered location is {stats.index[0]} (median PKR {stats['median'].iloc[0]:,.0f} per sq ft) and the cheapest is {stats.index[-1]} (PKR {stats['median'].iloc[-1]:,.0f}), {stats['median'].iloc[0] / stats['median'].iloc[-1]:.1f}x apart."
        )
print("\\n".join("- " + line for line in lines))
"""
        ),
    ]
    return new_notebook(cells=cells)


def modelling_notebook():
    from scripts.notebook_cells_modelling import build

    return build(md, code, SETUP)[0]


def error_notebook():
    from scripts.notebook_cells_modelling import build

    return build(md, code, SETUP)[1]


BUILDERS = {
    "01_data_audit.ipynb": audit_notebook,
    "02_eda.ipynb": eda_notebook,
    "03_modelling.ipynb": modelling_notebook,
    "04_error_analysis.ipynb": error_notebook,
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    NOTEBOOK_DIR.mkdir(exist_ok=True)
    for name, builder in BUILDERS.items():
        if argv and name not in argv:
            continue
        notebook = builder()
        notebook.metadata["kernelspec"] = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
        nbformat.write(notebook, NOTEBOOK_DIR / name)
        print("wrote", NOTEBOOK_DIR / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
