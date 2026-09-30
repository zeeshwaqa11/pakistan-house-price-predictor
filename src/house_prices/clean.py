import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from house_prices import config, load, units
from house_prices.config import CleaningSettings

PARSED_COLUMNS = [
    "listing_id",
    "property_type",
    "price",
    "location",
    "city",
    "province",
    "latitude",
    "longitude",
    "baths",
    "bedrooms",
    "area_sqft",
    "area_value",
    "area_unit_norm",
    "purpose",
    "date_added",
]

SECTOR_RE = re.compile(r"^[a-z]-\d+(/\d+)?$")


def standardise_location(name) -> str | None:
    if name is None or name is pd.NA or (isinstance(name, float) and np.isnan(name)):
        return None
    text = unicodedata.normalize("NFKC", str(name)).strip().lower()
    text = text.replace("_", " ").replace("&", " and ")
    text = re.sub(r"(?<=[a-z])\s*-\s*(?=[a-z])", " ", text)
    text = re.sub(r"[.,;:()]+", " ", text)
    for pattern, replacement in config.LOCATION_VARIANTS.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None
    tokens = []
    for token in text.split(" "):
        upper = token.upper()
        if upper in config.LOCATION_ACRONYMS or SECTOR_RE.match(token):
            tokens.append(upper)
        else:
            tokens.append(token.capitalize())
    return " ".join(tokens)


def standardise_label(value) -> str | None:
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()
    if not text:
        return None
    return text.title()


def parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", format="mixed")


def prepare(raw: pd.DataFrame, settings: CleaningSettings) -> tuple[pd.DataFrame, dict]:
    canonical = raw
    info = {"text_prices": units.count_text_prices(canonical["price"])}
    unit_norm = canonical["area_unit"].map(units.normalise_unit)
    marla_input = pd.DataFrame(
        {
            "price": units.parse_price_series(canonical["price"]),
            "area_value": pd.to_numeric(canonical["area_size"], errors="coerce"),
            "area_unit_norm": unit_norm,
            "city": canonical["city"].map(standardise_label),
            "location": canonical["location"].map(standardise_location),
            "property_type": canonical["property_type"].map(standardise_label),
        }
    )
    marla_sqft, marla_source = units.resolve_marla_factor(
        marla_input, settings.marla_sqft, settings.marla_min_groups, settings.marla_min_group_size
    )
    info["marla_sqft"] = marla_sqft
    info["marla_source"] = marla_source
    area_sqft = units.area_series_to_sqft(
        canonical["area_size"], canonical["area_unit"], canonical["area_text"], marla_sqft
    )
    unit_from_text = canonical["area_text"].map(lambda t: (units.parse_area_text(t) or [(None, None)])[0][1])
    parsed = pd.DataFrame(
        {
            "listing_id": canonical["listing_id"],
            "property_type": marla_input["property_type"],
            "price": marla_input["price"],
            "location": marla_input["location"],
            "city": marla_input["city"],
            "province": canonical["province"].map(standardise_label),
            "latitude": pd.to_numeric(canonical["latitude"], errors="coerce"),
            "longitude": pd.to_numeric(canonical["longitude"], errors="coerce"),
            "baths": pd.to_numeric(canonical["baths"], errors="coerce"),
            "bedrooms": pd.to_numeric(canonical["bedrooms"], errors="coerce"),
            "area_sqft": area_sqft,
            "area_value": marla_input["area_value"],
            "area_unit_norm": unit_norm.where(unit_norm.notna(), unit_from_text),
            "purpose": canonical["purpose"].astype("string").str.strip().str.lower(),
            "date_added": parse_dates(canonical["date_added"]),
        }
    )
    info["unit_counts"] = {str(k): int(v) for k, v in parsed["area_unit_norm"].value_counts(dropna=False).items()}
    return parsed[PARSED_COLUMNS].reset_index(drop=True), info


def rule_for_sale(df, s):
    return df["purpose"].fillna("") == config.PURPOSE_SALE


def rule_price_present(df, s):
    return df["price"].notna() & (df["price"] > 0)


def rule_price_floor(df, s):
    return df["price"] >= s.min_price_pkr


def rule_area_present(df, s):
    return df["area_sqft"].notna() & (df["area_sqft"] > 0)


def rule_area_bounds(df, s):
    return df["area_sqft"].between(s.min_area_sqft, s.max_area_sqft)


def rule_categories_present(df, s):
    return df[["city", "location", "property_type"]].notna().all(axis=1)


def rule_date_present(df, s):
    return df["date_added"].notna()


def rule_rooms_present(df, s):
    return df["bedrooms"].notna() & df["baths"].notna()


def rule_zero_bedrooms(df, s):
    residential = df["property_type"].fillna("").str.lower().isin(s.bedroom_required_types)
    return ~(residential & (df["bedrooms"] == 0))


def rule_absurd_rooms(df, s):
    ok_beds = df["bedrooms"].between(0, s.max_bedrooms)
    ok_baths = df["baths"].between(0, s.max_baths)
    return ok_beds & ok_baths


def rule_rare_city(df, s):
    counts = df["city"].map(df["city"].value_counts())
    return counts >= s.min_city_listings


def rule_rare_type(df, s):
    counts = df["property_type"].map(df["property_type"].value_counts())
    return counts >= s.min_type_listings


def rule_exact_duplicates(df, s):
    subset = [c for c in df.columns if c != "listing_id"]
    return ~df.duplicated(subset=subset, keep="first")


def rule_near_duplicates(df, s):
    key = pd.DataFrame(
        {
            "city": df["city"],
            "location": df["location"],
            "area": df["area_sqft"].round(0),
            "price": df["price"],
            "bedrooms": df["bedrooms"],
            "property_type": df["property_type"],
        }
    )
    return ~key.duplicated(keep="first")


def ppsf_bounds(df: pd.DataFrame, s: CleaningSettings) -> pd.DataFrame:
    log_ppsf = np.log(df["price"] / df["area_sqft"])
    global_q1, global_q3 = log_ppsf.quantile([0.25, 0.75])
    per_city = log_ppsf.groupby(df["city"]).agg(n="size", q1=lambda v: v.quantile(0.25), q3=lambda v: v.quantile(0.75))
    thin = per_city["n"] < s.ppsf_min_group
    per_city.loc[thin, "q1"] = global_q1
    per_city.loc[thin, "q3"] = global_q3
    iqr = per_city["q3"] - per_city["q1"]
    per_city["low"] = np.exp(per_city["q1"] - s.ppsf_iqr_k * iqr)
    per_city["high"] = np.exp(per_city["q3"] + s.ppsf_iqr_k * iqr)
    return per_city[["n", "low", "high"]]


def rule_ppsf_outliers(df, s):
    if df.empty:
        return pd.Series(dtype=bool, index=df.index)
    bounds = ppsf_bounds(df, s)
    ppsf = df["price"] / df["area_sqft"]
    low = df["city"].map(bounds["low"])
    high = df["city"].map(bounds["high"])
    return (ppsf >= low) & (ppsf <= high)


def transform_coordinates(df: pd.DataFrame, s: CleaningSettings) -> tuple[pd.DataFrame, int]:
    df = df.copy()
    lat, lon = df["latitude"], df["longitude"]
    lat_lo, lat_hi = config.PAKISTAN_BOUNDS["lat"]
    lon_lo, lon_hi = config.PAKISTAN_BOUNDS["lon"]
    bad = ~(lat.between(lat_lo, lat_hi) & lon.between(lon_lo, lon_hi))
    centre_lat = df["city"].map(lambda c: s.city_centres.get(c, (np.nan, np.nan))[0])
    centre_lon = df["city"].map(lambda c: s.city_centres.get(c, (np.nan, np.nan))[1])
    distance = units.haversine_km(lat, lon, centre_lat.astype(float), centre_lon.astype(float))
    far = pd.Series(distance > s.max_centre_km, index=df.index)
    reset = (bad | far) & (lat.notna() | lon.notna())
    df.loc[reset, ["latitude", "longitude"]] = np.nan
    return df, int(reset.sum())


@dataclass
class Step:
    key: str
    kind: str
    description: str
    filter_fn: Callable | None = None
    transform_fn: Callable | None = None


FILTER_STEPS = [
    Step("for_sale", "filter", "Purpose is not 'For Sale' (rentals are a different problem)", rule_for_sale),
    Step("price_present", "filter", "Price missing, unparseable, zero or negative", rule_price_present),
    Step(
        "price_floor",
        "filter",
        "Price below the minimum plausible sale price (likely a rent, deposit or typo)",
        rule_price_floor,
    ),
    Step("area_present", "filter", "Area missing, unparseable, zero or negative", rule_area_present),
    Step("area_bounds", "filter", "Area outside the plausible range in sq ft", rule_area_bounds),
    Step("categories_present", "filter", "City, location or property type missing", rule_categories_present),
    Step("date_present", "filter", "date_added missing or unparseable", rule_date_present),
    Step("rooms_present", "filter", "Bedrooms or baths missing", rule_rooms_present),
    Step("zero_bedrooms", "filter", "Zero bedrooms on a house, flat or portion", rule_zero_bedrooms),
    Step("absurd_rooms", "filter", "Bedroom or bath count above the cap or negative", rule_absurd_rooms),
    Step("rare_city", "filter", "City with too few listings to model or evaluate", rule_rare_city),
    Step("rare_type", "filter", "Property type with too few listings", rule_rare_type),
]

STEPS = FILTER_STEPS + [
    Step(
        "coordinates",
        "transform",
        "Coordinates outside Pakistan or too far from the listing's city centre set to missing (rows kept)",
        transform_fn=transform_coordinates,
    ),
    Step(
        "exact_duplicates",
        "filter",
        "Exact duplicate listings (every cleaned column identical)",
        rule_exact_duplicates,
    ),
    Step(
        "near_duplicates",
        "filter",
        "Near duplicates: same city, location, size, price, bedrooms and type",
        rule_near_duplicates,
    ),
    Step(
        "ppsf_outliers",
        "filter",
        "Price per sq ft outside the per-city IQR fence on log price per sq ft",
        rule_ppsf_outliers,
    ),
]


def run_steps(df: pd.DataFrame, settings: CleaningSettings, steps=None) -> tuple[pd.DataFrame, list[dict]]:
    steps = STEPS if steps is None else steps
    funnel = [
        {
            "key": "raw",
            "kind": "start",
            "description": "Rows in the raw file",
            "removed": 0,
            "modified": 0,
            "remaining": len(df),
        }
    ]
    for step in steps:
        before = len(df)
        modified = 0
        if step.kind == "filter":
            mask = step.filter_fn(df, settings).fillna(False).astype(bool)
            df = df[mask].reset_index(drop=True)
        else:
            df, modified = step.transform_fn(df, settings)
        funnel.append(
            {
                "key": step.key,
                "kind": step.kind,
                "description": step.description,
                "removed": before - len(df),
                "modified": modified,
                "remaining": len(df),
            }
        )
    return df, funnel


def finalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["listing_id"] = np.arange(len(df))
    for column in ("bedrooms", "baths"):
        df[column] = df[column].astype(int)
    df["date_added"] = pd.to_datetime(df["date_added"])
    return df.drop(columns=["purpose"]).reset_index(drop=True)


def funnel_markdown(funnel: list[dict], info: dict, settings: CleaningSettings, meta: dict) -> str:
    lines = [
        "# Data quality report",
        "",
        "Generated by `python -m house_prices.clean`. Do not edit by hand.",
        "",
        f"- Source file: `{meta['source_file']}`",
        f"- SHA-256: `{meta['file_sha256']}`",
        f"- Raw rows: {meta['raw_rows']:,}",
        f"- Cleaned rows: {meta['clean_rows']:,} ({meta['clean_rows'] / max(meta['raw_rows'], 1):.1%})",
    ]
    if meta.get("synthetic"):
        lines.append("- **This report was produced from SYNTHETIC data. It is not a result.**")
    lines += [
        "",
        "## Cleaning funnel",
        "",
        "| Step | Rule | Rows removed | Rows modified | Rows remaining |",
        "|---|---|---:|---:|---:|",
    ]
    for row in funnel:
        lines.append(
            f"| {row['key']} | {row['description']} | {row['removed']:,} | {row['modified']:,} | {row['remaining']:,} |"
        )
    lines += [
        "",
        "## Unit and price handling",
        "",
        f"- Area unit counts before cleaning: {json.dumps(info['unit_counts'])}",
        f"- Sq ft per Marla: **{info['marla_sqft']}** ({info['marla_source']}).",
        f"- 1 Kanal = {config.KANAL_IN_MARLA} Marla, 1 sq yd = {config.SQYD_SQFT:g} sq ft, "
        f"1 sq m = {config.SQM_SQFT} sq ft.",
        f'- Prices stored as text (for example "1.5 Crore" or "85 Lakh") converted to rupees: '
        f"{info['text_prices']:,} rows.",
        "",
        "## Thresholds",
        "",
        "```json",
        json.dumps({k: v for k, v in asdict(settings).items() if k != "city_centres"}, indent=2, default=str),
        "```",
        "",
        "## Notes",
        "",
        "- Outlier fences use only the price and area of listings, computed on the full cleaned file before "
        "any train/test split. The evaluation therefore describes performance on plausible listings, not on "
        "every raw record.",
        "- Near-duplicate removal can also drop genuinely distinct units (for example two identical flats in "
        "one building priced the same).",
    ]
    return "\n".join(lines) + "\n"


def clean(raw: pd.DataFrame, settings: CleaningSettings | None = None):
    settings = settings or CleaningSettings()
    parsed, info = prepare(raw, settings)
    cleaned, funnel = run_steps(parsed, settings)
    return finalise(cleaned), funnel, info, parsed


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path = config.PATHS.raw_csv if not argv else Path(argv[0])
    try:
        raw = load.load_raw(path)
    except load.DataMissingError as error:
        print(error, file=sys.stderr)
        return 1
    print(load.schema_report(raw))
    print()
    print(load.mapping_report(raw))
    canonical = load.apply_schema(raw)
    settings = CleaningSettings()
    cleaned, funnel, info, parsed = clean(canonical, settings)
    for directory in (config.PATHS.interim_dir, config.PATHS.processed_dir, config.PATHS.reports_dir):
        directory.mkdir(parents=True, exist_ok=True)
    parsed.to_parquet(config.PATHS.parsed, index=False)
    cleaned.to_parquet(config.PATHS.listings, index=False)
    meta = {
        "source_file": str(path),
        "file_sha256": load.file_sha256(path),
        "raw_rows": len(raw),
        "clean_rows": len(cleaned),
        "synthetic": load.is_synthetic(canonical),
        "marla_sqft": info["marla_sqft"],
        "marla_source": info["marla_source"],
        "date_min": str(cleaned["date_added"].min().date()),
        "date_max": str(cleaned["date_added"].max().date()),
        "funnel": funnel,
    }
    config.PATHS.cleaning_summary.write_text(json.dumps(meta, indent=2))
    config.PATHS.data_quality_md.write_text(funnel_markdown(funnel, info, settings, meta), encoding="utf-8")
    print()
    print(f"Raw rows {len(raw):,} -> cleaned rows {len(cleaned):,}")
    print(f"Wrote {config.PATHS.listings} and {config.PATHS.data_quality_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
