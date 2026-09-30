import re

import numpy as np
import pandas as pd

from house_prices import config

PRICE_MULTIPLIERS = {
    "arab": 1e9,
    "arb": 1e9,
    "billion": 1e9,
    "crore": 1e7,
    "cr": 1e7,
    "karor": 1e7,
    "million": 1e6,
    "mn": 1e6,
    "lakh": 1e5,
    "lakhs": 1e5,
    "lac": 1e5,
    "lacs": 1e5,
    "thousand": 1e3,
    "hazar": 1e3,
    "k": 1e3,
}

_NUMBER = r"(\d[\d,]*(?:\.\d+)?|\.\d+)"
_PRICE_RE = re.compile(rf"^\s*(?:pkr|rs\.?|rupees)?\s*{_NUMBER}\s*([a-z]*)\s*$", re.IGNORECASE)
_AREA_PAIR_RE = re.compile(rf"{_NUMBER}\s*([A-Za-z][A-Za-z\.\s]*?)(?=\s*\d|$)")


def parse_price(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip().lower().replace("₨", "")
    match = _PRICE_RE.match(text)
    if not match:
        return np.nan
    number = float(match.group(1).replace(",", ""))
    word = match.group(2)
    if not word:
        return number
    multiplier = PRICE_MULTIPLIERS.get(word)
    if multiplier is None:
        return np.nan
    return number * multiplier


def parse_price_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype(float)
    return series.map(parse_price).astype(float)


def count_text_prices(series: pd.Series) -> int:
    if pd.api.types.is_numeric_dtype(series):
        return 0
    numeric = pd.to_numeric(series, errors="coerce")
    return int((series.notna() & numeric.isna()).sum())


def normalise_unit(raw) -> str | None:
    if raw is None or raw is pd.NA or (isinstance(raw, float) and np.isnan(raw)):
        return None
    key = re.sub(r"[^a-z]", "", str(raw).lower())
    if not key:
        return None
    if "marla" in key:
        return "marla"
    if "kanal" in key:
        return "kanal"
    if "yd" in key or "yard" in key or key == "gaz":
        return "sqyd"
    if key.endswith("m") and key.startswith("sq") or "meter" in key or "metre" in key:
        return "sqm"
    if "ft" in key or "feet" in key or "foot" in key:
        return "sqft"
    return None


def sqft_per_unit(unit: str | None, marla_sqft: float) -> float:
    factors = {
        "marla": marla_sqft,
        "kanal": marla_sqft * config.KANAL_IN_MARLA,
        "sqft": 1.0,
        "sqyd": config.SQYD_SQFT,
        "sqm": config.SQM_SQFT,
    }
    return factors.get(unit, np.nan)


def to_sqft(value, unit, marla_sqft: float) -> float:
    normalised = normalise_unit(unit)
    if normalised is None or value is None or pd.isna(value):
        return np.nan
    return float(value) * sqft_per_unit(normalised, marla_sqft)


def from_sqft(sqft: float, unit: str, marla_sqft: float) -> float:
    normalised = normalise_unit(unit)
    return float(sqft) / sqft_per_unit(normalised, marla_sqft)


def parse_area_text(text) -> list[tuple[float, str | None]]:
    if text is None or text is pd.NA or (isinstance(text, float) and np.isnan(text)):
        return []
    pairs = []
    for number, unit in _AREA_PAIR_RE.findall(str(text)):
        pairs.append((float(number.replace(",", "")), normalise_unit(unit)))
    return pairs


def area_text_to_sqft(text, marla_sqft: float) -> float:
    pairs = parse_area_text(text)
    if not pairs or any(unit is None for _, unit in pairs):
        return np.nan
    return float(sum(value * sqft_per_unit(unit, marla_sqft) for value, unit in pairs))


def area_series_to_sqft(size: pd.Series, unit: pd.Series, text: pd.Series, marla_sqft: float) -> pd.Series:
    size_num = pd.to_numeric(size, errors="coerce")
    units = unit.map(normalise_unit)
    factor = units.map(lambda u: sqft_per_unit(u, marla_sqft) if u else np.nan).astype(float)
    direct = size_num * factor
    from_text = text.map(lambda t: area_text_to_sqft(t, marla_sqft)).astype(float)
    return direct.where(direct.notna() & size_num.notna(), from_text).astype(float)


def infer_marla_factor(
    df: pd.DataFrame,
    min_groups: int = 10,
    min_group_size: int = 5,
) -> tuple[float | None, int]:
    required = {"price", "area_value", "area_unit_norm", "city", "location", "property_type"}
    if not required.issubset(df.columns):
        return None, 0
    frame = df.dropna(subset=list(required)).copy()
    frame = frame[(frame["price"] > 0) & (frame["area_value"] > 0)]
    marla_like = frame["area_unit_norm"].isin(["marla", "kanal"])
    sqft_like = frame["area_unit_norm"].isin(["sqft", "sqyd", "sqm"])
    marla = frame[marla_like].copy()
    marla["marla_units"] = np.where(
        marla["area_unit_norm"] == "kanal",
        marla["area_value"] * config.KANAL_IN_MARLA,
        marla["area_value"],
    )
    marla["ppu"] = marla["price"] / marla["marla_units"]
    other = frame[sqft_like].copy()
    other["sqft"] = other.apply(lambda r: to_sqft(r["area_value"], r["area_unit_norm"], 1.0), axis=1)
    other["ppu"] = other["price"] / other["sqft"]
    keys = ["city", "location", "property_type"]
    left = marla.groupby(keys)["ppu"].agg(["median", "size"])
    right = other.groupby(keys)["ppu"].agg(["median", "size"])
    joined = left.join(right, lsuffix="_marla", rsuffix="_sqft", how="inner")
    joined = joined[(joined["size_marla"] >= min_group_size) & (joined["size_sqft"] >= min_group_size)]
    if len(joined) < min_groups:
        return None, len(joined)
    ratios = joined["median_marla"] / joined["median_sqft"]
    return float(ratios.median()), len(joined)


def resolve_marla_factor(
    df: pd.DataFrame | None,
    setting: str | float = config.MARLA_SQFT_SETTING,
    min_groups: int = 10,
    min_group_size: int = 5,
) -> tuple[float, str]:
    if not isinstance(setting, str):
        return float(setting), f"configured value {float(setting)} sq ft per Marla"
    if setting != "auto":
        return float(setting), f"configured value {float(setting)} sq ft per Marla"
    if df is not None:
        inferred, groups = infer_marla_factor(df, min_groups, min_group_size)
        low, high = config.MARLA_PLAUSIBLE_RANGE
        if inferred is not None and low <= inferred <= high:
            return inferred, (
                f"inferred from {groups} location/type groups that list both Marla and sq ft/sq yd areas"
            )
        if inferred is not None:
            return config.MARLA_SQFT_FALLBACK, (
                f"data implied {inferred:.1f} sq ft per Marla from {groups} groups, outside the plausible "
                f"range {low}-{high}, so the default {config.MARLA_SQFT_FALLBACK} was used"
            )
    return config.MARLA_SQFT_FALLBACK, (
        f"default {config.MARLA_SQFT_FALLBACK} sq ft per Marla; the data has too few locations listing both "
        "Marla and sq ft/sq yd areas to infer a factor"
    )


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))
