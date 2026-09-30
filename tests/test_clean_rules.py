import numpy as np
import pandas as pd
import pytest

from house_prices import clean, config, load
from house_prices.config import CleaningSettings

S = CleaningSettings()


def base_rows(n=3, **overrides):
    rows = pd.DataFrame(
        {
            "listing_id": range(n),
            "property_type": ["House"] * n,
            "price": [10_000_000.0 + i * 1_000_000 for i in range(n)],
            "location": [f"Loc {i}" for i in range(n)],
            "city": ["Lahore"] * n,
            "province": ["Punjab"] * n,
            "latitude": [31.55 + i * 0.01 for i in range(n)],
            "longitude": [74.34 + i * 0.01 for i in range(n)],
            "baths": [3] * n,
            "bedrooms": [4] * n,
            "area_sqft": [2000.0 + i * 100 for i in range(n)],
            "area_value": [7.0] * n,
            "area_unit_norm": ["marla"] * n,
            "purpose": ["for sale"] * n,
            "date_added": pd.to_datetime(["2019-01-01"] * n),
        }
    )
    for column, values in overrides.items():
        rows[column] = values
    return rows


def kept(rule, df, settings=S):
    return rule(df, settings).tolist()


def test_for_sale_only():
    df = base_rows(3, purpose=["for sale", "for rent", None])
    assert kept(clean.rule_for_sale, df) == [True, False, False]


def test_price_present_removes_zero_missing_negative():
    df = base_rows(4, price=[1e7, 0, np.nan, -5])
    assert kept(clean.rule_price_present, df) == [True, False, False, False]


def test_price_floor():
    df = base_rows(2, price=[99_999, 100_000])
    assert kept(clean.rule_price_floor, df) == [False, True]


def test_area_present_removes_zero_and_missing():
    df = base_rows(3, area_sqft=[1000.0, 0.0, np.nan])
    assert kept(clean.rule_area_present, df) == [True, False, False]


def test_area_bounds():
    df = base_rows(3, area_sqft=[10.0, 1500.0, 5_000_000.0])
    assert kept(clean.rule_area_bounds, df) == [False, True, False]


def test_categories_present():
    df = base_rows(3, city=["Lahore", None, "Lahore"], location=["a", "b", None])
    assert kept(clean.rule_categories_present, df) == [True, False, False]


def test_date_present():
    df = base_rows(2, date_added=[pd.Timestamp("2019-02-01"), pd.NaT])
    assert kept(clean.rule_date_present, df) == [True, False]


def test_rooms_present():
    df = base_rows(3, bedrooms=[3, np.nan, 3], baths=[2, 2, np.nan])
    assert kept(clean.rule_rooms_present, df) == [True, False, False]


def test_zero_bedroom_house_removed_but_room_kept():
    df = base_rows(3, property_type=["House", "Room", "Flat"], bedrooms=[0, 0, 2])
    assert kept(clean.rule_zero_bedrooms, df) == [False, True, True]


def test_absurd_room_counts():
    df = base_rows(4, bedrooms=[3, 16, 3, -1], baths=[2, 2, 40, 2])
    assert kept(clean.rule_absurd_rooms, df) == [True, False, False, False]


def test_rare_city_dropped():
    df = base_rows(6, city=["Lahore"] * 5 + ["Tiny"])
    out = kept(clean.rule_rare_city, df, CleaningSettings(min_city_listings=3))
    assert out == [True] * 5 + [False]


def test_rare_type_dropped():
    df = base_rows(6, property_type=["House"] * 5 + ["Castle"])
    out = kept(clean.rule_rare_type, df, CleaningSettings(min_type_listings=3))
    assert out == [True] * 5 + [False]


def test_exact_duplicates_ignore_listing_id():
    df = base_rows(3)
    df.loc[2] = df.loc[0]
    df.loc[2, "listing_id"] = 99
    assert kept(clean.rule_exact_duplicates, df) == [True, True, False]


def test_near_duplicates_same_key_different_coordinates():
    df = base_rows(3, location=["Same", "Same", "Other"])
    df["price"] = 1e7
    df["area_sqft"] = [2000.2, 1999.8, 2000.0]
    assert kept(clean.rule_near_duplicates, df) == [True, False, True]


def test_near_duplicates_different_bedrooms_are_kept():
    df = base_rows(2, location=["Same", "Same"], bedrooms=[3, 4])
    df["price"] = 1e7
    df["area_sqft"] = 2000.0
    assert kept(clean.rule_near_duplicates, df) == [True, True]


def test_ppsf_outlier_rule_is_per_city():
    rng = np.random.default_rng(1)
    n = 200
    lahore_ppsf = rng.normal(10_000, 1_000, n).clip(7000)
    karachi_ppsf = rng.normal(4_000, 400, n).clip(2500)
    df = pd.concat(
        [
            base_rows(n, city="Lahore", area_sqft=1000.0, price=lahore_ppsf * 1000),
            base_rows(n, city="Karachi", area_sqft=1000.0, price=karachi_ppsf * 1000),
        ],
        ignore_index=True,
    )
    df.loc[0, "price"] = 1000 * 90_000
    df.loc[n, "price"] = 1000 * 100
    df.loc[n + 1, "price"] = 1000 * 10_000
    out = clean.rule_ppsf_outliers(df, CleaningSettings(ppsf_min_group=50))
    assert not out.iloc[0]
    assert not out.iloc[n]
    assert not out.iloc[n + 1]
    assert out.iloc[1]
    assert out.iloc[n + 5]


def test_ppsf_rule_uses_global_fence_for_thin_cities():
    df = base_rows(60, city=["Lahore"] * 58 + ["Thin"] * 2, area_sqft=1000.0)
    df["price"] = 1000 * np.linspace(9000, 11000, 60)
    bounds = clean.ppsf_bounds(df, CleaningSettings(ppsf_min_group=100))
    assert bounds.loc["Thin", "low"] == bounds.loc["Lahore", "low"]


def test_coordinates_outside_pakistan_or_far_from_city_are_blanked():
    df = base_rows(4, latitude=[31.55, 0.0, 45.0, 24.86], longitude=[74.34, 0.0, 10.0, 67.0])
    out, modified = clean.transform_coordinates(df, S)
    assert out["latitude"].isna().tolist() == [False, True, True, True]
    assert modified == 3
    assert len(out) == len(df)


def test_coordinates_already_missing_not_counted():
    df = base_rows(2, latitude=[np.nan, 31.55], longitude=[np.nan, 74.34])
    out, modified = clean.transform_coordinates(df, S)
    assert modified == 0


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("  DHA  Defense Phase 5 ", "DHA Defence Phase 5"),
        ("gulshan-e-iqbal", "Gulshan E Iqbal"),
        ("Gulshan e Iqbal", "Gulshan E Iqbal"),
        ("bahria town", "Bahria Town"),
        ("Bahria  Town", "Bahria Town"),
        ("F-7", "F-7"),
        ("g-11/3", "G-11/3"),
        ("pechs", "PECHS"),
        ("  ", None),
        (None, None),
    ],
)
def test_standardise_location(raw, expected):
    assert clean.standardise_location(raw) == expected


def test_location_spelling_variants_merge():
    assert clean.standardise_location("DHA Defense") == clean.standardise_location("dha defence")


def test_run_steps_builds_funnel_that_adds_up():
    df = base_rows(6, purpose=["for sale"] * 5 + ["for rent"])
    df.loc[1, "price"] = 0
    out, funnel = clean.run_steps(df, S, steps=clean.FILTER_STEPS[:2])
    assert funnel[0]["remaining"] == 6
    assert funnel[1]["removed"] == 1 and funnel[2]["removed"] == 1
    assert funnel[-1]["remaining"] == len(out) == 4
    assert sum(r["removed"] for r in funnel) == 6 - len(out)


def test_full_clean_on_synthetic(synthetic_raw):
    cleaned, funnel, info, parsed = clean.clean(load.apply_schema(synthetic_raw))
    assert 0.4 * len(synthetic_raw) < len(cleaned) < len(synthetic_raw)
    assert (cleaned["price"] > 0).all()
    assert (cleaned["area_sqft"] > 0).all()
    assert cleaned["city"].isin(config.CITY_CENTRES).all()
    assert info["text_prices"] > 0
    assert funnel[-1]["remaining"] == len(cleaned)
    removed_total = sum(r["removed"] for r in funnel)
    assert len(synthetic_raw) - removed_total == len(cleaned)
    assert cleaned["location"].str.strip().eq(cleaned["location"]).all()
    assert cleaned["location"].nunique() < 60


def test_funnel_markdown_contains_every_step(synthetic_raw):
    cleaned, funnel, info, parsed = clean.clean(load.apply_schema(synthetic_raw))
    meta = {
        "source_file": "x.csv",
        "file_sha256": "0" * 64,
        "raw_rows": len(synthetic_raw),
        "clean_rows": len(cleaned),
        "synthetic": True,
    }
    md = clean.funnel_markdown(funnel, info, S, meta)
    for row in funnel:
        assert row["key"] in md
    assert "SYNTHETIC" in md
