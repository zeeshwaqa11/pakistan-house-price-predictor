import numpy as np
import pandas as pd
import pytest

from house_prices import config, units


def test_marla_uses_configurable_factor():
    assert units.to_sqft(5, "Marla", 272.25) == pytest.approx(1361.25)
    assert units.to_sqft(5, "Marla", 225.0) == pytest.approx(1125.0)


def test_kanal_is_twenty_marla():
    assert units.to_sqft(1, "Kanal", 272.25) == pytest.approx(5445.0)
    assert units.to_sqft(2, "Kanal", 225.0) == pytest.approx(9000.0)


def test_square_yard_and_square_feet_and_metres():
    assert units.to_sqft(200, "Sq. Yd.", 272.25) == pytest.approx(1800.0)
    assert units.to_sqft(1200, "Sq. Ft.", 272.25) == pytest.approx(1200.0)
    assert units.to_sqft(100, "sq m", 272.25) == pytest.approx(1076.39)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Marla", "marla"),
        ("marlas", "marla"),
        ("KANAL", "kanal"),
        ("Sq. Yd.", "sqyd"),
        ("square yards", "sqyd"),
        ("Sq. Ft.", "sqft"),
        ("sqft", "sqft"),
        ("Square Feet", "sqft"),
        ("Acre", None),
        (None, None),
        (np.nan, None),
    ],
)
def test_normalise_unit(raw, expected):
    assert units.normalise_unit(raw) == expected


def test_unknown_unit_gives_nan():
    assert np.isnan(units.to_sqft(3, "Acre", 272.25))


def test_round_trip_from_sqft():
    sqft = units.to_sqft(7, "Marla", 250.0)
    assert units.from_sqft(sqft, "Marla", 250.0) == pytest.approx(7)


def test_area_text_parsing_including_compound():
    assert units.area_text_to_sqft("5 Marla", 272.25) == pytest.approx(1361.25)
    assert units.area_text_to_sqft("1 Kanal 5 Marla", 272.25) == pytest.approx(5445 + 5 * 272.25)
    assert units.area_text_to_sqft("1,089 sqft", 272.25) == pytest.approx(1089)
    assert np.isnan(units.area_text_to_sqft("lots of land", 272.25))


def test_area_series_prefers_size_and_unit_then_text():
    size = pd.Series([5.0, np.nan, 0.0])
    unit = pd.Series(["Marla", None, "Marla"])
    text = pd.Series(["5 Marla", "2 Kanal", "0 Marla"])
    out = units.area_series_to_sqft(size, unit, text, 272.25)
    assert out.iloc[0] == pytest.approx(1361.25)
    assert out.iloc[1] == pytest.approx(10890.0)
    assert out.iloc[2] == 0.0


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1.5 Crore", 15_000_000),
        ("85 Lakh", 8_500_000),
        ("85 Lac", 8_500_000),
        ("2.35 crore", 23_500_000),
        ("1 Arab", 1_000_000_000),
        ("PKR 2,500,000", 2_500_000),
        ("Rs. 12 Lakh", 1_200_000),
        ("3500000", 3_500_000),
        (3_500_000, 3_500_000),
        (7.5e6, 7_500_000),
    ],
)
def test_parse_price(text, expected):
    assert units.parse_price(text) == pytest.approx(expected)


@pytest.mark.parametrize("bad", ["call for price", "", None, np.nan, "1.5 zorgs"])
def test_parse_price_unparseable_is_nan(bad):
    assert np.isnan(units.parse_price(bad))


def test_parse_price_series_mixed_and_count():
    series = pd.Series(["1.5 Crore", "85 Lakh", "4000000", None])
    out = units.parse_price_series(series)
    assert out.iloc[0] == 15_000_000
    assert out.iloc[1] == 8_500_000
    assert out.iloc[2] == 4_000_000
    assert np.isnan(out.iloc[3])
    assert units.count_text_prices(series) == 2
    assert units.count_text_prices(pd.Series([1.0, 2.0])) == 0


def _mixed_frame(marla_sqft, groups=12, per_group=6):
    rows = []
    rng = np.random.default_rng(0)
    for g in range(groups):
        ppsf = 10_000 * (1 + g * 0.05)
        for _ in range(per_group):
            marla = 5.0
            price_m = ppsf * marla * marla_sqft * rng.uniform(0.99, 1.01)
            rows.append(("Lahore", f"L{g}", "House", "marla", marla, price_m))
            sqft = 1300.0
            rows.append(("Lahore", f"L{g}", "House", "sqft", sqft, ppsf * sqft * rng.uniform(0.99, 1.01)))
    cols = ["city", "location", "property_type", "area_unit_norm", "area_value", "price"]
    return pd.DataFrame(rows, columns=cols)


def test_infer_marla_factor_recovers_planted_value():
    factor, groups = units.infer_marla_factor(_mixed_frame(250.0))
    assert factor == pytest.approx(250.0, rel=0.03)
    assert groups == 12


def test_resolve_auto_uses_inferred_when_plausible():
    factor, source = units.resolve_marla_factor(_mixed_frame(225.0), "auto")
    assert factor == pytest.approx(225.0, rel=0.03)
    assert "inferred" in source


def test_resolve_auto_falls_back_without_mixed_units():
    df = _mixed_frame(250.0)
    df = df[df["area_unit_norm"] == "marla"]
    factor, source = units.resolve_marla_factor(df, "auto")
    assert factor == config.MARLA_SQFT_FALLBACK
    assert "default" in source


def test_resolve_auto_rejects_implausible_inference():
    factor, source = units.resolve_marla_factor(_mixed_frame(900.0), "auto")
    assert factor == config.MARLA_SQFT_FALLBACK
    assert "outside the plausible range" in source


def test_resolve_explicit_setting():
    assert units.resolve_marla_factor(None, 225.0)[0] == 225.0
    assert units.resolve_marla_factor(None, "225")[0] == 225.0


def test_haversine_known_distance():
    d = units.haversine_km(31.5497, 74.3436, 33.6844, 73.0479)
    assert 265 < float(d) < 285
