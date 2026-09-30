import numpy as np
import pytest

from house_prices import formatting as f


@pytest.mark.parametrize(
    "value, expected",
    [
        (23_500_000, "PKR 2.35 Crore"),
        (8_500_000, "PKR 85 Lakh"),
        (10_000_000, "PKR 1 Crore"),
        (9_999_999, "PKR 1 Crore"),
        (9_950_000, "PKR 99.5 Lakh"),
        (100_000, "PKR 1 Lakh"),
        (99_999, "PKR 99,999"),
        (99_999.6, "PKR 1 Lakh"),
        (0, "PKR 0"),
        (1_250_000_000, "PKR 125 Crore"),
        (12_345_678_900, "PKR 1,234.57 Crore"),
        (15_000_000, "PKR 1.5 Crore"),
        (12_500_000, "PKR 1.25 Crore"),
        (1_234_567, "PKR 12.35 Lakh"),
    ],
)
def test_format_pkr(value, expected):
    assert f.format_pkr(value) == expected


def test_format_pkr_negative_and_missing():
    assert f.format_pkr(-8_500_000) == "-PKR 85 Lakh"
    assert f.format_pkr(np.nan) == "n/a"
    assert f.format_pkr(None) == "n/a"


def test_format_exact():
    assert f.format_exact(23_500_000) == "PKR 23,500,000"
    assert f.format_exact(23_500_000.4) == "PKR 23,500,000"
    assert f.format_exact(np.nan) == "n/a"


def test_format_range_same_and_mixed_units():
    assert f.format_range(19_000_000, 28_000_000) == "PKR 1.9 to 2.8 Crore"
    assert f.format_range(8_000_000, 9_500_000) == "PKR 80 to 95 Lakh"
    assert f.format_range(9_000_000, 12_000_000) == "PKR 90 Lakh to PKR 1.2 Crore"
    assert f.format_range(np.nan, 1) == "n/a"


def test_format_ppsf_and_area():
    assert f.format_ppsf(12_345.6) == "PKR 12,346 per sq ft"
    assert f.format_area(1361.25, 272.25) == "5 Marla (1,361 sq ft)"
    assert f.format_area(5445, 272.25) == "1 Kanal (5,445 sq ft)"
    assert f.format_area(150, 272.25) == "150 sq ft"
