import math

LAKH = 100_000
CRORE = 10_000_000


def _trim(number: float, places: int = 2) -> str:
    text = f"{number:,.{places}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _split_units(value: float) -> tuple[float, str]:
    value = round(value)
    if value < LAKH:
        return float(value), "rupees"
    if value < CRORE:
        lakh = round(value / LAKH, 2)
        if lakh < 100:
            return lakh, "Lakh"
    return round(value / CRORE, 2), "Crore"


def format_pkr(value: float) -> str:
    if _is_missing(value):
        return "n/a"
    if value < 0:
        return "-" + format_pkr(-value)
    amount, unit = _split_units(value)
    if unit == "rupees":
        return f"PKR {amount:,.0f}"
    return f"PKR {_trim(amount)} {unit}"


def format_exact(value: float) -> str:
    if _is_missing(value):
        return "n/a"
    return f"PKR {round(value):,}"


def format_range(low: float, high: float) -> str:
    if _is_missing(low) or _is_missing(high):
        return "n/a"
    low_amount, low_unit = _split_units(max(low, 0))
    high_amount, high_unit = _split_units(max(high, 0))
    if low_unit == high_unit and low_unit != "rupees":
        return f"PKR {_trim(low_amount)} to {_trim(high_amount)} {low_unit}"
    return f"{format_pkr(low)} to {format_pkr(high)}"


def format_ppsf(value: float) -> str:
    if _is_missing(value):
        return "n/a"
    return f"PKR {round(value):,} per sq ft"


def format_area(sqft: float, marla_sqft: float) -> str:
    if _is_missing(sqft):
        return "n/a"
    marla = sqft / marla_sqft
    if marla >= 20:
        kanal = marla / 20
        return f"{_trim(kanal, 1)} Kanal ({sqft:,.0f} sq ft)"
    if marla >= 1:
        return f"{_trim(marla, 1)} Marla ({sqft:,.0f} sq ft)"
    return f"{sqft:,.0f} sq ft"
