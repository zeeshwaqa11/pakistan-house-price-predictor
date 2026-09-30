import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from house_prices import config

CITY_SPECS = {
    "Karachi": ("Sindh", 24.8607, 67.0011, 9500, 0.30, 14),
    "Lahore": ("Punjab", 31.5497, 74.3436, 11500, 0.32, 14),
    "Islamabad": ("Islamabad Capital", 33.6844, 73.0479, 14500, 0.33, 12),
    "Rawalpindi": ("Punjab", 33.5651, 73.0169, 9000, 0.28, 10),
    "Faisalabad": ("Punjab", 31.4504, 73.1350, 6500, 0.25, 8),
}

TYPE_SPECS = {
    "House": (0.62, 1.0),
    "Flat": (0.24, 1.12),
    "Upper Portion": (0.05, 0.82),
    "Lower Portion": (0.05, 0.85),
    "Penthouse": (0.02, 1.3),
    "Farm House": (0.015, 0.45),
    "Room": (0.005, 0.9),
}

AREA_CATEGORIES = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 100000)]


def _area_category(marla: float) -> str:
    for low, high in AREA_CATEGORIES:
        if low <= marla < high:
            return f"{low}-{high} Marla" if high < 100000 else f"{low}+ Marla"
    return "20+ Marla"


def _location_table(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for city, (province, lat, lon, base, spread, n_loc) in CITY_SPECS.items():
        for k in range(n_loc):
            rows.append(
                {
                    "city": city,
                    "province": province,
                    "location": f"Sample Area {k + 1:02d}",
                    "lat": lat + rng.normal(0, 0.05),
                    "lon": lon + rng.normal(0, 0.05),
                    "factor": float(np.exp(rng.normal(0, spread))),
                    "base": base,
                    "weight": float(rng.pareto(1.2) + 0.4),
                }
            )
    table = pd.DataFrame(rows)
    table["weight"] = table["weight"] / table.groupby("city")["weight"].transform("sum")
    return table


def make_sample(n_rows: int = 6000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    locations = _location_table(rng)
    city_names = list(CITY_SPECS)
    city_weights = np.array([0.30, 0.30, 0.18, 0.12, 0.10])
    cities = rng.choice(city_names, size=n_rows, p=city_weights)
    type_names = list(TYPE_SPECS)
    type_probs = np.array([v[0] for v in TYPE_SPECS.values()])
    type_probs = type_probs / type_probs.sum()
    types = rng.choice(type_names, size=n_rows, p=type_probs)

    records = []
    start = pd.Timestamp("2018-07-01")
    for i in range(n_rows):
        city = cities[i]
        sub = locations[locations["city"] == city]
        loc = sub.iloc[rng.choice(len(sub), p=sub["weight"].to_numpy())]
        ptype = types[i]
        if ptype == "Flat":
            sqft = float(rng.choice([450, 600, 750, 900, 1100, 1350, 1600, 2000]) * rng.uniform(0.9, 1.1))
            unit, value = "Sq. Ft.", round(sqft)
        elif ptype == "Farm House":
            unit, value = "Kanal", int(rng.choice([4, 8, 10, 20]))
            sqft = value * 20 * config.MARLA_SQFT_FALLBACK
        elif ptype == "Room":
            unit, value = "Sq. Ft.", int(rng.choice([120, 150, 200]))
            sqft = float(value)
        else:
            if rng.random() < 0.18:
                unit, value = "Kanal", int(rng.choice([1, 1, 2, 3]))
                sqft = value * 20 * config.MARLA_SQFT_FALLBACK
            elif rng.random() < 0.12:
                unit, value = "Sq. Yd.", int(rng.choice([120, 200, 240, 400, 500]))
                sqft = value * config.SQYD_SQFT
            else:
                unit, value = "Marla", float(rng.choice([3, 4, 5, 6, 7, 8, 10, 12]))
                sqft = value * config.MARLA_SQFT_FALLBACK
        ppsf = (
            loc["base"]
            * loc["factor"]
            * TYPE_SPECS[ptype][1]
            * (max(sqft, 100.0) / 1500.0) ** -0.12
            * float(np.exp(rng.normal(0, 0.18)))
        )
        price = round(ppsf * sqft, -4)
        beds = int(np.clip(round(sqft / 480 + rng.normal(0, 0.8)), 1, 9))
        if ptype == "Room":
            beds = 1
        baths = int(np.clip(beds + rng.integers(-1, 2), 1, 9))
        marla_equiv = sqft / config.MARLA_SQFT_FALLBACK
        added = start + pd.Timedelta(days=int(rng.integers(0, 365)))
        records.append(
            {
                "property_id": 1_000_000 + i,
                "location_id": 5000 + int(locations.index[(locations["city"] == city)][0]),
                "page_url": f"synthetic://listing/{1_000_000 + i}",
                "property_type": ptype,
                "price": price,
                "location": loc["location"],
                "city": city,
                "province_name": loc["province"],
                "latitude": round(loc["lat"] + rng.normal(0, 0.008), 6),
                "longitude": round(loc["lon"] + rng.normal(0, 0.008), 6),
                "baths": baths,
                "area": f"{value} {unit}",
                "purpose": "For Sale",
                "bedrooms": beds,
                "date_added": f"{added.month}/{added.day}/{added.year}",
                "agency": config.SYNTHETIC_MARKER,
                "agent": config.SYNTHETIC_MARKER,
                "Area Type": unit,
                "Area Size": value,
                "Area Category": _area_category(marla_equiv),
            }
        )
    df = pd.DataFrame(records)
    return _inject_problems(df, rng)


def _inject_problems(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    df = df.copy()
    rent = df.sample(frac=0.18, random_state=1).copy()
    rent["purpose"] = "For Rent"
    rent["price"] = (rent["price"] * 0.004).round(-3)
    rent["property_id"] = rent["property_id"] + 5_000_000

    df["price"] = df["price"].astype(object)
    text_idx = rng.choice(n, size=max(1, n // 40), replace=False)
    for i in text_idx:
        p = float(df.at[i, "price"])
        df.at[i, "price"] = f"{p / 1e7:.2f} Crore" if p >= 1e7 else f"{p / 1e5:.0f} Lakh"

    def pick(frac):
        return rng.choice(n, size=max(1, int(n * frac)), replace=False)

    df.loc[pick(0.008), "price"] = 0
    df.loc[pick(0.005), "price"] = np.nan
    df.loc[pick(0.005), "Area Size"] = 0
    df.loc[pick(0.008), "bedrooms"] = 0
    df.loc[pick(0.005), "bedrooms"] = 60
    df.loc[pick(0.006), "baths"] = np.nan
    df.loc[pick(0.008), ["latitude", "longitude"]] = 0.0
    df.loc[pick(0.004), ["latitude", "longitude"]] = [45.0, 10.0]
    numeric_idx = [i for i in pick(0.006) if not isinstance(df.at[i, "price"], str)]
    for i in numeric_idx:
        df.at[i, "price"] = float(df.at[i, "price"]) * float(rng.choice([0.01, 100.0]))
    messy = pick(0.03)
    df.loc[messy, "location"] = df.loc[messy, "location"].str.upper() + " "
    df.loc[pick(0.02), "location"] = df["location"].str.lower()
    df.loc[pick(0.003), "property_type"] = "Room"
    out = pd.concat([df, rent], ignore_index=True)
    duplicates = out.sample(frac=0.03, random_state=2)
    out = pd.concat([out, duplicates], ignore_index=True)
    return out.sample(frac=1.0, random_state=3).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a SYNTHETIC property file for tests and CI.")
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    out = args.out or config.PATHS.sample_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    df = make_sample(args.rows, args.seed)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} SYNTHETIC rows to {out}")
    print("This file is fake. It exists for tests and CI only and must never be used for reported results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
