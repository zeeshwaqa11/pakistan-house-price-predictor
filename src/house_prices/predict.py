import json
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from house_prices import config, explain, features, intervals, units

TRAIN_COMMANDS = (
    "python -m house_prices.clean\n"
    "python -m house_prices.split\n"
    "python -m house_prices.train\n"
    "python -m house_prices.evaluate\n"
    "python -m house_prices.intervals\n"
    "python -m house_prices.explain"
)

AREA_UNITS = ("Marla", "Kanal", "Sq. Ft.")


class ModelNotFoundError(FileNotFoundError):
    pass


@dataclass
class ModelBundle:
    pipeline: object
    lower: object
    upper: object
    metadata: dict
    location_index: pd.DataFrame
    _explainer: object = field(default=None, repr=False)

    @property
    def marla_sqft(self) -> float:
        return float(self.metadata.get("marla_sqft") or config.MARLA_SQFT_FALLBACK)

    @property
    def explainer(self):
        if self._explainer is None:
            self._explainer = explain.make_explainer(self.pipeline)
        return self._explainer

    @property
    def cities(self) -> list[str]:
        return sorted(self.location_index["city"].unique())

    @property
    def property_types(self) -> list[str]:
        return list(self.metadata.get("categories", {}).get("property_type", ["House", "Flat"]))

    def locations_for(self, city: str) -> list[str]:
        sub = self.location_index[self.location_index["city"] == city]
        return sub.sort_values("n_train", ascending=False)["location"].tolist()


@dataclass
class Prediction:
    estimate: float
    lower: float
    upper: float
    area_sqft: float
    ppsf: float
    location_listings: int
    is_rare: bool
    grouped_as_other: bool
    inputs: pd.DataFrame


def required_files() -> list:
    paths = config.PATHS
    return [paths.model_file, paths.lower_file, paths.upper_file, paths.metadata_file, paths.location_index]


def model_files_exist() -> bool:
    return all(path.exists() for path in required_files())


def load_bundle() -> ModelBundle:
    missing = [p for p in required_files() if not p.exists()]
    if missing:
        raise ModelNotFoundError(
            "The trained model files were not found ("
            + ", ".join(p.name for p in missing)
            + "). Run these commands from the project folder, then reload:\n"
            + TRAIN_COMMANDS
        )
    return ModelBundle(
        pipeline=joblib.load(config.PATHS.model_file),
        lower=joblib.load(config.PATHS.lower_file),
        upper=joblib.load(config.PATHS.upper_file),
        metadata=json.loads(config.PATHS.metadata_file.read_text()),
        location_index=pd.read_csv(config.PATHS.location_index),
    )


def area_to_sqft(bundle: ModelBundle, value: float, unit: str) -> float:
    return units.to_sqft(value, unit, bundle.marla_sqft)


def validate_inputs(
    bundle: ModelBundle, city, location, property_type, area_value, area_unit, bedrooms, baths
) -> list[str]:
    problems = []
    if city not in bundle.cities:
        problems.append(f"City '{city}' is not one the model was trained on.")
    elif location not in bundle.locations_for(city):
        problems.append(f"Location '{location}' was not found in {city}.")
    if property_type not in bundle.property_types:
        problems.append(f"Property type '{property_type}' is not one the model was trained on.")
    if units.normalise_unit(area_unit) is None:
        problems.append(f"Unknown area unit '{area_unit}'.")
    else:
        try:
            sqft = area_to_sqft(bundle, float(area_value), area_unit)
        except (TypeError, ValueError):
            sqft = float("nan")
        low, high = config.INPUT_LIMITS["area_sqft"]
        if not np.isfinite(sqft) or not low <= sqft <= high:
            problems.append(f"Size must be between {low:,.0f} and {high:,.0f} sq ft (got {sqft:,.0f}).")
    for name, value in (("bedrooms", bedrooms), ("baths", baths)):
        low, high = config.INPUT_LIMITS[name]
        if value is None or not float(value).is_integer() or not low <= value <= high:
            problems.append(f"Number of {name} must be a whole number between {low} and {high}.")
    return problems


def build_input(bundle: ModelBundle, city, location, property_type, area_value, area_unit, bedrooms, baths):
    row = bundle.location_index[
        (bundle.location_index["city"] == city) & (bundle.location_index["location"] == location)
    ]
    lat = float(row["median_lat"].iloc[0]) if len(row) else np.nan
    lon = float(row["median_lon"].iloc[0]) if len(row) else np.nan
    if np.isnan(lat) or np.isnan(lon):
        lat, lon = config.CITY_CENTRES.get(city, (np.nan, np.nan))
    return pd.DataFrame(
        {
            "area_sqft": [area_to_sqft(bundle, float(area_value), area_unit)],
            "bedrooms": [int(bedrooms)],
            "baths": [int(baths)],
            "property_type": [property_type],
            "city": [city],
            "location": [location],
            "latitude": [lat],
            "longitude": [lon],
        }
    )[features.INPUT_COLUMNS]


def predict(bundle: ModelBundle, city, location, property_type, area_value, area_unit, bedrooms, baths) -> Prediction:
    problems = validate_inputs(bundle, city, location, property_type, area_value, area_unit, bedrooms, baths)
    if problems:
        raise ValueError(" ".join(problems))
    X = build_input(bundle, city, location, property_type, area_value, area_unit, bedrooms, baths)
    estimate, lower, upper = intervals.predict_interval(bundle.pipeline, bundle.lower, bundle.upper, X)
    row = bundle.location_index[
        (bundle.location_index["city"] == city) & (bundle.location_index["location"] == location)
    ]
    n_train = int(row["n_train"].iloc[0]) if len(row) else 0
    area_sqft = float(X["area_sqft"].iloc[0])
    return Prediction(
        estimate=float(estimate[0]),
        lower=float(lower[0]),
        upper=float(upper[0]),
        area_sqft=area_sqft,
        ppsf=float(estimate[0]) / area_sqft,
        location_listings=n_train,
        is_rare=n_train < config.RARE_LOCATION_COUNT,
        grouped_as_other=n_train < config.MIN_LOCATION_COUNT,
        inputs=X,
    )


def explain_inputs(bundle: ModelBundle, prediction: Prediction, area_label: str, draw: bool = True) -> dict:
    row = prediction.inputs.iloc[0]
    labels = {
        "Location": str(row["location"]),
        "City": str(row["city"]),
        "Property type": str(row["property_type"]),
        "Size": area_label,
        "Bedrooms": str(int(row["bedrooms"])),
        "Bathrooms": str(int(row["baths"])),
    }
    return explain.explain_prediction(bundle.pipeline, bundle.explainer, prediction.inputs, labels=labels, draw=draw)


def find_comparables(listings: pd.DataFrame, prediction: Prediction, n: int = config.COMPARABLES) -> pd.DataFrame:
    row = prediction.inputs.iloc[0]
    pool = listings[listings["city"] == row["city"]].copy()
    if pool.empty:
        return pool
    pool["size_gap"] = np.abs(np.log(pool["area_sqft"] / row["area_sqft"]))
    pool["type_gap"] = (pool["property_type"] != row["property_type"]).astype(float)
    same_location = pool["location"] == row["location"]
    if np.isfinite(row["latitude"]) and np.isfinite(row["longitude"]):
        distance = units.haversine_km(pool["latitude"], pool["longitude"], row["latitude"], row["longitude"])
        pool["distance_km"] = np.where(np.isnan(distance), 25.0, distance)
    else:
        pool["distance_km"] = 25.0
    pool.loc[same_location, "distance_km"] = 0.0
    pool["score"] = pool["size_gap"] + 0.6 * pool["type_gap"] + 0.08 * pool["distance_km"]
    pool["same_location"] = same_location
    best = pool.sort_values("score").head(n).copy()
    best["price_per_sqft"] = best["price"] / best["area_sqft"]
    return best[
        [
            "location",
            "property_type",
            "area_sqft",
            "bedrooms",
            "baths",
            "price",
            "price_per_sqft",
            "distance_km",
            "same_location",
            "date_added",
        ]
    ].reset_index(drop=True)


def preset_examples(bundle: ModelBundle) -> list[dict]:
    presets = []
    plans = [
        ("Family house", "House", "Marla", 10, 5, 5, 0),
        ("Compact flat", "Flat", "Sq. Ft.", 1000, 2, 2, 1),
        ("Large house", "House", "Kanal", 1, 6, 6, 2),
    ]
    ranked = bundle.location_index.groupby("city")["n_train"].sum().sort_values(ascending=False).index.tolist()
    for title, ptype, unit, size, beds, baths, rank in plans:
        if ptype not in bundle.property_types or not ranked:
            continue
        city = ranked[min(rank, len(ranked) - 1)]
        locations = bundle.locations_for(city)
        if not locations:
            continue
        presets.append(
            {
                "title": f"{title} in {city}",
                "city": city,
                "location": locations[0],
                "property_type": ptype,
                "unit": unit,
                "size": size,
                "bedrooms": beds,
                "baths": baths,
            }
        )
    return presets
