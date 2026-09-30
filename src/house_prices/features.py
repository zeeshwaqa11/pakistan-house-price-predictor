import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from house_prices import config, units

INPUT_COLUMNS = [
    "area_sqft",
    "bedrooms",
    "baths",
    "property_type",
    "city",
    "location",
    "latitude",
    "longitude",
]
NUMERIC_FEATURES = [
    "area_sqft",
    "log_area_sqft",
    "bedrooms",
    "baths",
    "latitude",
    "longitude",
    "distance_to_centre_km",
]
CATEGORICAL_FEATURES = ["city", "property_type"]
LOCATION_FEATURE = "location_key"
TARGET_COLUMN = "price"
TARGET_DERIVED_NAMES = (
    "price",
    "ppsf",
    "price_per_sqft",
    "price_per_sq_ft",
    "log_price",
    "price_per_marla",
)


def make_target(df: pd.DataFrame) -> pd.Series:
    return np.log(df[TARGET_COLUMN].astype(float))


def make_inputs(df: pd.DataFrame) -> pd.DataFrame:
    return df[INPUT_COLUMNS].copy()


class FeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, city_centres: dict | None = None):
        self.city_centres = city_centres

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        missing = [c for c in INPUT_COLUMNS if c not in X.columns]
        if missing:
            raise KeyError(f"Missing input columns: {missing}")
        frame = X[INPUT_COLUMNS]
        centres = self.city_centres or config.CITY_CENTRES
        area = frame["area_sqft"].astype(float)
        city = frame["city"].astype(str)
        centre_lat = city.map(lambda c: centres.get(c, (np.nan, np.nan))[0]).astype(float)
        centre_lon = city.map(lambda c: centres.get(c, (np.nan, np.nan))[1]).astype(float)
        lat = frame["latitude"].astype(float)
        lon = frame["longitude"].astype(float)
        out = pd.DataFrame(
            {
                "area_sqft": area.to_numpy(),
                "log_area_sqft": np.log(area.clip(lower=1.0)).to_numpy(),
                "bedrooms": frame["bedrooms"].astype(float).to_numpy(),
                "baths": frame["baths"].astype(float).to_numpy(),
                "latitude": lat.to_numpy(),
                "longitude": lon.to_numpy(),
                "distance_to_centre_km": units.haversine_km(lat, lon, centre_lat, centre_lon),
                "city": city.to_numpy(dtype=object),
                "property_type": frame["property_type"].astype(str).to_numpy(dtype=object),
                "location": frame["location"].astype(str).to_numpy(dtype=object),
            },
            index=X.index,
        )
        return out


class LocationGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, min_count: int = config.MIN_LOCATION_COUNT):
        self.min_count = min_count

    @staticmethod
    def make_key(city, location) -> pd.Series:
        return city.astype(str) + " | " + location.astype(str)

    def fit(self, X, y=None):
        keys = self.make_key(X["city"], X["location"])
        counts = keys.value_counts()
        self.frequent_ = set(counts[counts >= self.min_count].index)
        self.counts_ = counts.to_dict()
        return self

    def transform(self, X):
        keys = self.make_key(X["city"], X["location"])
        other = "Other (" + X["city"].astype(str) + ")"
        out = X.copy()
        out[LOCATION_FEATURE] = keys.where(keys.isin(self.frequent_), other).astype(object)
        return out.drop(columns=["location"])


def build_preprocessor(scale: bool = False, location_encoding: str = "target") -> ColumnTransformer:
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    if location_encoding == "target":
        location = TargetEncoder(
            target_type="continuous",
            smooth=config.TARGET_ENCODER_SMOOTH,
            cv=KFold(n_splits=config.TARGET_ENCODER_CV, shuffle=True, random_state=config.RANDOM_STATE),
        )
    elif location_encoding == "onehot":
        location = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        raise ValueError(f"Unknown location encoding: {location_encoding}")
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("loc", location, [LOCATION_FEATURE]),
        ],
        verbose_feature_names_out=True,
    )


def build_pipeline(
    model,
    scale: bool = False,
    location_encoding: str = "target",
    min_location_count: int = config.MIN_LOCATION_COUNT,
) -> Pipeline:
    return Pipeline(
        [
            ("features", FeatureBuilder()),
            ("location", LocationGrouper(min_count=min_location_count)),
            ("prep", build_preprocessor(scale=scale, location_encoding=location_encoding)),
            ("model", model),
        ]
    )
