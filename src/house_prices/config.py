import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[2]

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
DEPLOY_SETUP = "random"
INTERVAL_LEVEL = 0.8
MIN_LOCATION_COUNT = 20
RARE_LOCATION_COUNT = 50
TARGET_ENCODER_SMOOTH = "auto"
TARGET_ENCODER_CV = 5
SHAP_SAMPLE_SIZE = 2000
COMPARABLES = 5
INPUT_LIMITS = {"area_sqft": (100.0, 1_000_000.0), "bedrooms": (1, 15), "baths": (1, 15)}
DISCLAIMER = (
    "Estimates are based on historical asking prices from property listings, not sale prices, and are for "
    "learning purposes only. They are not financial or property advice."
)
MIN_GROUP_FOR_ERROR_TABLE = 30

DATASET_NAME = "Pakistan House Price Dataset"
DATASET_SOURCE = "Kaggle, listings scraped from Zameen.com"
RAW_FILENAME = "pakistan_property.csv"
DOWNLOAD_HINT = (
    "Download a Pakistan property-listings dataset from Kaggle (search for 'Pakistan House Price "
    "Dataset', listings scraped from Zameen.com), unzip it, and save the CSV as:\n"
    "    {path}\n"
    "Kaggle needs a free login, so the file cannot be fetched automatically."
)


def project_root() -> Path:
    return Path(os.environ.get("HOUSE_PRICES_ROOT", DEFAULT_ROOT))


def fast_mode() -> bool:
    return os.environ.get("HOUSE_PRICES_FAST", "0") == "1"


class _Paths:
    @property
    def root(self) -> Path:
        return project_root()

    @property
    def raw_csv(self) -> Path:
        return self.root / "data" / "raw" / RAW_FILENAME

    @property
    def sample_csv(self) -> Path:
        return self.root / "data" / "sample" / "synthetic_pakistan_property.csv"

    @property
    def interim_dir(self) -> Path:
        return self.root / "data" / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def parsed(self) -> Path:
        return self.interim_dir / "parsed.parquet"

    @property
    def cleaning_summary(self) -> Path:
        return self.interim_dir / "cleaning_summary.json"

    @property
    def listings(self) -> Path:
        return self.processed_dir / "listings.parquet"

    @property
    def splits(self) -> Path:
        return self.processed_dir / "splits.parquet"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def candidates_dir(self) -> Path:
        return self.models_dir / "candidates"

    @property
    def model_file(self) -> Path:
        return self.models_dir / "house_price_model.joblib"

    @property
    def lower_file(self) -> Path:
        return self.models_dir / "interval_lower.joblib"

    @property
    def upper_file(self) -> Path:
        return self.models_dir / "interval_upper.joblib"

    @property
    def metadata_file(self) -> Path:
        return self.models_dir / "model_metadata.json"

    @property
    def location_index(self) -> Path:
        return self.models_dir / "location_index.csv"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    @property
    def metrics_json(self) -> Path:
        return self.reports_dir / "metrics.json"

    @property
    def data_quality_md(self) -> Path:
        return self.reports_dir / "data_quality.md"

    @property
    def model_card_md(self) -> Path:
        return self.reports_dir / "model_card.md"

    @property
    def readme(self) -> Path:
        return self.root / "README.md"


PATHS = _Paths()

COLUMN_ALIASES = {
    "listing_id": ["property_id", "id", "listing_id"],
    "property_type": ["property_type", "type", "property type"],
    "price": ["price", "price_pkr"],
    "location": ["location", "locality", "area_name"],
    "city": ["city"],
    "province": ["province_name", "province"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon", "lng", "long"],
    "baths": ["baths", "bathrooms", "bath"],
    "bedrooms": ["bedrooms", "beds", "bedroom"],
    "area_text": ["area"],
    "area_size": ["area size", "area_size"],
    "area_unit": ["area type", "area_type", "area_unit"],
    "purpose": ["purpose"],
    "date_added": ["date_added", "date"],
    "agency": ["agency"],
}

REQUIRED_COLUMNS = [
    "property_type",
    "price",
    "location",
    "city",
    "latitude",
    "longitude",
    "baths",
    "bedrooms",
    "purpose",
    "date_added",
]

AREA_COLUMN_GROUPS = [["area_size", "area_unit"], ["area_text"]]

SYNTHETIC_MARKER = "SYNTHETIC DATA"

MARLA_SQFT_FALLBACK = 272.25
MARLA_SQFT_SETTING = "auto"
MARLA_PLAUSIBLE_RANGE = (200.0, 300.0)
KANAL_IN_MARLA = 20
SQYD_SQFT = 9.0
SQM_SQFT = 10.7639

PURPOSE_SALE = "for sale"

CITY_CENTRES = {
    "Karachi": (24.8607, 67.0011),
    "Lahore": (31.5497, 74.3436),
    "Islamabad": (33.6844, 73.0479),
    "Rawalpindi": (33.5651, 73.0169),
    "Faisalabad": (31.4504, 73.1350),
    "Multan": (30.1575, 71.5249),
    "Peshawar": (34.0151, 71.5249),
    "Quetta": (30.1798, 66.9750),
    "Hyderabad": (25.3960, 68.3578),
}

PAKISTAN_BOUNDS = {"lat": (23.5, 37.2), "lon": (60.8, 77.9)}

BEDROOM_REQUIRED_TYPES = (
    "house",
    "flat",
    "penthouse",
    "upper portion",
    "lower portion",
    "farm house",
)

LOCATION_VARIANTS = {
    r"\bdefense\b": "defence",
    r"\bdefence housing authority\b": "dha",
    r"\bbahria\s*town\b": "bahria town",
    r"\btownship\b": "township",
    r"\bphase\s*(\d+)\b": r"phase \1",
    r"\bscheme\s*(\d+)\b": r"scheme \1",
    r"\bg\s*t\s*road\b": "gt road",
    r"\bsociety\b": "society",
}

LOCATION_ACRONYMS = (
    "DHA",
    "PECHS",
    "PWD",
    "WAPDA",
    "CBR",
    "ECHS",
    "NFC",
    "EME",
    "AWT",
    "GT",
    "II",
    "III",
    "IV",
    "VI",
    "OPF",
    "PIA",
    "AIT",
    "FB",
    "KDA",
    "LDA",
    "CDA",
    "MDA",
)


@dataclass
class CleaningSettings:
    min_price_pkr: float = 100_000.0
    min_area_sqft: float = 50.0
    max_area_sqft: float = 2_000_000.0
    max_bedrooms: int = 15
    max_baths: int = 15
    min_city_listings: int = 50
    min_type_listings: int = 30
    max_centre_km: float = 100.0
    near_dup_coordinate_decimals: int | None = 4
    ppsf_iqr_k: float = 3.0
    ppsf_min_group: int = 100
    marla_sqft: str | float = MARLA_SQFT_SETTING
    marla_min_groups: int = 10
    marla_min_group_size: int = 5
    bedroom_required_types: tuple = BEDROOM_REQUIRED_TYPES
    city_centres: dict = field(default_factory=lambda: dict(CITY_CENTRES))
