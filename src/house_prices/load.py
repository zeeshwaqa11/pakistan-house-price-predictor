import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

from house_prices import config


class DataMissingError(FileNotFoundError):
    pass


class SchemaError(ValueError):
    pass


def normalise_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def read_csv_any_encoding(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise SchemaError(f"Could not decode {path} as utf-8 or latin-1")


def load_raw(path: Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else config.PATHS.raw_csv
    if not path.exists():
        raise DataMissingError(f"Dataset not found at {path}\n" + config.DOWNLOAD_HINT.format(path=path))
    return read_csv_any_encoding(path)


def resolve_columns(df: pd.DataFrame, aliases: dict | None = None) -> dict:
    aliases = aliases or config.COLUMN_ALIASES
    lookup = {}
    for actual in df.columns:
        lookup.setdefault(normalise_name(actual), actual)
    mapping = {}
    for canonical, candidates in aliases.items():
        for candidate in candidates:
            key = normalise_name(candidate)
            if key in lookup:
                mapping[canonical] = lookup[key]
                break
    return mapping


def apply_schema(df: pd.DataFrame, aliases: dict | None = None) -> pd.DataFrame:
    mapping = resolve_columns(df, aliases)
    missing = [c for c in config.REQUIRED_COLUMNS if c not in mapping]
    has_area = any(all(c in mapping for c in group) for group in config.AREA_COLUMN_GROUPS)
    if not has_area:
        missing.append("area (either 'area_size' + 'area_unit', or 'area_text')")
    if missing:
        raise SchemaError(
            "These columns could not be found in the file: "
            + ", ".join(missing)
            + "\nColumns present: "
            + ", ".join(map(str, df.columns))
            + "\nEdit COLUMN_ALIASES in src/house_prices/config.py so each canonical name lists the "
            "column that holds it in your file."
        )
    renamed = df[[actual for actual in mapping.values()]].rename(
        columns={actual: canonical for canonical, actual in mapping.items()}
    )
    for canonical in config.COLUMN_ALIASES:
        if canonical not in renamed.columns:
            renamed[canonical] = pd.NA
    return renamed[list(config.COLUMN_ALIASES)]


def is_synthetic(df: pd.DataFrame) -> bool:
    if "agency" not in df.columns:
        return False
    values = df["agency"].dropna().astype(str)
    return len(values) > 0 and bool((values == config.SYNTHETIC_MARKER).all())


def schema_report(df: pd.DataFrame, sample_rows: int = 5) -> str:
    lines = [f"Rows: {len(df):,}", f"Columns: {df.shape[1]}", "", "Schema (dtype, non-null, distinct):"]
    for column in df.columns:
        series = df[column]
        lines.append(
            f"  {str(column):<24} {str(series.dtype):<10} non-null={series.notna().sum():>9,}  "
            f"distinct={series.nunique(dropna=True):>8,}"
        )
    lines.append("")
    lines.append(f"Sample ({sample_rows} random rows):")
    with pd.option_context("display.max_columns", None, "display.width", 200, "display.max_colwidth", 40):
        lines.append(df.sample(min(sample_rows, len(df)), random_state=config.RANDOM_STATE).to_string())
    return "\n".join(lines)


def mapping_report(df: pd.DataFrame) -> str:
    mapping = resolve_columns(df)
    lines = ["Column mapping (canonical <- file column):"]
    for canonical in config.COLUMN_ALIASES:
        lines.append(f"  {canonical:<14} <- {mapping.get(canonical, '(not found)')}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0]) if argv else None
    try:
        df = load_raw(path)
    except DataMissingError as error:
        print(error, file=sys.stderr)
        return 1
    print(schema_report(df))
    print()
    print(mapping_report(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
