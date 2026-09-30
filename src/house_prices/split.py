import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from house_prices import config

SETUPS = ("random", "time")


def random_split(
    df: pd.DataFrame, test_size: float = config.TEST_SIZE, random_state: int = config.RANDOM_STATE
) -> tuple[np.ndarray, np.ndarray]:
    train_idx, test_idx = train_test_split(
        df.index.to_numpy(),
        test_size=test_size,
        random_state=random_state,
        stratify=df["city"],
    )
    return np.sort(train_idx), np.sort(test_idx)


def time_split(df: pd.DataFrame, test_size: float = config.TEST_SIZE) -> tuple[np.ndarray, np.ndarray]:
    dates = df["date_added"].sort_values()
    cutoff = dates.iloc[int(len(dates) * (1 - test_size))]
    test_mask = df["date_added"] >= cutoff
    return df.index[~test_mask].to_numpy(), df.index[test_mask].to_numpy()


def make_splits(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    splits = pd.DataFrame({"listing_id": df["listing_id"].to_numpy()})
    for name, fn in (("random", random_split), ("time", time_split)):
        train_idx, test_idx = fn(df)
        column = np.full(len(df), "train", dtype=object)
        column[test_idx] = "test"
        splits[f"{name}_split"] = column
    return splits


def apply_split(df: pd.DataFrame, splits: pd.DataFrame, setup: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if setup not in SETUPS:
        raise ValueError(f"setup must be one of {SETUPS}")
    merged = df.merge(splits[["listing_id", f"{setup}_split"]], on="listing_id", how="inner")
    train = merged[merged[f"{setup}_split"] == "train"].drop(columns=[f"{setup}_split"])
    test = merged[merged[f"{setup}_split"] == "test"].drop(columns=[f"{setup}_split"])
    return train.reset_index(drop=True), test.reset_index(drop=True)


def load_split(setup: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(config.PATHS.listings)
    splits = pd.read_parquet(config.PATHS.splits)
    return apply_split(df, splits, setup)


def describe_splits(df: pd.DataFrame, splits: pd.DataFrame) -> str:
    lines = []
    for setup in SETUPS:
        train, test = apply_split(df, splits, setup)
        lines.append(
            f"{setup:>6}: train {len(train):,} ({train['date_added'].min().date()} to "
            f"{train['date_added'].max().date()}), test {len(test):,} "
            f"({test['date_added'].min().date()} to {test['date_added'].max().date()})"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    if not config.PATHS.listings.exists():
        print("Cleaned listings not found. Run: python -m house_prices.clean", file=sys.stderr)
        return 1
    df = pd.read_parquet(config.PATHS.listings)
    splits = make_splits(df)
    config.PATHS.processed_dir.mkdir(parents=True, exist_ok=True)
    splits.to_parquet(config.PATHS.splits, index=False)
    print(describe_splits(df, splits))
    print(f"Wrote {config.PATHS.splits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
