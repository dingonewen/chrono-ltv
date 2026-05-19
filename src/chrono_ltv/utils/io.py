"""Helpers for reading and writing datasets."""

from pathlib import Path

import pandas as pd
from loguru import logger


def save_parquet(df: pd.DataFrame, path: Path, *, overwrite: bool = True) -> None:
    """Persist *df* to *path* as Parquet (snappy-compressed)."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"File already exists and overwrite=False: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="snappy")
    logger.info(f"Saved {len(df):,} rows → {path}")


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a Parquet file and return a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df):,} rows ← {path}")
    return df
