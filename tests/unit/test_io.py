"""Unit tests for chrono_ltv.utils.io."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from chrono_ltv.utils.io import load_parquet, save_parquet


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


class TestSaveParquet:
    def test_creates_file(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        dest = tmp_path / "out.parquet"
        save_parquet(sample_df, dest)
        assert dest.exists()

    def test_creates_parent_dirs(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        dest = tmp_path / "nested" / "deep" / "out.parquet"
        save_parquet(sample_df, dest)
        assert dest.exists()

    def test_overwrite_true_replaces_file(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        dest = tmp_path / "out.parquet"
        save_parquet(sample_df, dest)
        save_parquet(sample_df, dest, overwrite=True)  # should not raise
        assert dest.exists()

    def test_overwrite_false_raises_on_existing(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        dest = tmp_path / "out.parquet"
        save_parquet(sample_df, dest)
        with pytest.raises(FileExistsError):
            save_parquet(sample_df, dest, overwrite=False)


class TestLoadParquet:
    def test_roundtrip(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        dest = tmp_path / "out.parquet"
        save_parquet(sample_df, dest)
        loaded = load_parquet(dest)
        pd.testing.assert_frame_equal(loaded, sample_df)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_parquet(tmp_path / "nonexistent.parquet")
