"""Unit tests for the data cleaning step in src/train.py."""
import numpy as np
import pandas as pd

from src.train import clean_dataframe, split_features_target


def make_raw():
    return pd.DataFrame({
        "age": [63, 37, "?", 56],
        "chol": [233, "NA", 204, 236],
        "ca": [0, 1, 2, np.nan],
        "num": [0, 2, 4, np.nan],
    })


def test_missing_markers_become_nan_and_columns_are_numeric():
    df = clean_dataframe(make_raw())
    assert df["age"].isna().sum() == 1
    assert df["chol"].isna().sum() == 1
    assert all(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)


def test_rows_without_target_are_dropped():
    df = clean_dataframe(make_raw())
    assert len(df) == 3


def test_target_is_mapped_to_binary():
    df = clean_dataframe(make_raw())
    assert df["num"].tolist() == [0, 1, 1]
    assert set(df["num"].unique()) <= {0, 1}


def test_unnamed_target_column_is_renamed():
    raw = make_raw().rename(columns={"num": "target"})
    df = clean_dataframe(raw)
    assert "num" in df.columns and "target" not in df.columns


def test_split_drops_columns_that_are_entirely_missing():
    df = clean_dataframe(make_raw())
    df["empty"] = np.nan
    X, y, features = split_features_target(df)
    assert "empty" not in features
    assert "num" not in X.columns
    assert len(X) == len(y)
