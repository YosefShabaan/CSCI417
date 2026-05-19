from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def split_features_target(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def train_test_split_data(X, y, test_size: float, random_state: int):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
