from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from typing import Any

from .config import FEATURE_COLUMNS


@dataclass
class ModelResult:
    model: Pipeline | Any | None
    train_rows: int
    test_rows: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp


def train_model(features: pd.DataFrame) -> ModelResult:
    usable = features.dropna(subset=["excess_return"]).copy()
    usable = usable.dropna(subset=FEATURE_COLUMNS, how="all")
    split_idx = int(len(sorted(usable["date"].unique())) * 0.72)
    split_date = sorted(usable["date"].unique())[split_idx]
    train = usable[usable["date"] <= split_date]
    test = usable[usable["date"] > split_date]

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                VotingRegressor(
                    estimators=[
                        (
                            "hgb",
                            HistGradientBoostingRegressor(
                                max_iter=180,
                                learning_rate=0.045,
                                max_leaf_nodes=24,
                                l2_regularization=0.05,
                                random_state=7,
                            ),
                        ),
                        (
                            "rf",
                            RandomForestRegressor(
                                n_estimators=70,
                                min_samples_leaf=10,
                                max_features=0.7,
                                n_jobs=-1,
                                random_state=17,
                            ),
                        ),
                        (
                            "extra",
                            ExtraTreesRegressor(
                                n_estimators=90,
                                min_samples_leaf=8,
                                max_features=0.8,
                                n_jobs=-1,
                                random_state=29,
                            ),
                        ),
                    ],
                    weights=[0.48, 0.25, 0.27],
                ),
            ),
        ]
    )
    model.fit(train[FEATURE_COLUMNS], train["excess_return"])
    return ModelResult(model, len(train), len(test), pd.Timestamp(split_date), pd.Timestamp(test["date"].min()))


def score_frame(model: Pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    scored["score"] = model.predict(scored[FEATURE_COLUMNS])
    scored["score_rank"] = scored.groupby("date")["score"].rank(pct=True)
    return scored
