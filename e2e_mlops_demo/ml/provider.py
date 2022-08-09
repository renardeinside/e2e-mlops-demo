import logging
from typing import Any, Optional

import pandas as pd
from hyperopt import hp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN

from e2e_mlops_demo.models import (
    SearchSpace,
    TrainData,
    TestData,
    ModelData,
)


class Provider:
    TARGET_COLUMN: str = "target"
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    @staticmethod
    def get_search_space() -> SearchSpace:
        search_space = {
            "classifier": {
                "criterion": hp.choice("criterion", ["gini", "entropy", "log_loss"]),
                "max_depth": hp.uniformint("max_depth", 3, 10),
            }
        }
        return search_space

    @staticmethod
    def get_pipeline(params: Optional[SearchSpace]) -> Pipeline:
        if params is None:
            params = {}
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    RandomForestClassifier(
                        **params.get("classifier", {}), class_weight={0: 1, 1: 100}
                    ),
                ),
            ]
        )
        return pipeline

    @classmethod
    def get_data(cls, data: pd.DataFrame, logger: Optional[Any] = None) -> ModelData:
        if not logger:
            logger = logging.getLogger(cls.__name__)
            logger.setLevel(logging.INFO)
        # split into columns
        X = data.drop(columns=[cls.TARGET_COLUMN])
        y = data[cls.TARGET_COLUMN]

        # oversample-sample the target class
        X_resampled, y_resampled = ADASYN(random_state=cls.RANDOM_STATE).fit_resample(
            X, y
        )

        # split into train test
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled,
            y_resampled,
            test_size=cls.TEST_SIZE,
            random_state=cls.RANDOM_STATE,
        )
        logger.info(f"Train shape: {X_train.shape}")
        logger.info(f"Test  shape: {X_test.shape}")
        logger.info(f"target percentage in oversampled train: {y_train.sum() / len(y_train)}")
        logger.info(f"target percentage in oversampled test: {y_train.sum() / len(y_train)}")
        return ModelData(
            train=TrainData(X=X_train, y=y_train), test=TestData(X=X_test, y=y_test)
        )
