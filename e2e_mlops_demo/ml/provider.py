import logging
from typing import Any, Optional

import pandas as pd
from hyperopt import hp
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

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
                "max_depth": hp.uniformint("max_depth", 3, 10),
                "n_estimators": hp.uniformint("n_estimators", 10, 100),
                "learning_rate": hp.uniform("learning_rate", 0.001, 0.5),
                "reg_alpha": hp.uniform("reg_alpha", 0.01, 0.1),
                "base_score": hp.uniform("base_score", 0.001, 0.1)
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
                    XGBClassifier(**params.get("classifier", {})),
                ),
            ]
        )
        return pipeline

    @classmethod
    def get_data(
        cls,
        data: pd.DataFrame,
        logger: Optional[Any] = None,
        limit: Optional[int] = None,
    ) -> ModelData:
        if not logger:
            logger = logging.getLogger(cls.__name__)
            logger.setLevel(logging.INFO)
        # split into columns
        X = data.drop(columns=[cls.TARGET_COLUMN])
        y = data[cls.TARGET_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=cls.TEST_SIZE,
            random_state=cls.RANDOM_STATE,
        )

        # over-sample the target class
        X_train, y_train = ADASYN(random_state=cls.RANDOM_STATE).fit_resample(
            X_train, y_train
        )

        if limit:
            X_train = X_train.head(limit)
            y_train = y_train.head(limit)
            X_test = X_test.head(limit)
            y_test = y_test.head(limit)

        # split into train test
        logger.info(f"Train shape: {X_train.shape}")
        logger.info(f"Test  shape: {X_test.shape}")
        logger.info(
            f"target percentage in oversampled train: {y_train.sum() / len(y_train)}"
        )
        logger.info(
            f"target percentage in oversampled test: {y_train.sum() / len(y_train)}"
        )
        return ModelData(
            train=TrainData(X=X_train, y=y_train), test=TestData(X=X_test, y=y_test)
        )
