import logging
import tempfile
from typing import Any, Optional

import mlflow.sklearn
import pandas as pd
from hyperopt import STATUS_OK
from hyperopt import hp
from pyspark.cloudpickle.cloudpickle_fast import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from e2e_mlops_demo.models import SearchSpace, TrainData, TestData, ModelData


class Provider:
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
                ("classifier", RandomForestClassifier(**params.get("classifier", {}))),
            ]
        )
        return pipeline


class DataProvider:
    TARGET_COLUMN: str = "target"
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    @classmethod
    def provide(cls, data: pd.DataFrame, logger: Optional[Any] = None) -> ModelData:
        if not logger:
            logger = logging.getLogger(cls.__name__)
            logger.setLevel(logging.INFO)
        X = data.drop(columns=[cls.TARGET_COLUMN])
        y = data[cls.TARGET_COLUMN]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=cls.TEST_SIZE, random_state=cls.RANDOM_STATE
        )
        logger.info(f"Train shape: {X_train.shape}")
        logger.info(f"Test  shape: {X_test.shape}")
        logger.info(f"target percentage in train: {y_train.sum() / len(y_train)}")
        logger.info(f"target percentage in test: {y_train.sum() / len(y_train)}")
        logger.info(f"target percentage in dataset: {y.sum() / len(y)}")
        return ModelData(
            train=TrainData(X=X_train, y=y_train), test=TestData(X=X_test, y=y_test)
        )


class Trainer:
    def __init__(self, model_data: ModelData):
        self.data = model_data
        self.verify_serialization()

    def train(self, parameters: SearchSpace):
        """
        Please note that logging functionality won't work inside the training function.
        This function is running on executor instances.
        """
        pipeline = Provider.get_pipeline(parameters)
        pipeline.fit(self.data.train.X, self.data.train.y)
        result = mlflow.sklearn.eval_and_log_metrics(pipeline, self.data.test.X, self.data.test.y, prefix="test_")
        return {"status": STATUS_OK, "loss": -1 * result['test_roc_auc_score']}

    def verify_serialization(self):
        try:
            with tempfile.TemporaryFile() as t_file:
                dump(self, t_file)
        except Exception as _:
            raise RuntimeError(
                f"""
            Failed to serialize model builder functional class {self.__class__.__name__}.
            This typically means that functional class contains dependencies that cannot be serialized, for instance:
                - SparkSession
                - any other runtime-dependent objects
            Please check that these objects are not defined as class or object properties.
            """
            )
