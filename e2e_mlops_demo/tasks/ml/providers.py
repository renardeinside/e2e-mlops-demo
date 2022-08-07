import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from hyperopt import hp
from typing import Dict
from hyperopt.pyll.base import Apply as HyperOptSpace
import pandas as pd
from typing import Any
from sklearn.model_selection import train_test_split
from hyperopt import STATUS_OK


class Provider:
    def get_search_space() -> Dict[str, HyperOptSpace]:
        search_space = {
            "classifier": {
                "criterion": hp.choice("criterion", ["gini", "entropy", "log_loss"]),
                "max_depth": hp.uniformint("max_depth", 3, 10),
            }
        }
        return search_space

    @staticmethod
    def get_pipeline() -> Pipeline:
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("classifier", RandomForestClassifier())]
        )
        return pipeline


class Trainer:
    TARGET_COLUMN: str = "target"
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    def __init__(self, data: pd.DataFrame) -> None:
        self.logger = logging.getLogger("model-trainer")
        self.logger.setLevel(logging.INFO)
        self.X_train, self.X_test, self.y_train, self.y_test = self.prepare_X_y(data)

    def prepare_X_y(self, data):
        X = data.drop(columns=[self.TARGET_COLUMN])
        y = data[self.TARGET_COLUMN]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=self.TEST_SIZE, random_state=self.RANDOM_STATE
        )
        return X_train, X_test, y_train, y_test

    def train(self, parameters: Dict[str, Dict[str, Any]]):
        self.logger.info(f"Started training function with parameters {parameters}")
        pipeline = Provider.get_pipeline()
        return {"status": STATUS_OK, "loss": -1}
