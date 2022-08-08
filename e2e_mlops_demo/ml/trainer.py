import os
import tempfile
from typing import Optional

import mlflow.sklearn
from hyperopt import STATUS_OK
from pyspark.cloudpickle import dump

from e2e_mlops_demo.models import ModelData, DatabricksApiInfo, MlflowInfo, SearchSpace
from e2e_mlops_demo.ml.provider import Provider


class Trainer:
    def __init__(
        self,
        model_data: ModelData,
        experiment: str,
        databricks_info: Optional[DatabricksApiInfo] = None,
    ):
        self.data = model_data
        self.verify_serialization()
        self._mlflow_info = self.get_mlflow_info(experiment)
        self._databricks_info = databricks_info

    @staticmethod
    def get_mlflow_info(experiment: str):
        return MlflowInfo(
            tracking_uri="databricks", registry_uri="databricks", experiment=experiment
        )

    def setup_mlflow_properties(self):  # pragma: no cover

        # this is required for a proper mlflow setup on the worker nodes
        if self._databricks_info:
            os.environ["DATABRICKS_HOST"] = self._databricks_info.host
            os.environ["DATABRICKS_TOKEN"] = self._databricks_info.token

        mlflow.set_registry_uri(self._mlflow_info.registry_uri)
        mlflow.set_tracking_uri(self._mlflow_info.tracking_uri)
        mlflow.set_experiment(self._mlflow_info.experiment)

    def train(self, parameters: SearchSpace):
        """
        Please note that logging functionality won't work inside the training function.
        This function is running on executor instances.
        """
        self.setup_mlflow_properties()
        pipeline = Provider.get_pipeline(parameters)
        pipeline.fit(self.data.train.X, self.data.train.y)
        result = mlflow.sklearn.eval_and_log_metrics(
            pipeline, self.data.test.X, self.data.test.y, prefix="test_"
        )
        return {"status": STATUS_OK, "loss": -1 * result["test_roc_auc_score"]}

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
