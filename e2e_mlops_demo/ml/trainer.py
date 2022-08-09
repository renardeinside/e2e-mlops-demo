import os
import tempfile
from typing import Optional

import mlflow.sklearn
from hyperopt import STATUS_OK, tpe, fmin, Trials
from pyspark.cloudpickle import dump

from e2e_mlops_demo.ml.provider import Provider
from e2e_mlops_demo.models import ModelData, DatabricksApiInfo, SearchSpace, MlflowInfo


class Trainer:
    def __init__(
        self,
        model_data: ModelData,
        experiment_name: str,
        databricks_info: Optional[DatabricksApiInfo] = None,
        mlflow_info: Optional[MlflowInfo] = None,
    ):
        self.data = model_data
        self._databricks_info = databricks_info
        self._mlflow_info = mlflow_info
        self._experiment_id = self._get_experiment_id(experiment_name)
        self._parent_run_id = self.initialize_parent_run()
        self.verify_serialization()

    @staticmethod
    def _get_experiment_id(experiment_name) -> str:
        _exp = mlflow.get_experiment_by_name(experiment_name)
        if _exp:
            return _exp.experiment_id
        else:
            return mlflow.create_experiment(experiment_name)

    def initialize_parent_run(self) -> str:
        with mlflow.start_run(experiment_id=self._experiment_id) as parent_run:
            return parent_run.info.run_id

    def _setup_mlflow_auth(self):  # pragma: no cover
        # this is required for a proper mlflow setup on the worker nodes
        if self._databricks_info:
            os.environ["DATABRICKS_HOST"] = self._databricks_info.host
            os.environ["DATABRICKS_TOKEN"] = self._databricks_info.token

        if self._mlflow_info:
            mlflow.set_registry_uri(self._mlflow_info.registry_uri)
            mlflow.set_tracking_uri(self._mlflow_info.tracking_uri)
        else:
            mlflow.set_registry_uri("databricks")
            mlflow.set_tracking_uri("databricks")

    def _objective(self, parameters: SearchSpace):
        """
        Please note that logging functionality won't work inside the training function.
        This function is running on executor instances.
        """
        if not self._parent_run_id:
            raise RuntimeError("Parent run id is not defined")

        self._setup_mlflow_auth()

        with mlflow.start_run(
            run_id=self._parent_run_id, experiment_id=self._experiment_id
        ):
            with mlflow.start_run(nested=True, experiment_id=self._experiment_id):
                # pipeline = Provider.get_pipeline(parameters)
                # pipeline.fit(self.data.train.X, self.data.train.y)
                # result = mlflow.sklearn.eval_and_log_metrics(
                #     pipeline, self.data.test.X, self.data.test.y, prefix="test_"
                # )
                mlflow.log_metric("test", 1)
                # mlflow.log_params(parameters.get("classifier", {}))
                return {"status": STATUS_OK, "loss": -1}

    def train(self, max_evals: int, trials: Trials):

        best_params = fmin(
            fn=self._objective,
            space=Provider.get_search_space(),
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
        )
        return best_params

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
