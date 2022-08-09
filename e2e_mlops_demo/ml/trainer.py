import inspect
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import mlflow.sklearn
from hyperopt import STATUS_OK, tpe, fmin, Trials
from mlflow.models.signature import infer_signature
from pyspark.cloudpickle import dump
from sklearn.pipeline import Pipeline
from sklearn.metrics import cohen_kappa_score
import e2e_mlops_demo
from e2e_mlops_demo.ml.provider import Provider
from e2e_mlops_demo.models import ModelData, SearchSpace, MlflowInfo


class Trainer:
    def __init__(
        self,
        model_data: ModelData,
        experiment_name: str,
        mlflow_info: Optional[MlflowInfo] = None,
    ):
        self.data = model_data
        self._mlflow_info = mlflow_info
        self.experiment_id = self.prepare_experiment(experiment_name)
        self.parent_run_id = self.initialize_parent_run(self.experiment_id)
        self.verify_serialization()

    @staticmethod
    def prepare_experiment(experiment_name) -> str:
        _exp = mlflow.set_experiment(experiment_name)
        return _exp.experiment_id

    @staticmethod
    def initialize_parent_run(experiment_id: str) -> str:
        with mlflow.start_run(experiment_id=experiment_id) as parent_run:
            return parent_run.info.run_id

    def _train_model_and_log_results(
        self, parameters: SearchSpace
    ) -> Tuple[Dict[str, Any], Pipeline]:
        pipeline = Provider.get_pipeline(parameters)
        pipeline.fit(self.data.train.X, self.data.train.y)

        _params = pipeline.get_params()
        _params.pop("steps")

        for step_name, _ in pipeline.steps:
            _params.pop(step_name)

        mlflow.log_params(_params)
        results = mlflow.sklearn.eval_and_log_metrics(
            pipeline, self.data.test.X, self.data.test.y, prefix="test_"
        )
        kappa = cohen_kappa_score(pipeline.predict(self.data.test.X), self.data.test.y)
        mlflow.log_metric("kappa", kappa)
        results["test_kappa"] = kappa
        return results, pipeline

    def setup_mlflow(self):
        if self._mlflow_info:
            mlflow.set_tracking_uri(self._mlflow_info.tracking_uri)
            mlflow.set_registry_uri(self._mlflow_info.registry_uri)

    def _objective(self, parameters: SearchSpace):
        """
        Please note that logging functionality won't work inside the training function.
        This function is running on executor instances.
        """
        self.setup_mlflow()
        if not self.parent_run_id:
            raise RuntimeError("Parent run id is not defined")
        with mlflow.start_run(
            run_id=self.parent_run_id, experiment_id=self.experiment_id
        ):
            with mlflow.start_run(nested=True, experiment_id=self.experiment_id):
                results, _ = self._train_model_and_log_results(parameters)
                return {"status": STATUS_OK, "loss": -1 * results["test_kappa"]}

    def _register_model(self, model: Pipeline, model_name: str):
        signature = infer_signature(
            self.data.train.X, model.predict_proba(self.data.train.X)
        )
        code_path = Path(inspect.getfile(e2e_mlops_demo)).parent
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name,
            signature=signature,
            code_paths=[str(code_path)],
        )

    def train(self, max_evals: int, trials: Trials, model_name: str) -> None:

        classifier_params = fmin(
            fn=self._objective,
            space=Provider.get_search_space(),
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
        )

        best_params = {
            "classifier": classifier_params
        }

        with mlflow.start_run(
            run_id=self.parent_run_id, experiment_id=self.experiment_id
        ):
            _, final_model = self._train_model_and_log_results(best_params)
            self._register_model(final_model, model_name)

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
