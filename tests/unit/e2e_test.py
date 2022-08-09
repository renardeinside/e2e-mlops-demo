from unittest.mock import MagicMock, patch

import pandas as pd
from hyperopt import SparkTrials
from pyspark.sql import SparkSession

from e2e_mlops_demo.ml.provider import Provider
from e2e_mlops_demo.ml.trainer import Trainer
from e2e_mlops_demo.models import MlflowInfo
from e2e_mlops_demo.tasks.dataset_loader_task import DatasetLoaderTask
from e2e_mlops_demo.tasks.model_builder_task import ModelBuilderTask
from e2e_mlops_demo.utils import EnvironmentInfoProvider


def test_loader(spark: SparkSession):
    task_conf = {"output": {"database": "test", "table": "fraud_data"}, "limit": 100}
    task = DatasetLoaderTask(spark, task_conf)
    task.launch()
    assert spark.table(task.get_output_table_name()).count() == task_conf["limit"]


def test_trainer(
    spark: SparkSession, dataset_fixture: pd.DataFrame, mlflow_local: MlflowInfo
):
    model_data = Provider.get_data(dataset_fixture)
    experiment_name = "test-trainer"
    trainer = Trainer(model_data, experiment_name, mlflow_info=mlflow_local)
    best_params = trainer.train(max_evals=2, trials=SparkTrials())
    print(best_params)


def test_builder(spark: SparkSession, dataset_fixture: pd.DataFrame, mlflow_local):
    with patch.object(
        EnvironmentInfoProvider, "get_databricks_api_info", return_value=None
    ), patch.object(
        EnvironmentInfoProvider, "get_mlflow_info", return_value=mlflow_local
    ):
        builder = ModelBuilderTask(spark, {"experiment": "test", "max_evals": 2})
        builder._read_data = MagicMock(return_value=dataset_fixture)
        builder._get_trials = MagicMock(return_value=SparkTrials())
        builder.launch()
