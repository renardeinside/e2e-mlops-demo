from unittest.mock import MagicMock, patch

import pandas as pd
from hyperopt import Trials
from pyspark.sql import SparkSession

from e2e_mlops_demo.tasks.dataset_loader_task import DatasetLoaderTask
from e2e_mlops_demo.ml.provider import Provider
from e2e_mlops_demo.ml.trainer import Trainer
from e2e_mlops_demo.tasks.model_builder_task import ModelBuilderTask


def test_loader(spark: SparkSession):
    task_conf = {"output": {"database": "test", "table": "fraud_data"}, "limit": 100}
    task = DatasetLoaderTask(spark, task_conf)
    task.launch()
    assert spark.table(task.get_output_table_name()).count() == task_conf["limit"]


def test_trainer(spark: SparkSession, dataset_fixture: pd.DataFrame):
    model_data = Provider.get_data(dataset_fixture)
    trainer = Trainer(model_data, "test-trainer")
    trainer.setup_mlflow_properties = MagicMock(return_value=None)
    trainer.train({})


def test_builder(spark: SparkSession, dataset_fixture: pd.DataFrame):
    builder = ModelBuilderTask(spark, {"experiment": "test"})
    with patch.object(Trainer, "setup_mlflow_properties", return_value=None):
        builder._get_databricks_api_info = MagicMock(return_value=None)
        builder._read_data = MagicMock(return_value=dataset_fixture)
        builder._get_trials = MagicMock(return_value=Trials())
        builder.launch()
