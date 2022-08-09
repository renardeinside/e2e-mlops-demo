from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from hyperopt import Trials, SparkTrials, hp
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession

from e2e_mlops_demo.ml.provider import Provider
from e2e_mlops_demo.ml.trainer import Trainer
from e2e_mlops_demo.models import MlflowInfo, SourceMetadata
from e2e_mlops_demo.tasks.dataset_loader_task import DatasetLoaderTask
from e2e_mlops_demo.tasks.model_builder_task import ModelBuilderTask
from e2e_mlops_demo.utils import EnvironmentInfoProvider


def test_loader(spark: SparkSession):
    task_conf = {"output": {"database": "test", "table": "fraud_data"}, "limit": 100}
    task = DatasetLoaderTask(spark, task_conf)
    task.launch()
    assert spark.table(task.get_output_table_name()).count() == task_conf["limit"]


def test_trainer_trials(
    spark: SparkSession, dataset_fixture: pd.DataFrame, mlflow_local: MlflowInfo
):
    model_data = Provider.get_data(
        dataset_fixture, SourceMetadata(version=0, database="db", table="table")
    )

    def custom_search_space():
        search_space = {
            "classifier": {
                "max_depth": hp.choice("max_depth", np.arange(3, 10, dtype=int)),
            }
        }
        return search_space

    Provider.get_search_space = custom_search_space

    experiment_name = "test-trainer"
    model_name = "test-model"
    trainer = Trainer(model_data, experiment_name, mlflow_local)
    trainer.train(
        max_evals=2, trials=SparkTrials(1, spark_session=spark), model_name=model_name
    )
    found_model = [
        _m for _m in MlflowClient().list_registered_models() if _m.name == model_name
    ]
    assert len(found_model) == 1


def test_builder(spark: SparkSession, dataset_fixture: pd.DataFrame, mlflow_local):
    with patch.object(
        EnvironmentInfoProvider, "get_mlflow_info", return_value=mlflow_local
    ):
        builder = ModelBuilderTask(
            spark, {"experiment": "test", "max_evals": 5, "model_name": "builder-test"}
        )
        builder._read_data = MagicMock(
            return_value=(
                dataset_fixture,
                SourceMetadata(version=0, database="db", table="table"),
            )
        )
        builder._get_trials = MagicMock(return_value=Trials())
        builder.launch()
