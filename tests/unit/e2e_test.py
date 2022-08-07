from unittest.mock import MagicMock
from e2e_mlops_demo.tasks.dataset_loader_task import DatasetLoaderTask
from e2e_mlops_demo.tasks.model_builder_task import ModelBuilderTask
from pyspark.sql import SparkSession


def test_loader(spark: SparkSession):
    task_conf = {"output": {"database": "test", "table": "fraud_data"}, "limit": 100}
    task = DatasetLoaderTask(spark, task_conf)
    task.launch()
    assert spark.table(task.get_output_table_name()).count() == task_conf["limit"]


def test_builder(spark: SparkSession):
    _df = DatasetLoaderTask(spark, {}).get_data(100)
    builder = ModelBuilderTask(spark, {"experiment": "test"})
    builder._read_data = MagicMock(return_value=_df)
    builder.launch()
