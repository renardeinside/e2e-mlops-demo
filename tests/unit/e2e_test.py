from e2e_mlops_demo.tasks.dataset_loader_task import DatasetLoaderTask
from pyspark.sql import SparkSession
from pathlib import Path
import mlflow
import logging


def test_loader(spark: SparkSession, tmp_path: Path):
    task_conf = {"output": {"database": "test", "table": "fraud_data"}, "limit": 100}
    task = DatasetLoaderTask(spark, task_conf)
    task.launch()
    assert spark.table(task.get_output_table_name()).count() == task_conf["limit"]