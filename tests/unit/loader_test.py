from pyspark.sql import SparkSession

from e2e_mlops_demo.tasks.dataset_loader_task import DatasetLoaderTask


def test_loader(spark: SparkSession):
    task_conf = {"output": {"database": "test", "table": "fraud_data"}, "limit": 100}
    task = DatasetLoaderTask(spark, task_conf)
    task.launch()
    assert spark.table(task.get_output_table_name()).count() == task_conf["limit"]
