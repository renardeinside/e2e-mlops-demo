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


# def test_jobs(spark: SparkSession, tmp_path: Path):
#     logging.info("Testing the ETL job")
#     common_config = {"database": "default", "table": "sklearn_housing"}
#     test_etl_config = {"output": common_config}
#     etl_job = SampleETLTask(spark, test_etl_config)
#     etl_job.launch()
#     table_name = f"{test_etl_config['output']['database']}.{test_etl_config['output']['table']}"
#     _count = spark.table(table_name).count()
#     assert _count > 0
#     logging.info("Testing the ETL job - done")

#     logging.info("Testing the ML job")
#     test_ml_config = {
#         "input": common_config,
#         "experiment": "/Shared/e2e-mlops-demo/sample_experiment"
#     }
#     ml_job = SampleMLTask(spark, test_ml_config)
#     ml_job.launch()
#     experiment = mlflow.get_experiment_by_name(test_ml_config['experiment'])
#     assert experiment is not None
#     runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
#     assert runs.empty is False
#     logging.info("Testing the ML job - done")
