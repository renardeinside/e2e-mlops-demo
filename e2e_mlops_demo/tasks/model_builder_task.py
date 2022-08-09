import pandas as pd
from hyperopt import SparkTrials, Trials

from e2e_mlops_demo.common import Task
from e2e_mlops_demo.ml.provider import Provider
from e2e_mlops_demo.ml.trainer import Trainer
from e2e_mlops_demo.utils import EnvironmentInfoProvider


class ModelBuilderTask(Task):
    def _read_data(self) -> pd.DataFrame:
        db = self.conf["input"]["database"]
        table = self.conf["input"]["table"]
        self.logger.info(f"Reading dataset from {db}.{table}")
        _data: pd.DataFrame = self.spark.table(f"{db}.{table}").toPandas()
        self.logger.info(f"Loaded dataset, total size: {len(_data)}")
        return _data

    @staticmethod
    def _get_trials() -> Trials:
        return SparkTrials(parallelism=2)

    def _train_model(self, data: pd.DataFrame):
        self.logger.info("Starting the model training")
        model_data = Provider.get_data(data, self.logger)
        databricks_api_info = EnvironmentInfoProvider.get_databricks_api_info(
            self.dbutils
        )
        mlflow_info = EnvironmentInfoProvider.get_mlflow_info()
        trainer = Trainer(
            model_data, self.conf["experiment"], databricks_api_info, mlflow_info
        )
        best_params = trainer.train(self.conf.get("max_evals", 20), self._get_trials())
        self.logger.info(f"Best model params: {best_params}")
        self.logger.info("Model training finished")

    def launch(self):
        self.logger.info("Launching sample ETL job")
        _data = self._read_data()
        self._train_model(_data)
        self.logger.info("Sample ETL job finished!")


def entrypoint():  # pragma: no cover
    task = ModelBuilderTask()
    task.launch()


# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == "__main__":
    entrypoint()
