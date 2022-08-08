import mlflow
import mlflow.sklearn
import pandas as pd
from hyperopt import fmin, tpe, SparkTrials, Trials

from e2e_mlops_demo.common import Task
from e2e_mlops_demo.providers import Provider, Trainer, DataProvider


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
        return SparkTrials()

    def setup_mlflow(self):
        mlflow.set_experiment(self.conf["experiment"])

    def _train_model(self, data: pd.DataFrame):
        self.logger.info("Starting the model training")
        model_data = DataProvider.provide(data, self.logger)
        with mlflow.start_run():
            trainer = Trainer(model_data)
            best_params = fmin(
                fn=trainer.train,
                space=Provider.get_search_space(),
                algo=tpe.suggest,
                max_evals=self.conf.get("max_evals", 20),
                trials=self._get_trials(),
            )
            self.logger.info(f"Best params {best_params}")
        self.logger.info("Model training finished")

    def launch(self):
        self.logger.info("Launching sample ETL job")
        self.setup_mlflow()
        _data = self._read_data()
        self._train_model(_data)
        self.logger.info("Sample ETL job finished!")


def entrypoint():  # pragma: no cover
    task = ModelBuilderTask()
    task.launch()


# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == "__main__":
    entrypoint()
