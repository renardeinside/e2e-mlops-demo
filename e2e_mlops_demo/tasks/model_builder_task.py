from cgi import test
import logging
from e2e_mlops_demo.common import Task
from e2e_mlops_demo.tasks.ml.providers import Provider, Trainer
import pandas as pd
import mlflow.sklearn
import mlflow
from hyperopt import fmin, STATUS_OK, tpe, SparkTrials, Trials
from hyperopt.tpe import logger
from sklearn.metrics import roc_auc_score

logger = logging.getLogger("hyperopt-trainer")
logger.setLevel(logging.INFO)


class ModelBuilderTask(Task):

    def _read_data(self) -> pd.DataFrame:
        db = self.conf["input"]["database"]
        table = self.conf["input"]["table"]
        self.logger.info(f"Reading dataset from {db}.{table}")
        _data: pd.DataFrame = self.spark.table(f"{db}.{table}").toPandas()
        self.logger.info(f"Loaded dataset, total size: {len(_data)}")
        return _data

    def setup_mlflow(self):
        mlflow.set_experiment(self.conf["experiment"])

    def _train_model(self, data: pd.DataFrame):
        trainer = Trainer(data)
        trials = SparkTrials(1)
        
        best_params = fmin(
            fn=trainer.train,
            space=Provider.get_search_space(),
            algo=tpe.suggest,
            max_evals=5,
            trials=trials
        )
        self.logger.info(f"Best params {best_params}")

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
