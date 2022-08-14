from typing import Tuple

import pandas as pd
from delta.tables import DeltaTable
from hyperopt import SparkTrials, Trials

from e2e_mlops_demo.common import Task
from e2e_mlops_demo.ml.provider import Provider
from e2e_mlops_demo.ml.trainer import Trainer
from e2e_mlops_demo.models import SourceMetadata
from e2e_mlops_demo.utils import EnvironmentInfoProvider


class ModelBuilderTask(Task):
    def _read_data(self) -> Tuple[pd.DataFrame, SourceMetadata]:
        db = self.conf["input"]["database"]
        table_name = self.conf["input"]["table"]
        full_table_name = f"{db}.{table_name}"
        table = DeltaTable.forName(self.spark, full_table_name)
        last_version = table.history(limit=1).toPandas()["version"][0]
        _data = self.spark.sql(f"select * from {full_table_name} VERSION AS OF {last_version}").toPandas()
        self.logger.info(f"Loaded dataset, total size: {len(_data)}")
        self.logger.info(f"Dataset version: {last_version}")
        return _data, SourceMetadata(version=last_version, database=db, table=table_name)

    def _get_num_executors(self) -> int:  # pragma: no cover
        tracker = self.spark.sparkContext._jsc.sc().statusTracker()  # noqa
        return len(tracker.getExecutorInfos()) - 1

    def _get_trials(self) -> Trials:
        return SparkTrials(parallelism=self._get_num_executors())

    def _train_model(self, data: pd.DataFrame, source_metadata: SourceMetadata):
        self.logger.info("Starting the model training")
        model_data = Provider.get_data(data, source_metadata, self.logger, limit=self.conf.get("limit"))
        mlflow_info = EnvironmentInfoProvider.get_mlflow_info()
        trainer = Trainer(model_data, self.conf["experiment"], mlflow_info)
        trainer.train(self.conf.get("max_evals", 20), self._get_trials(), self.conf["model_name"])
        self.logger.info("Model training finished")

    def launch(self):
        self.logger.info("Launching sample ETL job")
        data, source_metadata = self._read_data()
        self._train_model(data, source_metadata)
        self.logger.info("Sample ETL job finished!")


def entrypoint():  # pragma: no cover
    task = ModelBuilderTask()
    task.launch()


# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == "__main__":
    entrypoint()
