import logging
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

import mlflow
import pandas as pd
import pytest
from delta import configure_spark_with_delta_pip
from imblearn.datasets import make_imbalance
from pyspark.sql import SparkSession
from sklearn.datasets import make_classification

from e2e_mlops_demo.ml.provider import Provider
from e2e_mlops_demo.ml.trainer import Trainer

from e2e_mlops_demo.models import MlflowInfo, SourceMetadata


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    """
    This fixture provides preconfigured SparkSession with Hive and Delta support.
    After the test session, temporary warehouse directory is deleted.
    :return: SparkSession
    """
    logging.info("Configuring Spark session for testing environment")
    warehouse_dir = tempfile.TemporaryDirectory().name
    _builder = (
        SparkSession.builder.master("local[1]")
        .config("spark.hive.metastore.warehouse.dir", Path(warehouse_dir).as_uri())
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )
    spark: SparkSession = configure_spark_with_delta_pip(_builder).getOrCreate()
    logging.info("Spark session configured")
    yield spark
    logging.info("Shutting down Spark session")
    spark.stop()
    if Path(warehouse_dir).exists():
        shutil.rmtree(warehouse_dir)


@pytest.fixture(scope="session")
def docker_compose_file():
    _p = Path(__file__).parent.parent / "docker" / "docker-compose-test.yml"
    return _p.absolute()


@pytest.fixture(scope="session", autouse=True)
def mlflow_local(docker_services) -> MlflowInfo:
    """
    This fixture provides local instance of mlflow with support for tracking and registry functions.
    After the test session:
    * temporary storage for tracking and registry is deleted.
    * Active run will be automatically stopped to avoid verbose errors.
    :return: None
    """
    logging.info("Configuring local MLflow instance")
    _port = docker_services.port_for("mlflow", 5005)
    mlflow_uri = f"http://localhost:{_port}"
    mlflow.set_registry_uri(mlflow_uri)
    mlflow.set_tracking_uri(mlflow_uri)

    yield MlflowInfo(tracking_uri=mlflow_uri, registry_uri=mlflow_uri)

    mlflow.end_run()
    logging.info("Test session finished, unrolling the MLflow instance")


@pytest.fixture(scope="function")
def dataset_fixture() -> pd.DataFrame:
    X, y = make_classification(n_samples=10000, n_classes=2, random_state=42)
    X, y = make_imbalance(X, y, sampling_strategy={0: 1000, 1: 100}, random_state=42)
    df = pd.DataFrame(X, columns=[f"v{i}" for i in range(X.shape[-1])])
    df["target"] = y
    return df


@pytest.fixture(scope="function")
def model_fixture(spark: SparkSession, dataset_fixture: pd.DataFrame, mlflow_local: MlflowInfo) -> str:
    logging.info(f"Preparing model instance in mlflow registry {mlflow_local.registry_uri}")
    model_data = Provider.get_data(dataset_fixture, SourceMetadata(version=0, database="db", table="table"))

    experiment_name = "test-trainer"
    model_name = f"test-model-{uuid4()}"
    trainer = Trainer(model_data, experiment_name, mlflow_local)
    _test_model = Provider.get_pipeline({})
    _test_model.fit(model_data.train.X, model_data.train.y)
    trainer.register_model(_test_model, model_name)
    logging.info(f"Model with name {model_name} is registered in Model Registry")
    yield model_name
