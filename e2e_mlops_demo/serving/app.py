import os
import tempfile
from typing import Optional

import mlflow.sklearn
from fastapi import FastAPI
from mlflow.entities.model_registry import ModelVersion
from mlflow.pyfunc import PyFuncModel
from mlflow.tracking import MlflowClient

from e2e_mlops_demo.models import PredictionInfo
from e2e_mlops_demo.serving._types import get_pydantic_model


def load_model(model_name: str) -> tuple[PyFuncModel, ModelVersion]:
    _client = MlflowClient()
    _model = _client.get_registered_model(model_name)
    _versions = _model.latest_versions

    if not _versions:
        raise Exception(f"No versions for model {model_name} were found!")

    latest_version = _versions[-1]

    full_model_uri = f"models:/{model_name}/{latest_version.version}"
    with tempfile.TemporaryDirectory() as temp_dir:
        return (
            mlflow.pyfunc.load_model(full_model_uri, temp_dir + "/model"),
            latest_version,
        )


def get_app(model_name: Optional[str] = None) -> FastAPI:
    if not model_name:
        if "MODEL_NAME" not in os.environ:
            raise Exception("Please provide model name to serve")
        model_name = os.environ["MODEL_NAME"]
    model, version_info = load_model(model_name)
    PayloadModel = get_pydantic_model(model.metadata.get_input_schema(), "Payload")
    app = FastAPI()

    @app.post(
        "/invocations",
        response_model=PredictionInfo,
        summary="predicts if transaction is fraudulent or normal",
        description="""Returns predictions for the credit card transaction classification.
        Please note that all parameters shall be provided in the request body, empty values are not allowed.
        """,
    )
    def invoke(payload: PayloadModel) -> PredictionInfo:
        _value = model.predict([payload.dict()])[0]
        return PredictionInfo(value=_value, model_version=version_info.version)

    return app
