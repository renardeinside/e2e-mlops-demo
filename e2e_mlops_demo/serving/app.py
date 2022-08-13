import os
import tempfile
from typing import Optional, List

import mlflow.sklearn
from fastapi import FastAPI
from fastapi_versioning import VersionedFastAPI, version

from mlflow.entities.model_registry import ModelVersion
from mlflow.pyfunc import PyFuncModel
from mlflow.tracking import MlflowClient

from e2e_mlops_demo.models import PredictionInfo
from e2e_mlops_demo.serving._types import get_pydantic_model
from e2e_mlops_demo import __version__


def get_model_versions(model_name: str) -> List[ModelVersion]:
    _client = MlflowClient()
    all_versions = _client.search_model_versions(f"name = '{model_name}'")

    if not all_versions:
        raise Exception(f"No versions for model {model_name} were found!")

    requested_versions_string = os.environ.get("MODEL_VERSIONS")

    if not requested_versions_string:
        _version = max(all_versions, key=lambda v: int(v.version))
        return [_version]
    else:
        requested_versions = [int(v) for v in requested_versions_string.split(",")]
        prepared_versions = []
        for req_v in requested_versions:
            if req_v not in [v.version for v in all_versions]:
                raise Exception(
                    f"Requested model version {req_v} doesn't exist in registry"
                )
            else:
                _v = [v for v in all_versions if v.version == req_v][0]
                prepared_versions.append(_v)
        return prepared_versions


def load_model(
    model_name: str, model_version: ModelVersion
) -> tuple[PyFuncModel, ModelVersion]:
    full_model_uri = f"models:/{model_name}/{model_version.version}"
    with tempfile.TemporaryDirectory() as temp_dir:
        return (
            mlflow.pyfunc.load_model(full_model_uri, temp_dir + "/model"),
            model_version,
        )


def get_app(model_name: Optional[str] = None) -> FastAPI:
    if not model_name:
        if "MODEL_NAME" not in os.environ:
            raise Exception("Please provide model name to serve")
        model_name = os.environ["MODEL_NAME"]

    app = FastAPI(title="Credit Card Transactions Classifier 🚀", version=__version__)
    versions = get_model_versions(model_name)
    for _version in versions:
        model, version_info = load_model(model_name, _version)
        PayloadModel = get_pydantic_model(model.metadata.get_input_schema(), "Payload")

        @app.post(
            "/invocations",
            response_model=PredictionInfo,
            summary="predictions for the credit card transaction classification",
            description="""Returns predictions for the credit card transaction classification. 
            Empty values (e.g. `null`) are not allowed and won't pass the validation, resulting in status code `422`.
            """,
        )
        @version(_version.version)
        def invoke(payload: PayloadModel) -> PredictionInfo:
            _value = model.predict([payload.dict()])[0]
            return PredictionInfo(value=_value, model_version=version_info.version)

    app = VersionedFastAPI(
        app, version_format="{major}", prefix_format="/v{major}", enable_latest=True
    )
    return app
