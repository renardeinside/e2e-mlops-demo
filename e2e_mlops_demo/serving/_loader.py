import os
import tempfile
from typing import List

import mlflow.pyfunc
from mlflow.entities.model_registry import ModelVersion
from mlflow.pyfunc import PyFuncModel
from mlflow.tracking import MlflowClient


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
            if req_v not in [int(v.version) for v in all_versions]:
                raise Exception(
                    f"Requested model version {req_v} doesn't exist in registry"
                )
            else:
                _v = [v for v in all_versions if int(v.version) == req_v][0]
                prepared_versions.append(_v)
        return prepared_versions


def load_model(model_name: str, model_version: ModelVersion) -> PyFuncModel:
    full_model_uri = f"models:/{model_name}/{model_version.version}"
    with tempfile.TemporaryDirectory() as temp_dir:
        return mlflow.pyfunc.load_model(full_model_uri, temp_dir + "/model")
