import os
from typing import Optional

from fastapi import FastAPI, APIRouter
from mlflow.pyfunc import PyFuncModel

from e2e_mlops_demo import __version__
from e2e_mlops_demo.models import PredictionInfo
from e2e_mlops_demo.serving._loader import get_model_versions, load_model
from e2e_mlops_demo.serving._types import get_pydantic_model


def prepare_router(
    model: PyFuncModel, version: int, version_name: Optional[str] = None
) -> APIRouter:
    if not version_name:
        version_name = f"v{version}"

    router = APIRouter(prefix=f"/{version_name}", tags=[version_name])

    PayloadModel = get_pydantic_model(
        model.metadata.get_input_schema(), f"Payload{version_name.capitalize()}"
    )

    @router.post(
        "/invocations",
        response_model=PredictionInfo,
        summary="predictions for the credit card transaction classification",
        description="""Returns predictions for the credit card transaction classification. 
            Empty values (e.g. `null`) are not allowed and won't pass the validation, resulting in status code `422`.
            """,
    )
    def invoke(payload: PayloadModel) -> PredictionInfo:
        _value = model.predict([payload.dict()])[0]
        return PredictionInfo(value=_value, model_version=version)

    return router


def get_app(model_name: Optional[str] = None) -> FastAPI:
    if not model_name:
        if "MODEL_NAME" not in os.environ:
            raise Exception("Please provide model name to serve")
        model_name = os.environ["MODEL_NAME"]

    app = FastAPI(
        title="Credit Card Transactions Classifier 🚀",
        version=__version__,
        description="Please check the relevant model version API for the schema description.",
    )
    versions = get_model_versions(model_name)

    loaded_models = {int(v.version): load_model(model_name, v) for v in versions}

    for _version, model in loaded_models.items():
        router = prepare_router(model, _version)
        app.include_router(router)

    _latest_version = max(loaded_models)
    _latest_model = loaded_models.get(_latest_version)
    latest_router = prepare_router(
        _latest_model, _latest_version, version_name="latest"
    )
    app.include_router(latest_router)

    return app
