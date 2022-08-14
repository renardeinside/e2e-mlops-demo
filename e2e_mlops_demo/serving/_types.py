import datetime as dt

import pydantic
from mlflow.types import Schema
from pydantic import BaseModel

# please note that this type mapping is based on DataType from mlflow
_COL_TYPE_MAPPING = {
    "bool": bool,
    "double": float,
    "int32": int,
    "int64": int,
    "float32": float,
    "float64": float,
    "str": str,
    "bytes": bytes,
    "datetime64": dt.datetime,
}


def get_pydantic_model(schema: Schema, name: str) -> BaseModel:
    fields = {item["name"]: (_COL_TYPE_MAPPING[item["type"]], ...) for item in schema.to_dict()}
    return pydantic.create_model(name, **fields)
