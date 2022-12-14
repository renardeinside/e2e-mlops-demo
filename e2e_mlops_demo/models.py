"""
This module contains logical models, not the machine learning ones
"""
import datetime as dt
from typing import Dict, Any, Optional

import pandas as pd
from pydantic import BaseModel, validator

SearchSpace = Dict[str, Dict[str, Any]]


class FlexibleBaseModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True


class TrainData(FlexibleBaseModel):
    X: pd.DataFrame
    y: pd.Series


class TestData(FlexibleBaseModel):
    X: pd.DataFrame
    y: pd.Series


class SourceMetadata(BaseModel):
    version: int
    database: str
    table: str


class ModelData(FlexibleBaseModel):
    train: TrainData
    test: TestData
    source_metadata: SourceMetadata


class MlflowInfo(BaseModel):
    registry_uri: str
    tracking_uri: str


class PredictionInfo(BaseModel):
    value: int
    model_version: int
    predicted_at: Optional[dt.datetime] = None

    @validator("predicted_at", pre=True, always=True)
    def provide_dt(cls, _) -> dt.datetime:
        return dt.datetime.now()

    class Config:
        json_encoders = {dt.datetime: lambda v: v.isoformat()}
