from typing import Dict, Any

import pandas as pd
from pydantic import BaseModel

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


class ModelData(FlexibleBaseModel):
    train: TrainData
    test: TestData
