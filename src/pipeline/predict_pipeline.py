import os
import sys
from dataclasses import dataclass

import pandas as pd

from src.exception import CustomException
from src.utils import load_object


@dataclass
class CustomData:
    carat: float
    depth: float
    table: float
    x: float
    y: float
    z: float
    cut: str
    color: str
    clarity: str

    def get_data_as_dataframe(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "carat": [self.carat],
                "depth": [self.depth],
                "table": [self.table],
                "x": [self.x],
                "y": [self.y],
                "z": [self.z],
                "cut": [self.cut],
                "color": [self.color],
                "clarity": [self.clarity],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)


class PredictPipeline:
    def __init__(
        self,
        model_path: str = os.path.join("artifacts", "model.pkl"),
        preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl"),
    ):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path

    def predict(self, features: pd.DataFrame):
        try:
            preprocessor = load_object(self.preprocessor_path)
            model = load_object(self.model_path)

            transformed_data = preprocessor.transform(features)
            predictions = model.predict(transformed_data)
            return predictions

        except Exception as e:
            raise CustomException(e, sys)
