import os
import sys
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    random_state: int = 42


class ModelTrainer:
    def __init__(self, config: Optional[ModelTrainerConfig] = None):
        self.model_trainer_config = config or ModelTrainerConfig()

    def _build_candidates(self) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
        candidates: dict[str, tuple[Any, dict[str, list[Any]]]] = {
            "catboost": (
                CatBoostRegressor(verbose=False, random_seed=self.model_trainer_config.random_state),
                {
                    "depth": [4, 6, 8],
                    "learning_rate": [0.03, 0.05, 0.1],
                    "iterations": [300, 500, 700],
                },
            ),
            "hist_gradient_boosting": (
                HistGradientBoostingRegressor(random_state=self.model_trainer_config.random_state),
                {
                    "learning_rate": [0.03, 0.05, 0.1],
                    "max_depth": [None, 6, 10],
                    "max_leaf_nodes": [31, 63, 127],
                },
            ),
            "extra_trees": (
                ExtraTreesRegressor(random_state=self.model_trainer_config.random_state, n_jobs=-1),
                {
                    "n_estimators": [200, 400, 600],
                    "max_depth": [None, 15, 25],
                    "min_samples_split": [2, 5, 10],
                },
            ),
            "random_forest": (
                RandomForestRegressor(random_state=self.model_trainer_config.random_state, n_jobs=-1),
                {
                    "n_estimators": [250, 500],
                    "max_depth": [None, 12, 20],
                    "min_samples_leaf": [1, 2, 4],
                },
            ),
        }

        if XGBRegressor is not None:
            candidates["xgboost"] = (
                XGBRegressor(
                    objective="reg:squarederror",
                    random_state=self.model_trainer_config.random_state,
                    n_estimators=500,
                    n_jobs=-1,
                ),
                {
                    "max_depth": [4, 6, 8],
                    "learning_rate": [0.03, 0.05, 0.1],
                    "subsample": [0.8, 0.9, 1.0],
                    "colsample_bytree": [0.8, 0.9, 1.0],
                },
            )

        return candidates

    def initiate_model_training(self, train_array: np.ndarray, test_array: np.ndarray) -> tuple[float, float, float]:
        try:
            logging.info("Starting model training")

            xtrain, ytrain, xtest, ytest = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            candidates = self._build_candidates()
            best_name = ""
            best_model = None
            best_score = float("-inf")

            for model_name, (estimator, search_space) in candidates.items():
                logging.info("Tuning model: %s", model_name)
                search = RandomizedSearchCV(
                    estimator=estimator,
                    param_distributions=search_space,
                    n_iter=8,
                    cv=3,
                    scoring="r2",
                    n_jobs=-1,
                    random_state=self.model_trainer_config.random_state,
                    verbose=0,
                )
                search.fit(xtrain, ytrain)

                tuned_model = search.best_estimator_
                predictions = tuned_model.predict(xtest)
                score = r2_score(ytest, predictions)

                logging.info(
                    "Model %s | best params=%s | test r2=%.4f",
                    model_name,
                    search.best_params_,
                    score,
                )

                if score > best_score:
                    best_score = score
                    best_name = model_name
                    best_model = tuned_model

            if best_model is None:
                raise RuntimeError("Training failed: no candidate model was selected")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            ytest_pred = best_model.predict(xtest)
            mae = mean_absolute_error(ytest, ytest_pred)
            rmse = float(np.sqrt(mean_squared_error(ytest, ytest_pred)))
            r2 = r2_score(ytest, ytest_pred)

            logging.info("Best model selected: %s", best_name)
            logging.info("Saved model to %s", self.model_trainer_config.trained_model_file_path)
            logging.info("Evaluation metrics | MAE=%.4f RMSE=%.4f R2=%.4f", mae, rmse, r2)

            return mae, rmse, r2

        except Exception as e:
            raise CustomException(e, sys)

    # Backward compatibility for legacy calls.
    def initate_model_training(self, train_array: np.ndarray, test_array: np.ndarray) -> tuple[float, float, float]:
        return self.initiate_model_training(train_array, test_array)
