import pandas as pd
from pandas import DataFrame

from src.models.model_wrapper import ModelWrapper
from xgboost import XGBRegressor


class XGBRegressorWrapper(ModelWrapper):

    def __init__(self):
        super().__init__()

    def get_base_model(self,params):
        return XGBRegressor(
            **params
        )

    def fit(self, X, y, iterations, params=None):
        params = params.copy() or {}
        params.update({
            'random_state': 0,
            'n_estimators': iterations,
        })

        self.model: XGBRegressor = XGBRegressor(
            **params
        )

        self.model.fit(X, y)

    def train_until_optimal(self, train_X, validation_X, train_y, validation_y, params=None):
        params = params.copy() or {}
        params.update({
            'random_state': 0,
            'n_estimators': 2000,
            'early_stopping_rounds': 5,
        })
        self.model: XGBRegressor = XGBRegressor(
            **params
        )
        self.model.fit(train_X, train_y, eval_set=[(validation_X, validation_y)], verbose=False)

    def predict(self, X) -> any:
        return self.model.predict(X)

    def get_best_iteration(self) -> int:
        return self.model.best_iteration

    def get_loss(self) -> dict[str, dict[str, list[float]]]:
        if self.model is None:
            print("ERROR: No model has been fitted")
            return {}

        return self.model.evals_result()

    def get_feature_importance(self, features) -> DataFrame:
        if self.model is None:
            print("ERROR: No model has been fitted")
            return pd.DataFrame()

        importances = self.model.feature_importances_

        # sort and merge importances and column names into a dataframe
        feature_importances = sorted(zip(importances, features), reverse=True)
        sorted_importances, sorted_features = zip(*feature_importances)
        return pd.DataFrame({'feats': sorted_features[:50], 'importance': sorted_importances[:50]})
