import pandas as pd
from pandas import DataFrame

from src.models.model_wrapper import ModelWrapper
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance


class HGBRegressorWrapper(ModelWrapper):

    def __init__(self):
        super().__init__()
        self.importances = None

    def get_base_model(self, params):
        return HistGradientBoostingRegressor(
            **params
        )

    def fit(self, X, y, iterations, params=None):
        self.train_until_optimal(X, None, y, None, params=params)
        self.importances = permutation_importance(self.model, X, y, n_repeats=10, random_state=0)

    def train_until_optimal(self, train_X, validation_X, train_y, validation_y, params=None):
        params = params.copy() or {}
        params.update({
            'random_state': 0,
            'max_iter': 2000,
            'early_stopping': True,
            'n_iter_no_change': 10
        })

        self.model: HistGradientBoostingRegressor = HistGradientBoostingRegressor(
            **params
        )
        # HistGradientBoostingRegressor has built in cross validation set extraction,
        # and won't need validation sets for early stopping.
        # Avoid merging in validation_X and validation_y, as it will cause Train data leakage when cross validating.
        self.model.fit(train_X, train_y)

    def predict(self, X) -> any:
        return self.model.predict(X)

    def get_best_iteration(self) -> int:
        return self.model.n_iter_

    def get_loss(self) -> dict[str, dict[str, list[float]]]:
        if self.model is None:
            print("ERROR: No model has been fitted")
            return {}

        return {
            'validation_0': {
                'rmse': abs(self.model.validation_score_) / 10000
            }
        }

    def get_feature_importance(self, features) -> DataFrame:
        if self.importances is None:
            print("ERROR: No model has been fitted")
            return pd.DataFrame()

        # sort and merge importances and column names into a dataframe
        feature_importances = sorted(zip(self.importances.importances_mean, features), reverse=True)
        sorted_importances, sorted_features = zip(*feature_importances)
        return pd.DataFrame({'feats': sorted_features[:50], 'importance': sorted_importances[:50]})
