from abc import ABC, abstractmethod

from pandas import DataFrame, Series

from src.trainers.trainer import Trainer


class HyperparameterOptimizer(ABC):

    def __init__(self, trainer: Trainer):
        self.trainer: Trainer = trainer
        self.params: dict = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_child_weight': 1,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 1,
            'n_jobs': -1,  # Use all available cores
        }

    @abstractmethod
    def tune(self, X: DataFrame, y: Series, final_lr: float) -> dict:
        """
        Does some computer magic to tune hyperparameters.
        :param X:
        :param y:
        :param final_lr:
        :return:
        """
        pass

    def get_optimal_boost_rounds(self, X: DataFrame, y: Series) -> int:
        """
        Gets the optimal boost rounds for the provided data and the current params
        :param X:
        :param y:
        :return:
        """
        _, optimal_boosting_rounds = self.trainer.validate_model(X, y, log_level=0, params=self.params)
        return optimal_boosting_rounds

    def space_to_params(self, space: dict) -> dict:
        return {
            'objective': 'reg:squarederror',
            'learning_rate': 0.03,
            'max_depth': int(space['max_depth']),
            'min_child_weight': int(space['min_child_weight']),
            'gamma': space['gamma'],
            'colsample_bytree': space['colsample_bytree'],
            'subsample': space['subsample'],
            'reg_alpha': space['reg_alpha'],
            'reg_lambda': space['reg_lambda'],
            'scale_pos_weight': 1,
            'n_jobs': -1,
        }