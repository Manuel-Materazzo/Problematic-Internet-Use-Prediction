from abc import ABC, abstractmethod

from pandas import DataFrame, Series

from src.trainer import Trainer


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

    def __get_optimal_boost_rounds(self, X: DataFrame, y: Series) -> int:
        """
        Gets the optimal boost rounds for the provided data and the current params
        :param X:
        :param y:
        :return:
        """
        _, optimal_boosting_rounds = self.trainer.cross_validation(X, y, log_level=0, **self.params)
        return optimal_boosting_rounds
