from abc import ABC, abstractmethod

from pandas import DataFrame, Series

from src.models.model_wrapper import ModelWrapper
from src.trainers.trainer import Trainer


class HyperparameterOptimizer(ABC):

    def __init__(self, trainer: Trainer, model_wrapper: ModelWrapper):
        self.trainer: Trainer = trainer
        self.model_wrapper: ModelWrapper = model_wrapper
        self.params: dict = model_wrapper.get_starter_params()

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
