from pandas import DataFrame, Series
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from src.hyperparameter_optimizers.hp_optimizer import HyperparameterOptimizer
from src.trainers.trainer import Trainer


class HyperoptBayesianOptimizer(HyperparameterOptimizer):
    def __init__(self, trainer: Trainer):
        super().__init__(trainer)
        self.y = None
        self.X = None
        self.domain_space = {
            'max_depth': hp.quniform("max_depth", 3, 10, 1),
            'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
            'gamma': hp.uniform('gamma', 0, 0.5),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
            'subsample': hp.uniform('subsample', 0.5, 1),
            'reg_alpha': hp.uniform('reg_alpha', 0, 10),
            'reg_lambda': hp.uniform('reg_lambda', 0, 1),
        }

    def tune(self, X: DataFrame, y: Series, final_lr: float) -> dict:
        """
        Calculates the best hyperparameters for the dataset by performing a bayesian optimization
        Trains cross-validated model for each combination of hyperparameters, and picks the best based on MAE.
        :param X:
        :param y:
        :param final_lr:
        :return:
        """
        self.X = X
        self.y = y
        trials = Trials()

        best_hyperparams = fmin(fn=self.__objective,
                                space=self.domain_space,
                                algo=tpe.suggest,
                                max_evals=100,
                                trials=trials)

        self.params = self.space_to_params(best_hyperparams)

        self.params['learning_rate'] = final_lr

        return self.params

    def __objective(self, space):
        """
        Defines the objective function to be minimized.
        Trains a model with hyperparameters and returns the cross validated MAE.
        :return:
        """
        params = self.space_to_params(space)
        mae, _ = self.trainer.validate_model(self.X, self.y, log_level=0, params=params)
        return {'loss': mae, 'status': STATUS_OK}
