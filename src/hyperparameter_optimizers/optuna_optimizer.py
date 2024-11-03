from pandas import DataFrame, Series
import optuna

from src.hyperparameter_optimizers.hp_optimizer import HyperparameterOptimizer
from src.trainers.trainer import Trainer


class OptunaOptimizer(HyperparameterOptimizer):
    def __init__(self, trainer: Trainer):
        super().__init__(trainer)
        self.y = None
        self.X = None

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

        study = optuna.create_study(direction='minimize')
        study.optimize(self.__objective, n_trials=100)
        self.params.update(study.best_params)

        self.params['learning_rate'] = final_lr

        return self.params

    def __objective(self, trial):
        """
        Defines the objective function to be minimized.
        Trains a model with hyperparameters and returns the cross validated MAE.
        :return:
        """
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.03,
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 6),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
            'subsample': trial.suggest_float('subsample', 0.5, 1),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'scale_pos_weight': 1,
            'n_jobs': -1,
        }

        mae, _ = self.trainer.validate_model(self.X, self.y, log_level=0, **params)
        return mae
