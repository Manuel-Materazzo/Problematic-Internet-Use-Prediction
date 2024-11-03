import itertools

from pandas import DataFrame, Series

from src.hyperparameter_optimizers.hp_optimizer import HyperparameterOptimizer
from src.trainers.trainer import Trainer


class AccurateGridOptimizer(HyperparameterOptimizer):
    def __init__(self, trainer: Trainer):
        super().__init__(trainer)

    def tune(self, X: DataFrame, y: Series, final_lr: float) -> dict:
        """
        Calculates the best hyperparameters for the dataset by performing a grid search
        Trains a cross-validated model for each combination of hyperparameters, and picks the best based on MAE
        :param X:
        :param y:
        :param final_lr:
        :return:
        """
        # get optimal boost rounds
        optimal_br = self.get_optimal_boost_rounds(X, y)

        print("Step 1, searching for optimal max_depth and min_child_weight:")
        self.params.update(
            self.__do_grid_search(X, y, optimal_br, {
                'max_depth': range(3, 10),
                'min_child_weight': range(1, 6)
            })
        )

        print("Step 2, searching for optimal gamma:")
        self.params.update(
            self.__do_grid_search(X, y, optimal_br, {
                'gamma': [i / 10.0 for i in range(0, 5)]
            })
        )

        # Recalibrate boosting rounds
        optimal_br = self.get_optimal_boost_rounds(X, y)

        print("Step 3, searching for optimal subsample and colsample_bytree:")
        self.params.update(
            self.__do_grid_search(X, y, optimal_br, {
                'subsample': [i / 100.0 for i in range(60, 100, 5)],
                'colsample_bytree': [i / 100.0 for i in range(60, 100, 5)]
            })
        )

        print("Step 4, searching for optimal reg_alpha:")
        self.params.update(
            self.__do_grid_search(X, y, optimal_br, {
                'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
            })
        )

        # No worky, always return 0.1 despite best result lying elsewhere
        # print("Step 5, searching for optimal learning_rate:")
        # self.params.update(
        #     self.__do_grid_search(X, y, optimal_br, {
        #         'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
        #     })
        # )

        self.params['learning_rate'] = final_lr

        return self.params

    def __do_grid_search(self, X: DataFrame, y: Series, optimal_boosting_rounds: int, param_grid: dict,
                         log_level=1) -> dict:
        """
        Trains cross-validated model for each combination of the provided hyperparameters, and picks the best based on MAE
        :param pipeline:
        :param X:
        :param y:
        :param param_grid:
        :return:
        """
        # Generate all possible combinations of hyperparameters
        param_combinations = [dict(zip(param_grid, v)) for v in itertools.product(*param_grid.values())]

        best_params = None
        best_score = float('inf')
        results = []

        for params in param_combinations:

            full_params = self.params.copy()
            full_params.update(params)

            mae, _ = self.trainer.validate_model(X, y, log_level=0, rounds=optimal_boosting_rounds, **full_params)
            results.append((params, mae))

            if mae < best_score:
                best_score = mae
                best_params = params

        if log_level > 0:
            print("Best parameters found: ", best_params)
            print("Best MAE: {}".format(best_score))

        if log_level > 1:
            # Print all results
            for params, mae in results:
                print(f"Parameters: {params}, MAE: {mae}")

        return best_params
