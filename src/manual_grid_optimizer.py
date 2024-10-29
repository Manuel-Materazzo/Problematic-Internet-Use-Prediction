import itertools

from pandas import DataFrame, Series

from src.trainer import Trainer


class GridOptimizer:
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

    def __get_optimal_boost_rounds(self, X: DataFrame, y: Series) -> int:
        """
        Gets the optimal boost rounds for the provided data and the current params
        :param X:
        :param y:
        :return:
        """
        _, optimal_boosting_rounds = self.trainer.cross_validation(X, y, log_level=0, **self.params)
        return optimal_boosting_rounds

    def tune(self, X: DataFrame, y: Series, final_lr: float) -> dict:
        """
        Calculates the best hyperparameters for the dataset by performing a grid search
        GridSearch trains cross-validated model for each combination of hyperparameters, and picks the best based on MAE
        :param X:
        :param y:
        :param final_lr:
        :return:
        """
        # get optimal boost rounds
        optimal_br = self.__get_optimal_boost_rounds(X, y)

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
        optimal_br = self.__get_optimal_boost_rounds(X, y)

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

        self.params['learning_rate'] = final_lr

        return self.params

    def __do_grid_search(self, X: DataFrame, y: Series, optimal_boosting_rounds: int, param_grid: dict, log_level=1) -> dict:
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

            mae, _ = self.trainer.cross_validation(X, y, log_level=0, rounds=optimal_boosting_rounds, **full_params)
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
                print(f"Parameters: {params}, MAE: ±{mae:.0f}€")

        return best_params
