from pandas import DataFrame, Series
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.hyperparameter_optimizers.hp_optimizer import HyperparameterOptimizer
from src.trainer import Trainer


class BalancedGridOptimizer(HyperparameterOptimizer):
    def __init__(self, trainer: Trainer):
        super().__init__(trainer)

    def __get_full_pipeline(self, optimal_boosting_rounds: int) -> Pipeline:
        """
        Gets a pipeline that includes model training
        :param optimal_boosting_rounds:
        :return:
        """
        return self.trainer.get_pipeline().get_pipeline_with_training(XGBRegressor(
            random_state=0,
            n_estimators=optimal_boosting_rounds,
            **self.params
        ))

    def tune(self, X: DataFrame, y: Series, final_lr: float, log_level=0) -> dict:
        """
        Calculates the best hyperparameters for the dataset by performing a grid search
        Trains a cross-validated model for each combination of hyperparameters, and picks the best based on MAE
        :param log_level:
        :param X:
        :param y:
        :param final_lr:
        :return:
        """
        # get optimal boost rounds
        optimal_br = self.get_optimal_boost_rounds(X, y)

        # using model__ notation to add support for the model training pipeline
        if log_level > 0:
            print("Step 1, searching for optimal max_depth and min_child_weight:")
        optimal_params = self.__do_grid_search(self.__get_full_pipeline(optimal_br), X, y, {
            'model__max_depth': range(3, 10),
            'model__min_child_weight': range(1, 6)
        }, log_level)
        self.params['max_depth'] = optimal_params['model__max_depth']
        self.params['min_child_weight'] = optimal_params['model__min_child_weight']

        if log_level > 0:
            print("Step 2, searching for optimal gamma:")
        optimal_params = self.__do_grid_search(self.__get_full_pipeline(optimal_br), X, y, {
            'model__gamma': [i / 10.0 for i in range(0, 5)]
        }, log_level)
        self.params['gamma'] = optimal_params['model__gamma']

        # Recalibrate boosting rounds
        optimal_br = self.get_optimal_boost_rounds(X, y)

        if log_level > 0:
            print("Step 3, searching for optimal subsample and colsample_bytree:")
        optimal_params = self.__do_grid_search(self.__get_full_pipeline(optimal_br), X, y, {
            'model__subsample': [i / 100.0 for i in range(60, 100, 5)],
            'model__colsample_bytree': [i / 100.0 for i in range(60, 100, 5)]
        }, log_level)
        self.params['subsample'] = optimal_params['model__subsample']
        self.params['colsample_bytree'] = optimal_params['model__colsample_bytree']

        if log_level > 0:
            print("Step 4, searching for optimal reg_alpha:")
        optimal_params = self.__do_grid_search(self.__get_full_pipeline(optimal_br), X, y, {
            'model__reg_alpha': [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
        }, log_level)
        self.params['reg_alpha'] = optimal_params['model__reg_alpha']

        # No worky, always return 0.1 despite best result lying elsewhere
        # if log_level > 0:
        #   print("Step 5, searching for optimal learning_rate:")
        # optimal_params = self.__do_grid_search(self.__get_full_pipeline(optimal_br), X, y, {
        #     'model__learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, ]
        # }, log_level)
        # self.params['learning_rate'] = optimal_params['model__learning_rate']

        self.params['learning_rate'] = final_lr

        return self.params

    def __do_grid_search(self, pipeline: Pipeline, X: DataFrame, y: Series, param_grid: dict, log_level=1) -> dict:
        """
        Trains cross-validated model for each combination of the provided hyperparameters, and picks the best based on MAE
        :param pipeline:
        :param X:
        :param y:
        :param param_grid:
        :return:
        """
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error', verbose=1, n_jobs=-1)
        grid_search.fit(X, y)

        if log_level > 0:
            print("Best parameters found: ", grid_search.best_params_)
            print("Best MAE: ", -grid_search.best_score_)

        if log_level > 1:
            # Print all parameters and corresponding MAE
            results = grid_search.cv_results_
            for i in range(len(results['params'])):
                print(f"Parameters: {results['params'][i]}")
                print(f"Mean Absolute Error (MAE): {abs(results['mean_test_score'][i])}")
                print()

        return grid_search.best_params_
