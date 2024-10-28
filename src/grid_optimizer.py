from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor


class GridOptimizer:
    def __init__(self, trainer):
        self.trainer = trainer
        self.params = {
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

    def __get_full_pipeline(self, optimal_boosting_rounds):
        # get a pipeline that includes model training
        return self.trainer.get_pipeline().get_pipeline_with_training(XGBRegressor(
            random_state=0,
            n_estimators=optimal_boosting_rounds,
            **self.params
        ))

    def __get_optimal_boost_rounds(self, X, y):
        _, optimal_boosting_rounds = self.trainer.cross_validation(X, y, log_level=0, **self.params)
        return optimal_boosting_rounds

    def tune(self, X, y, final_lr):
        # get optimal boost rounds
        optimal_br = self.__get_optimal_boost_rounds(X, y)

        # using model__ notation to add support for the model training pipeline
        print("Step 1, searching for optimal max_depth and min_child_weight:")
        optimal_params = self.__do_grid_search(self.__get_full_pipeline(optimal_br), X, y, {
            'model__max_depth': range(3, 10),
            'model__min_child_weight': range(1, 6)
        })
        self.params['max_depth'] = optimal_params['model__max_depth']
        self.params['min_child_weight'] = optimal_params['model__min_child_weight']

        print("Step 2, searching for optimal gamma:")
        optimal_params = self.__do_grid_search(self.__get_full_pipeline(optimal_br), X, y, {
            'model__gamma': [i / 10.0 for i in range(0, 5)]
        })
        self.params['gamma'] = optimal_params['model__gamma']

        # Recalibrate boosting rounds
        optimal_br = self.__get_optimal_boost_rounds(X, y)

        print("Step 3, searching for optimal subsample and colsample_bytree:")
        optimal_params = self.__do_grid_search(self.__get_full_pipeline(optimal_br), X, y, {
            'model__subsample': [i / 100.0 for i in range(60, 100, 5)],
            'model__colsample_bytree': [i / 100.0 for i in range(60, 100, 5)]
        })
        self.params['subsample'] = optimal_params['model__subsample']
        self.params['colsample_bytree'] = optimal_params['model__colsample_bytree']


        print("Step 4, searching for optimal reg_alpha:")
        optimal_params = self.__do_grid_search(self.__get_full_pipeline(optimal_br), X, y, {
            'model__reg_alpha': [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
        })
        self.params['reg_alpha'] = optimal_params['model__reg_alpha']

        self.params['learning_rate'] = final_lr

        return self.params

    def __do_grid_search(self, pipeline, X, y, param_grid):
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error', verbose=1,
                                   n_jobs=-1)
        grid_search.fit(X, y)
        print("Best parameters found: ", grid_search.best_params_)
        print("Best MAE: ", -grid_search.best_score_)
        print()

        # Print all parameters and corresponding MAE
        # results = grid_search.cv_results_
        # for i in range(len(results['params'])):
        #     print(f"Parameters: {results['params'][i]}")
        #     print(f"Mean Absolute Error (MAE): {abs(results['mean_test_score'][i])}")
        #     print()

        return grid_search.best_params_
