import pandas as pd
import numpy as np
import xgboost as xgb
from pandas import DataFrame, Series
from scipy.stats import trim_mean
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

from src.data_trasformation_pipeline import DataTrasformationPipeline


class Trainer:

    def __init__(self, pipeline: DataTrasformationPipeline):
        self.pipeline: DataTrasformationPipeline = pipeline

    def get_pipeline(self):
        return self.pipeline

    def train_model(self, train_X: DataFrame, train_y: Series, val_X: DataFrame = None, val_y: Series = None,
                    rounds=1000, **xgb_params) -> XGBRegressor:
        processed_train_X = self.pipeline.fit_transform(train_X)

        # if we have validation sets, train with early stopping rounds
        if val_y is not None:
            processed_val_X = self.pipeline.transform(val_X)
            model = XGBRegressor(
                random_state=0,
                n_estimators=rounds,
                early_stopping_rounds=5,
                **xgb_params
            )
            model.fit(processed_train_X, train_y, eval_set=[(processed_val_X, val_y)], verbose=False)
        # else train with all the data
        else:
            model = XGBRegressor(
                random_state=0,
                n_estimators=rounds,
                **xgb_params
            )
            model.fit(processed_train_X, train_y)

        return model

    def __cross_train(self, X: DataFrame, y: Series, train_index: int, val_index: int, rounds=None,
                      **xgb_params) -> (int, int):
        # split train and validation data
        train_X, val_X = X.iloc[train_index], X.iloc[val_index]
        train_y, val_y = y.iloc[train_index], y.iloc[val_index]

        # if no rounds, train with early stopping
        if rounds is None:
            model = self.train_model(train_X, train_y, val_X, val_y, **xgb_params)
        # else train normally
        else:
            model = self.train_model(train_X, train_y, rounds=rounds, **xgb_params)

        # re-process val_X to obtain MAE
        processed_val_X = self.pipeline.transform(val_X)

        # Predict and calculate MAE
        predictions = model.predict(processed_val_X)
        mae = mean_absolute_error(predictions, val_y)

        try:
            # number of boosting rounds used in the best model, MAE
            return model.best_iteration, mae
        # if the model was trained without early stopping, return the provided training round
        except AttributeError:
            return rounds, mae

    def cross_validation(self, X: DataFrame, y: Series, rounds=None, log_level=2, **xgb_params) -> (float, int):
        # Initialize KFold
        kf = KFold(
            n_splits=5,  # dataset divided into 5 folds, 4 for training and 1 for validation
            shuffle=True,
            random_state=0
        )

        # Placeholder for cross-validation MAE scores
        cv_scores = []
        best_rounds = []

        # Loop through each fold
        for train_index, val_index in kf.split(X):
            best_iteration, mae = self.__cross_train(X, y, train_index, val_index, rounds=rounds, **xgb_params)

            best_rounds.append(best_iteration)
            cv_scores.append(mae)

        # Calculate the mean MAE from cross-validation
        mean_mae_cv = np.mean(cv_scores)
        # Calculate optimal boosting rounds
        optimal_boost_rounds = int(np.mean(best_rounds))
        pruned_optimal_boost_rounds = int(trim_mean(best_rounds, proportiontocut=0.1))  # trim extreme values

        if log_level > 0:
            print("Cross-Validation MAE: ±{:,.0f}€".format(mean_mae_cv))
            if log_level > 1:
                print(cv_scores)
            print("Optimal Boosting Rounds: ", optimal_boost_rounds)
            if log_level > 1:
                print("Pruned Optimal Boosting Rounds: ", pruned_optimal_boost_rounds)
                print(best_rounds)

        return optimal_boost_rounds


    def classic_validation(self, X, y, **xgb_params):
        # Split into validation and training data
        train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
        # Get trained model
        model = self.train_model(train_X, train_y, val_X, val_y, **xgb_params)
        # preprocess validation data
        processed_val_X = pd.DataFrame(self.pipeline.transform(val_X))
        # Predict validation y using validation X
        predictions = model.predict(processed_val_X)
        # Calculate MAE
        mae = mean_absolute_error(predictions, val_y)
        print("Validation MAE : ±{:,.0f}€".format(mae))
