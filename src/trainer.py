import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import trim_mean
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


class Trainer:

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def train_model(self, train_X, train_y, val_X=None, val_y=None, rounds=1000, **xgb_params):
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

    def __cross_train(self, X, y, train_index, val_index, **xgb_params):
        # split train and validation data
        train_X, val_X = X.iloc[train_index], X.iloc[val_index]
        train_y, val_y = y.iloc[train_index], y.iloc[val_index]

        model = self.train_model(train_X, train_y, val_X, val_y, **xgb_params)

        # re-process val_X to obtain MAE
        processed_val_X = self.pipeline.transform(val_X)

        # Predict and calculate MAE
        predictions = model.predict(processed_val_X)
        mae = mean_absolute_error(predictions, val_y)

        # number of boosting rounds used in the best model, MAE
        return model.best_iteration, mae

    def cross_validation(self, X, y, **xgb_params):
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
            best_iteration, mae = self.__cross_train(X, y, train_index, val_index, **xgb_params)

            best_rounds.append(best_iteration)
            cv_scores.append(mae)

        # Calculate the mean MAE from cross-validation
        mean_mae_cv = np.mean(cv_scores)
        # Calculate optimal boosting rounds
        optimal_boost_rounds = int(np.mean(best_rounds))
        pruned_optimal_boost_rounds = int(trim_mean(best_rounds, proportiontocut=0.1))  # trim extreme values

        print("Cross-Validation MAE: ±{:,.0f}€".format(mean_mae_cv))
        print(cv_scores)
        print("Optimal Boosting Rounds: ", optimal_boost_rounds)
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
