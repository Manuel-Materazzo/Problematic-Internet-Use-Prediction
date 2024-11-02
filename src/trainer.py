import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from scipy.stats import trim_mean
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, confusion_matrix

from src.pipelines.dt_pipeline import DTPipeline


def show_confusion_matrix(real_values: Series, predictions):
    print(real_values.values)
    print(predictions)
    cm = confusion_matrix(real_values, predictions)
    sns.heatmap(cm, annot=True, fmt='', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Real Data')
    plt.show()


class Trainer:

    def __init__(self, pipeline: DTPipeline):
        self.pipeline: DTPipeline = pipeline
        self.model = None

    def get_pipeline(self) -> DTPipeline:
        """
        Returns the pipeline that gets used for training.
        :return:
        """
        return self.pipeline

    def train_model(self, train_X: DataFrame, train_y: Series, val_X: DataFrame = None, val_y: Series = None,
                    rounds=1000, **xgb_params) -> XGBRegressor:
        """
        Trains a XGBoost regressor on the provided training data.
        When validation data is provided, the model is trained with early stopping.
        :param train_X:
        :param train_y:
        :param val_X:
        :param val_y:
        :param rounds:
        :param xgb_params:
        :return:
        """
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
        """
        Trains a XGBoost regressor on the provided training data by splitting it into training and validation sets.
        The split function does not shuffle, and it's based on the provided indexes.
        If no rounds are provided, the model is trained using early stopping and will return the optimal number of
        boosting rounds alongside the MAE.
        :param X:
        :param y:
        :param train_index:
        :param val_index:
        :param rounds:
        :param xgb_params:
        :return:
        """
        # split train and validation data
        train_X, val_X = X.iloc[train_index], X.iloc[val_index]
        train_y, val_y = y.iloc[train_index], y.iloc[val_index]

        # if no rounds, train with early stopping
        if rounds is None:
            self.model = self.train_model(train_X, train_y, val_X, val_y, **xgb_params)
        # else train normally
        else:
            self.model = self.train_model(train_X, train_y, rounds=rounds, **xgb_params)

        # re-process val_X to obtain MAE
        processed_val_X = self.pipeline.transform(val_X)

        # Predict and calculate MAE
        predictions = self.model.predict(processed_val_X)
        mae = mean_absolute_error(predictions, val_y)

        try:
            # number of boosting rounds used in the best model, MAE
            return self.model.best_iteration, mae
        # if the model was trained without early stopping, return the provided training rounds
        except AttributeError:
            return rounds, mae

    def cross_validation(self, X: DataFrame, y: Series, rounds=None, log_level=2, **xgb_params) -> (float, int):
        """
        Trains 5 XGBoost regressors on the provided training data by cross-validation.
        Data is splitted into 5 folds, each model is trained on 4 folds and validated on 1 fold.
        The validation fold is always different, so we are basically training and validating over the entire dataset.
        MAE score and optimal boosting rounds of each model are then meaned to get overall values.
        If no rounds are provided, the models are trained using early stopping and will return the optimal number of
        boosting rounds alongside the MAE.

        :param X:
        :param y:
        :param rounds:
        :param log_level:
        :param xgb_params:
        :return:
        """
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
            print("Cross-Validation MAE: {}".format(mean_mae_cv))
            if log_level > 1:
                print(cv_scores)
            print("Optimal Boosting Rounds: ", optimal_boost_rounds)
            if log_level > 1:
                print("Pruned Optimal Boosting Rounds: ", pruned_optimal_boost_rounds)
                print(best_rounds)

        # Cross validate model with the optimal boosting round, to check on MAE discrepancies
        if rounds is None and log_level > 0:
            print("Generating MAE with optimal boosting rounds")
            self.cross_validation(X, y, optimal_boost_rounds, log_level=1, **xgb_params)

        return mean_mae_cv, optimal_boost_rounds

    def simple_cross_validation(self, X: DataFrame, y: Series, rounds=1000, log_level=1, **xgb_params) -> (float, int):
        """
        Trains 5 XGBoost regressors on the provided training data by cross-validation.
        This method uses default xgb.cv strategy for cross-validation.
        X is preprocessed using fit_transform on the pipeline, this will probably cause
        "Train-Test Contamination Data Leakage" and provide a MAE estimate with lower accuracy.
        :param X:
        :param y:
        :param rounds:
        :param xgb_params:
        :return:
        """

        print("WARNING: using simple_cross_validation can cause train data leakage, prefer cross_validation or "
              "classic_validation instead")

        processed_X = self.pipeline.fit_transform(X)

        dtrain = xgb.DMatrix(processed_X, label=y)

        cv_results = xgb.cv(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=rounds,
            nfold=5,
            metrics='mae',
            early_stopping_rounds=5,
            seed=0,
            as_pandas=True)

        # Extract the mean of the MAE from cross-validation results
        mae_cv = cv_results[
            'test-mae-mean'].min()  # optimal point (iteration) where the model achieved its best performance
        best_round = cv_results[
            'test-mae-mean'].idxmin()  # if you train the model again, same seed, no early stopping, you can put this index as num_boost_round to get same result

        if log_level > 0:
            print("#{} Cross-Validation MAE: {}".format(best_round, mae_cv))

        return mae_cv, best_round

    def classic_validation(self, X: DataFrame, y: Series, log_level=1, **xgb_params) -> (float, int):
        """
        Trains a XGBoost regressor on the provided training data by splitting it into training and validation sets.
        This uses early stopping and will return the optimal number of boosting rounds alongside the MAE.
        :param log_level:
        :param X:
        :param y:
        :param xgb_params:
        :return:
        """
        # Split into validation and training data
        train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
        # Get trained model
        self.model = self.train_model(train_X, train_y, val_X, val_y, **xgb_params)
        # preprocess validation data
        processed_val_X = pd.DataFrame(self.pipeline.transform(val_X))
        # Predict validation y using validation X
        predictions = self.model.predict(processed_val_X)
        # Calculate MAE
        mae = mean_absolute_error(predictions, val_y)
        # TODO: ouptut confusion matrix with XGBClassifier
        # show_confusion_matrix(val_y, predictions)
        if log_level > 0:
            print("Validation MAE : {}".format(mae))

        return mae, self.model.best_iteration

    def show_feature_importance(self, X: DataFrame):
        if self.model is None:
            print("No model has been fitted")
            return

        features = list(X.columns)  # Extract original features
        importances = self.model.feature_importances_

        feature_importances = sorted(zip(importances, features), reverse=False)
        sorted_importances, sorted_features = zip(*feature_importances)

        print(sorted_importances, sorted_features)

        #TODO: show feature names on the plot
        plt.figure(figsize=(12, 6))
        plt.title('Relative Feature Importance')
        plt.barh(range(len(sorted_importances)), sorted_importances, color='b', align='center')
        plt.yticks(range(len(sorted_features)), sorted_features)
        plt.show()
