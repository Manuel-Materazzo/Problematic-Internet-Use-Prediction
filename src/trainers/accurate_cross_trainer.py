import numpy as np
from pandas import DataFrame, Series
from scipy.stats import trim_mean
from sklearn.model_selection import KFold

from src.enums.accuracy_metric import AccuracyMetric
from src.pipelines.dt_pipeline import DTPipeline
from src.trainers.trainer import Trainer


class AccurateCrossTrainer(Trainer):

    def __init__(self, pipeline: DTPipeline, metric: AccuracyMetric = AccuracyMetric.MAE):
        super().__init__(pipeline, metric=metric)

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
        accuracy = self.calculate_accuracy(predictions, val_y)

        try:
            # number of boosting rounds used in the best model, MAE
            return self.model.best_iteration, accuracy
        # if the model was trained without early stopping, return the provided training rounds
        except AttributeError:
            return rounds, accuracy

    def validate_model(self, X: DataFrame, y: Series, rounds=None, log_level=2, **xgb_params) -> (float, int):
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

        self.evals = []

        # Placeholder for cross-validation MAE scores
        cv_scores = []
        best_rounds = []

        # Loop through each fold
        for train_index, val_index in kf.split(X):
            best_iteration, mae = self.__cross_train(X, y, train_index, val_index, rounds=rounds, **xgb_params)

            best_rounds.append(best_iteration)
            cv_scores.append(mae)

        # Calculate the mean accuracy from cross-validation
        mean_accuracy = np.mean(cv_scores)
        # Calculate optimal boosting rounds
        optimal_boost_rounds = int(np.mean(best_rounds))
        pruned_optimal_boost_rounds = int(trim_mean(best_rounds, proportiontocut=0.1))  # trim extreme values

        if log_level > 0:
            print("Cross-Validation {}: {}".format(self.metric.value, mean_accuracy))
            if log_level > 1:
                print(cv_scores)
            print("Optimal Boosting Rounds: ", optimal_boost_rounds)
            if log_level > 1:
                print("Pruned Optimal Boosting Rounds: ", pruned_optimal_boost_rounds)
                print(best_rounds)

        # Cross validate model with the optimal boosting round, to check on MAE discrepancies
        if rounds is None and log_level > 0:
            print("Generating {} with optimal boosting rounds".format(self.metric.value))
            self.validate_model(X, y, optimal_boost_rounds, log_level=1, **xgb_params)

        return mean_accuracy, optimal_boost_rounds
