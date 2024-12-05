import numpy as np
from pandas import DataFrame, Series
from scipy.stats import trim_mean
from sklearn.model_selection import KFold

from src.enums.accuracy_metric import AccuracyMetric
from src.models.model_wrapper import ModelWrapper
from src.pipelines.dt_pipeline import DTPipeline
from src.trainers.trainer import Trainer


class AccurateCrossTrainer(Trainer):

    def __init__(self, pipeline: DTPipeline, model_wrapper: ModelWrapper, metric: AccuracyMetric = AccuracyMetric.MAE,
                 grouping_columns: list[str] = None):
        super().__init__(pipeline, model_wrapper, metric=metric, grouping_columns=grouping_columns)

    def __cross_train(self, X: DataFrame, y: Series, train_index: int, val_index: int, iterations=None,
                      params=None) -> (int, int):
        """
        Trains a Model on the provided training data by splitting it into training and validation sets.
        The split function does not shuffle, and it's based on the provided indexes.
        If no rounds are provided, the model is trained using early stopping and will return the optimal number of
        iterations alongside the accuracy.
        :param X:
        :param y:
        :param train_index:
        :param val_index:
        :param iterations:
        :param params:
        :return:
        """
        # split train and validation data
        train_X, val_X = X.iloc[train_index], X.iloc[val_index]
        train_y, val_y = y.iloc[train_index], y.iloc[val_index]

        # if no rounds, train with early stopping
        if iterations is None:
            self.train_model(train_X, train_y, val_X, val_y, params=params)
        # else train normally
        else:
            self.train_model(train_X, train_y, iterations=iterations, params=params)

        # re-process val_X to obtain accuracy
        processed_val_X = self.pipeline.transform(val_X)

        # Predict and calculate accuracy
        predictions = self.get_predictions(processed_val_X)
        accuracy = self.calculate_accuracy(predictions, val_y)

        try:
            # number of boosting rounds used in the best model, accuracy
            return self.model_wrapper.get_best_iteration(), accuracy
        # if the model was trained without early stopping, return the provided training rounds
        except AttributeError:
            return iterations, accuracy

    def validate_model(self, X: DataFrame, y: Series, iterations=None, log_level=2, params=None) -> (float, int):
        """
        Trains 5 XGBoost regressors on the provided training data by cross-validation.
        Data is splitted into 5 folds, each model is trained on 4 folds and validated on 1 fold.
        The validation fold is always different, so we are basically training and validating over the entire dataset.
        MAE score and optimal boosting rounds of each model are then meaned to get overall values.
        If no rounds are provided, the models are trained using early stopping and will return the optimal number of
        boosting rounds alongside the MAE.

        :param X:
        :param y:
        :param iterations:
        :param log_level:
        :param params:
        :return:
        """
        # Initialize KFold
        kf = self.get_kfold_type()

        if self.grouping_columns is not None:
            groups = X[self.grouping_columns]
        else:
            groups = None

        self.evals = []

        # Placeholder for cross-validation MAE scores
        cv_scores = []
        best_rounds = []

        # Loop through each fold
        for train_index, val_index in kf.split(X, y, groups):
            best_iteration, mae = self.__cross_train(X, y, train_index, val_index, iterations=iterations, params=params)

            best_rounds.append(best_iteration or 0)
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
        if iterations is None and log_level > 0:
            print("Generating {} with optimal boosting rounds".format(self.metric.value))
            self.validate_model(X, y, optimal_boost_rounds, log_level=1, params=params)

        return mean_accuracy, optimal_boost_rounds
