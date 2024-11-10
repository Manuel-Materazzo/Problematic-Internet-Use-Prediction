import numpy as np
from pandas import DataFrame, Series
from scipy.stats import trim_mean

from src.enums.accuracy_metric import AccuracyMetric
from src.pipelines.dt_pipeline import DTPipeline
from sklearn.model_selection import KFold

from src.trainers.accurate_cross_trainer import AccurateCrossTrainer
from src.trainers.trainer import Trainer


class CachedAccurateCrossTrainer(Trainer):
    """
    Wrapper of SimpleTrainer that takes X and Y at initialization time and caches kfold splits.
    """

    def __init__(self, pipeline: DTPipeline, X: DataFrame, y: Series, metric: AccuracyMetric = AccuracyMetric.MAE):
        super().__init__(pipeline, metric=metric)
        self.X = X
        self.y = y
        self.splits = self.__cache_splits()
        self.trainer = AccurateCrossTrainer(pipeline)

    def __cache_splits(self) -> list:
        """
        Splits X and y into 5 folds at initialization time and returns them as a list.
        :return:
        """
        kf = KFold(
            n_splits=5,
            shuffle=True,
            random_state=0
        )

        splits = []

        for train_index, val_index in kf.split(self.X):
            # split train and validation data
            train_X, val_X = self.X.iloc[train_index], self.X.iloc[val_index]
            train_y, val_y = self.y.iloc[train_index], self.y.iloc[val_index]

            splits.append([train_X, val_X, train_y, val_y])

        return splits

    def __cross_train(self, split, rounds=None, **xgb_params) -> (int, int):

        # if no rounds, train with early stopping
        if rounds is None:
            self.model = self.trainer.train_model(split[0], split[2], split[1], split[3], **xgb_params)
        # else train normally
        else:
            self.model = self.trainer.train_model(split[0], split[2], rounds=rounds, **xgb_params)

        # re-process val_X to obtain MAE
        processed_val_X = self.trainer.pipeline.transform(split[1])

        # Predict and calculate MAE
        predictions = self.model.predict(processed_val_X)
        accuracy = self.calculate_accuracy(predictions, split[3])

        try:
            # number of boosting rounds used in the best model, MAE
            return self.model.best_iteration, accuracy
        # if the model was trained without early stopping, return the provided training rounds
        except AttributeError:
            return rounds, accuracy

    def validate_model(self, X: DataFrame, y: Series, rounds=None, log_level=2, **xgb_params) -> (float, int):

        # Placeholder for cross-validation MAE scores
        cv_scores = []
        best_rounds = []

        self.evals = []

        # Loop through each fold
        for split in self.splits:
            best_iteration, mae = self.__cross_train(split, rounds=rounds, **xgb_params)

            best_rounds.append(best_iteration)
            cv_scores.append(mae)

        # extract evals
        self.evals = self.trainer.evals

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
