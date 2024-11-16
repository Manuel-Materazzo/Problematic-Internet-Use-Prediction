import numpy as np
from pandas import DataFrame, Series
from scipy.stats import trim_mean

from src.enums.accuracy_metric import AccuracyMetric
from src.models.model_wrapper import ModelWrapper
from src.pipelines.dt_pipeline import DTPipeline
from sklearn.model_selection import KFold

from src.trainers.accurate_cross_trainer import AccurateCrossTrainer
from src.trainers.trainer import Trainer


class CachedAccurateCrossTrainer(Trainer):
    """
    Wrapper of SimpleTrainer that takes X and Y at initialization time and caches kfold splits.
    """

    def __init__(self, pipeline: DTPipeline, model_wrapper: ModelWrapper, X: DataFrame, y: Series,
                 metric: AccuracyMetric = AccuracyMetric.MAE):
        super().__init__(pipeline, model_wrapper, metric=metric)
        self.X = X
        self.y = y
        self.splits = self.__cache_splits()
        self.trainer = AccurateCrossTrainer(pipeline, model_wrapper)

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

    def __cross_train(self, split, iterations=None, params=None) -> (int, int):

        # if no rounds, train with early stopping
        if iterations is None:
            self.model = self.trainer.train_model(split[0], split[2], split[1], split[3], params=params)
        # else train normally
        else:
            self.model = self.trainer.train_model(split[0], split[2], iterations=iterations, params=params)

        # re-process val_X to obtain MAE
        processed_val_X = self.trainer.pipeline.transform(split[1])

        # Predict and calculate MAE
        predictions = self.model.predict(processed_val_X)
        accuracy = self.calculate_accuracy(predictions, split[3])

        try:
            # number of boosting rounds used in the best model, MAE
            return self.model.get_best_iteration(), accuracy
        # if the model was trained without early stopping, return the provided training rounds
        except AttributeError:
            return iterations, accuracy

    def validate_model(self, X: DataFrame, y: Series, log_level=2, iterations=None, params=None) -> (float, int):

        # Placeholder for cross-validation MAE scores
        cv_scores = []
        best_rounds = []

        self.evals = []

        # Loop through each fold
        for split in self.splits:
            best_iteration, mae = self.__cross_train(split, iterations=iterations, params=params)

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
        if iterations is None and log_level > 0:
            print("Generating {} with optimal boosting rounds".format(self.metric.value))
            self.validate_model(X, y, iterations=optimal_boost_rounds, log_level=1, params=params)

        return mean_accuracy, optimal_boost_rounds
