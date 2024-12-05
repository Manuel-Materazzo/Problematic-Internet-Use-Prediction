import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split

from src.enums.accuracy_metric import AccuracyMetric
from src.models.model_wrapper import ModelWrapper
from src.pipelines.dt_pipeline import DTPipeline
from src.trainers.trainer import Trainer


class SimpleTrainer(Trainer):

    def __init__(self, pipeline: DTPipeline, model_wrapper: ModelWrapper, metric: AccuracyMetric = AccuracyMetric.MAE,
                 grouping_columns: list[str] = None):
        super().__init__(pipeline, model_wrapper, metric=metric, grouping_columns=grouping_columns)

    def validate_model(self, X: DataFrame, y: Series, log_level=1, iterations=None, params=None) -> (float, int):
        """
        Trains a Model on the provided training data by splitting it into training and validation sets.
        This uses early stopping and will return the optimal number of iterations alongside the accuracy.
        :param iterations: ignored
        :param log_level:
        :param X:
        :param y:
        :param params:
        :return:
        """
        # Split into validation and training data
        train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
        # Get trained model
        self.evals = []
        self.train_model(train_X, train_y, val_X, val_y, params=params)
        # preprocess validation data
        processed_val_X = pd.DataFrame(self.pipeline.transform(val_X))
        # Predict validation y using validation X
        predictions = self.get_predictions(processed_val_X)
        # Calculate accuracy
        accuracy = self.calculate_accuracy(predictions, val_y)
        if log_level > 0:
            print("Validation {}: {}".format(self.metric.value, accuracy))

        return accuracy, self.model_wrapper.get_best_iteration()
