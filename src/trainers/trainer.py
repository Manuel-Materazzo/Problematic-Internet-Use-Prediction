import math

import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from src.enums.accuracy_metric import AccuracyMetric
from src.pipelines.dt_pipeline import DTPipeline
from abc import ABC, abstractmethod


def show_confusion_matrix(real_values: Series, predictions):
    print(real_values.values)
    print(predictions)
    cm = confusion_matrix(real_values, predictions)
    sns.heatmap(cm, annot=True, fmt='', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Real Data')
    plt.show()


class Trainer(ABC):
    def __init__(self, pipeline: DTPipeline, metric: AccuracyMetric = AccuracyMetric.MAE):
        self.pipeline: DTPipeline = pipeline
        self.metric: AccuracyMetric = metric
        self.model = None

    def get_pipeline(self) -> DTPipeline:
        """
        Returns the pipeline that gets used for training.
        :return:
        """
        return self.pipeline

    def show_feature_importance(self, X: DataFrame):
        if self.model is None:
            print("No model has been fitted")
            return

        features = list(X.columns)  # Extract original features
        importances = self.model.feature_importances_

        feature_importances = sorted(zip(importances, features), reverse=False)
        sorted_importances, sorted_features = zip(*feature_importances)

        print(sorted_importances, sorted_features)

        # TODO: show feature names on the plot
        plt.figure(figsize=(12, 6))
        plt.title('Relative Feature Importance')
        plt.barh(range(len(sorted_importances)), sorted_importances, color='b', align='center')
        plt.yticks(range(len(sorted_features)), sorted_features)
        plt.show()

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

    @abstractmethod
    def validate_model(self, X: DataFrame, y: Series, log_level=1, rounds=None, **xgb_params) -> (float, int):
        pass

    def calculate_accuracy(self, predictions: Series, real_values: Series) -> float:
        """
        Calculates the accuracy of the provided predictions, using the metric specified when creating the trainer.
        :param predictions:
        :param real_values:
        :return:
        """
        match self.metric:
            case AccuracyMetric.MAE:
                return mean_absolute_error(real_values, predictions)
            case AccuracyMetric.MSE:
                return mean_squared_error(real_values, predictions)
            case AccuracyMetric.RMSE:
                return math.sqrt(mean_squared_error(real_values, predictions))
