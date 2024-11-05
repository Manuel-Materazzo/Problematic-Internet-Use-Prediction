import math
import pickle

import pandas as pd
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


def save_model(model: XGBRegressor):
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)


def load_model() -> XGBRegressor:
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)


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

        # Apply the same trasformations as the training process
        processed_X = self.pipeline.transform(X)

        # Get columns and importance list
        features = list(processed_X.columns)
        importances = self.model.feature_importances_

        # sort and merge importances and column names into a dataframe
        feature_importances = sorted(zip(importances, features), reverse=True)
        sorted_importances, sorted_features = zip(*feature_importances)
        importance_df = pd.DataFrame({'feats': sorted_features[:50], 'importance': sorted_importances[:50]})

        print(importance_df)
        # plot it!
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df, x='importance', y='feats')

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
