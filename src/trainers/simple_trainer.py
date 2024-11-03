import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from src.pipelines.dt_pipeline import DTPipeline
from src.trainers.trainer import Trainer


class SimpleTrainer(Trainer):

    def __init__(self, pipeline: DTPipeline):
        super().__init__(pipeline)

    def validate_model(self, X: DataFrame, y: Series, log_level=1, rounds=None, **xgb_params) -> (float, int):
        """
        Trains a XGBoost regressor on the provided training data by splitting it into training and validation sets.
        This uses early stopping and will return the optimal number of boosting rounds alongside the MAE.
        :param rounds: ignored
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
