import xgboost as xgb
from pandas import DataFrame, Series

from src.enums.accuracy_metric import AccuracyMetric
from src.pipelines.dt_pipeline import DTPipeline
from src.trainers.trainer import Trainer


class FastCrossTrainer(Trainer):

    def __init__(self, pipeline: DTPipeline, metric: AccuracyMetric = AccuracyMetric.MAE):
        super().__init__(pipeline, metric=metric)

    def validate_model(self, X: DataFrame, y: Series, rounds=1000, log_level=1, **xgb_params) -> (float, int):
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
            metrics=self.metric.value.lower(),
            early_stopping_rounds=5,
            seed=0,
            as_pandas=True)

        self.evals = []

        # for each split
        for i in range(5):
            # extract Series of accuracy for the split
            split_accuracy = cv_results[f'test-{self.metric.value.lower()}-mean'] + cv_results[f'test-{self.metric.value.lower()}-std']*i
            # format dictionary to be standard compliant
            self.evals.append({
                'validation_0': {
                    self.metric.value.lower(): split_accuracy
                }
            })

        # Extract the mean of the accuracy from cross-validation results
        # optimal point (iteration) where the model achieved its best performance
        accuracy = cv_results['test-' + self.metric.value.lower() + '-mean'].min()
        # if you train the model again, same seed, no early stopping, you can put this index as num_boost_round to get same result
        best_round = cv_results['test-' + self.metric.value.lower() + '-mean'].idxmin()

        if log_level > 0:
            print("#{} Cross-Validation {}: {}".format(best_round, self.metric.value, accuracy))

        return accuracy, best_round
