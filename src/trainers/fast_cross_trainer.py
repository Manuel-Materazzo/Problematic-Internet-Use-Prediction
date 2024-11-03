import xgboost as xgb
from pandas import DataFrame, Series

from src.pipelines.dt_pipeline import DTPipeline
from src.trainers.trainer import Trainer


class FastCrossTrainer(Trainer):

    def __init__(self, pipeline: DTPipeline):
        super().__init__(pipeline)

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
