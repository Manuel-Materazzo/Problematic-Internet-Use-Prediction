import inspect
import unittest

from src.enums.accuracy_metric import AccuracyMetric
from src.models.catboost_classifier import CatBoostClassifierWrapper
from src.models.catboost_regressor import CatBoostRegressorWrapper
from src.models.hgb_classifier import HGBClassifierWrapper
from src.models.hgb_regressor import HGBRegressorWrapper
from src.models.lgbm_classifier import LGBMClassifierWrapper
from src.models.lgbm_regressor import LGBMRegressorWrapper
from src.models.xgb_classifier import XGBClassifierWrapper
from src.models.xgb_regressor import XGBRegressorWrapper


class TrainerBase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.classification_X = None
        cls.classification_y = None
        cls.regression_X = None
        cls.regression_y = None
        cls.classification_pipeline = None
        cls.regression_pipeline = None
        cls.trainer = None

    def test_validate_catboost_classifier(self):
        if self.trainer is not None:
            model_type = CatBoostClassifierWrapper()
            self._validate(model_type, AccuracyMetric.AUC, self.classification_X, self.classification_y)

    def test_validate_catboost_regressor(self):
        if self.trainer is not None:
            model_type = CatBoostRegressorWrapper()
            self._validate(model_type, AccuracyMetric.RMSE, self.regression_X, self.regression_y)

    def test_validate_hgb_classifier(self):
        if self.trainer is not None:
            model_type = HGBClassifierWrapper()
            self._validate(model_type, AccuracyMetric.AUC, self.classification_X, self.classification_y)

    def test_validate_hgb_regressor(self):
        if self.trainer is not None:
            model_type = HGBRegressorWrapper()
            self._validate(model_type, AccuracyMetric.RMSE, self.regression_X, self.regression_y)

    def test_validate_lgmb_classifier(self):
        if self.trainer is not None:
            model_type = LGBMClassifierWrapper()
            self._validate(model_type, AccuracyMetric.AUC, self.classification_X, self.classification_y)

    def test_validate_lgmb_regressor(self):
        if self.trainer is not None:
            model_type = LGBMRegressorWrapper()
            self._validate(model_type, AccuracyMetric.RMSE, self.regression_X, self.regression_y)

    def test_validate_xgb_classifier(self):
        if self.trainer is not None:
            model_type = XGBClassifierWrapper()
            self._validate(model_type, AccuracyMetric.AUC, self.classification_X, self.classification_y)

    def test_validate_xgb_regressor(self):
        if self.trainer is not None:
            model_type = XGBRegressorWrapper()
            self._validate(model_type, AccuracyMetric.RMSE, self.regression_X, self.regression_y)

    def test_train_catboost_classifier(self):
        if self.trainer is not None:
            model_type = CatBoostClassifierWrapper()
            self._train(model_type, AccuracyMetric.AUC, self.classification_X, self.classification_y)

    def test_train_catboost_regressor(self):
        if self.trainer is not None:
            model_type = CatBoostRegressorWrapper()
            self._train(model_type, AccuracyMetric.RMSE, self.regression_X, self.regression_y)

    def test_train_hgb_classifier(self):
        if self.trainer is not None:
            model_type = HGBClassifierWrapper()
            self._train(model_type, AccuracyMetric.AUC, self.classification_X, self.classification_y)

    def test_train_hgb_regressor(self):
        if self.trainer is not None:
            model_type = HGBRegressorWrapper()
            self._train(model_type, AccuracyMetric.RMSE, self.regression_X, self.regression_y)

    def test_train_lgmb_classifier(self):
        if self.trainer is not None:
            model_type = LGBMClassifierWrapper()
            self._train(model_type, AccuracyMetric.AUC, self.classification_X, self.classification_y)

    def test_train_lgmb_regressor(self):
        if self.trainer is not None:
            model_type = LGBMRegressorWrapper()
            self._train(model_type, AccuracyMetric.RMSE, self.regression_X, self.regression_y)

    def test_train_xgb_classifier(self):
        if self.trainer is not None:
            model_type = XGBClassifierWrapper()
            self._train(model_type, AccuracyMetric.AUC, self.classification_X, self.classification_y)

    def test_train_xgb_regressor(self):
        if self.trainer is not None:
            model_type = XGBRegressorWrapper()
            self._train(model_type, AccuracyMetric.RMSE, self.regression_X, self.regression_y)

    def _validate(self, model_type, metric, X, y):
        trainer_instance = self._instantiate_trainer(X, y, model_type, metric)

        accuracy, best_iteration, prediction_comparison = trainer_instance.validate_model(X, y, iterations=10,
                                                                                          output_prediction_comparison=True)
        trainer_instance.show_loss()
        trainer_instance.show_feature_importance(X)
        self.assertGreaterEqual(accuracy, 0)
        self.assertGreaterEqual(best_iteration, 0)
        self.assertFalse(prediction_comparison.empty)

    def _train(self, model_type, metric, X, y):

        trainer_instance = self._instantiate_trainer(X, y, model_type, metric)

        model = trainer_instance.train_model(X, y, iterations=10)
        self.assertIsNotNone(model)

    def _instantiate_trainer(self, X, y, model_type, metric):
        sig = inspect.signature(self.trainer.__init__)
        data_required = 'X' in sig.parameters

        if data_required:
            return self.trainer(pipeline=self.classification_pipeline, X=X, y=y, model_wrapper=model_type,
                                metric=metric)
        else:
            return self.trainer(pipeline=self.classification_pipeline, model_wrapper=model_type, metric=metric)
