from src.hyperparameter_optimizers.custom_grid_optimizer import CustomGridOptimizer
from src.pipelines.housing_prices_competition_dt_pipeline import HousingPricesCompetitionDTPipeline
from tests.data_load import load_regression_data
from tests.hyperparameter_optimizers.hp_optimizer_base import HpOptimizerBase


class TestCustomGridOptimizer(HpOptimizerBase):

    @classmethod
    def setUpClass(cls):
        cls.regression_X, cls.regression_y = load_regression_data()
        cls.pipeline = HousingPricesCompetitionDTPipeline(cls.regression_X)
        cls.optimizer = CustomGridOptimizer
