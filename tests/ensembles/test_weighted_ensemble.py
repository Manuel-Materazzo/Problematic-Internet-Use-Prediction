from src.ensembles.weighted_ensemble import WeightedEnsemble
from tests.ensembles.ensemble_base import EnsembleBase


class TestWeightedEnsemble(EnsembleBase):

    @classmethod
    def setUpClass(cls):
        cls.ensemble = WeightedEnsemble
