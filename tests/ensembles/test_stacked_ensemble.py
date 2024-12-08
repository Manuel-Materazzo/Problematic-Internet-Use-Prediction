from src.ensembles.stacked_ensemble import StackedEnsemble
from tests.ensembles.ensemble_base import EnsembleBase


class TestWeightedEnsemble(EnsembleBase):

    @classmethod
    def setUpClass(cls):
        cls.ensemble = StackedEnsemble
