from src.ensembles.ensemble import Ensemble, EnsembleMember


class RidgeEnsemble(Ensemble):

    def __init__(self, members: list[EnsembleMember]):
        super().__init__(members)

