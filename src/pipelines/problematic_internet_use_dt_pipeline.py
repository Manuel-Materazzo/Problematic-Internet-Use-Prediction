from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.pipelines.dt_pipeline import DTPipeline


class ProblematicInternetUseDTPipeline(DTPipeline):
    def __init__(self, X: DataFrame):
        super().__init__(X)

    def build_pipeline(self) -> Pipeline | ColumnTransformer:

        # Preprocessing for numerical data
        numerical_transformer = SimpleImputer(strategy='median')

        preprocessor = ColumnTransformer(transformers=[
                ('num', numerical_transformer, self.numerical_cols)
        ])

        # Bundle preprocessing
        return Pipeline(steps=[
            ('preprocessor', preprocessor)
        ], memory=None)
