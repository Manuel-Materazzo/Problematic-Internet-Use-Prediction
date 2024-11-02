import pandas as pd
from abc import ABC, abstractmethod
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


class DTPipeline(ABC):

    def __init__(self, X: DataFrame, imputation_enabled: bool):
        # Select categorical columns
        self.categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]
        # Select numerical columns
        self.numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

        self.imputation_enabled: bool = imputation_enabled
        self.pipeline = self.build_pipeline()

    @abstractmethod
    def build_pipeline(self) -> Pipeline | ColumnTransformer:
        pass

    def fit_transform(self, dataframe: DataFrame) -> DataFrame:
        """
        Wrapping function to fit and transform the dataframe.
        :param dataframe:
        :return:
        """
        imputed_dataframe = pd.DataFrame(self.pipeline.fit_transform(dataframe))
        return imputed_dataframe

    def transform(self, dataframe: DataFrame) -> DataFrame:
        """
        Wrapping function to transform the dataframe.
        :param dataframe:
        :return:
        """
        imputed_dataframe = pd.DataFrame(self.pipeline.transform(dataframe))
        return imputed_dataframe

    def get_pipeline_with_training(self, model: XGBRegressor) -> Pipeline:
        """
        Gets the pipeline with the added model training step
        :param model:
        :return:
        """
        return Pipeline(steps=[
            ('preprocessor', self.pipeline),
            ('model', model)
        ])

    def find_one_hot_encoder(self, transformer):
        """
        Checks if the pipeline contains one hot encoder and returns the instance.
        :param pipeline:
        :return:
        """
        if isinstance(transformer, Pipeline):
            for name, step in transformer.steps:
                result = self.find_one_hot_encoder(step)
                if result is not None:
                    return result
        elif isinstance(transformer, ColumnTransformer):
            for name, trans, columns in transformer.transformers_:
                result = self.find_one_hot_encoder(trans)
                if result is not None:
                    return result
        elif isinstance(transformer, OneHotEncoder):
            return transformer
        return None
