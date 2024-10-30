import pandas as pd
from abc import ABC, abstractmethod
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


class DTPipeline(ABC):

    def __init__(self, X: DataFrame, imputation_enabled: bool):
        self.imputation_enabled: bool = imputation_enabled
        self.pipeline = self.__build_pipeline()

        # Select categorical columns
        self.categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]
        # Select numerical columns
        self.numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

    @abstractmethod
    def __build_pipeline(self) -> Pipeline | ColumnTransformer:
        pass

    def fit_transform(self, dataframe: DataFrame) -> DataFrame:
        """
        Wrapping function to fit and transform the dataframe.
        :param dataframe:
        :return:
        """
        return pd.DataFrame(self.pipeline.fit_transform(dataframe))

    def transform(self, dataframe: DataFrame) -> DataFrame:
        """
        Wrapping function to transform the dataframe.
        :param dataframe:
        :return:
        """
        return pd.DataFrame(self.pipeline.transform(dataframe))

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
