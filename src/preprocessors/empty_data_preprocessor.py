from pandas import DataFrame

from src.preprocessors.data_preprocessor import DataPreprocessor


class EmptyDataPreprocessor(DataPreprocessor):

    def preprocess_data(self, X: DataFrame):
        # Do nothing
        pass
