import os
import pickle
from abc import ABC, abstractmethod

from pandas import DataFrame


def load_preprocessor() -> any:
    path = os.path.dirname(os.path.realpath(__file__)) + '/../../target/preprocessor.pkl'
    with open(path, 'rb') as file:
        return pickle.load(file)


class DataPreprocessor(ABC):

    @abstractmethod
    def preprocess_data(self, X: DataFrame):
        """
        Preprocesses the data.
        :return:
        """

    def save_preprocessor(self):
        os.makedirs('../target', exist_ok=True)

        with open('../target/preprocessor.pkl', 'wb') as file:
            pickle.dump(self, file)
