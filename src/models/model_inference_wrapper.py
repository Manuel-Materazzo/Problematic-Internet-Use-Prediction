from abc import ABC, abstractmethod


class ModelInferenceWrapper(ABC):
    """
    Interface for wrapping whatever can predict values in order to standardize methods at inference time.
    """

    @abstractmethod
    def predict(self, X):
        """
        Generate predictions based on input data.
        :param X:
        :return:
        """
