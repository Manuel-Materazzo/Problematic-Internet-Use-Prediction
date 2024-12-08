import optuna
import platform
import optuna_distributed
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split

from src.ensembles.ensemble import EnsembleMember, Ensemble


class WeightedEnsemble(Ensemble):

    def __init__(self, members: list[EnsembleMember]):
        super().__init__(members)

    def post_validation_callback(self, X: DataFrame, y: Series, oof_predictions: DataFrame):
        """
        Callback to optimize weights when done training. Do not call manually.
        :param oof_predictions:
        :param X:
        :param y:
        :return:
        """
        print("Optimizing ensemble weights...")
        self._optimize_weights(X, y)

    def show_weights(self):
        """
        Shows ensemble weights
        :return:
        """

        if self.weights is None:
            print("Weights are still not optimized."
                  "This probably means the validation was skipped or the callback is set incorrectly.")
            return

        # Pie chart
        weights = self.weights[self.weights >= 0.005]  # hide small weights in pie chart
        plt.pie(weights, labels=weights.index, autopct="%.0f%%")
        plt.title('Ensemble weights')
        plt.show()

    def _optimize_weights(self, X: DataFrame, y: Series):
        """
        Trains each model of the ensemble on half of the provided data, and calculates the optimal ensemble weights.
        :param X:
        :param y:
        :return:
        """
        # Split into validation and training data
        train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

        predictions_array = []
        model_names = []

        # train each model in the ensemble
        for member in self.members:
            trainer = member['trainer']
            params = member['params']
            optimizer = member['optimizer']

            # if we have an optimizer set and no params are provided, calculate optimal params
            if optimizer is not None and params is None:
                print("No hyperparams provided, auto-optimizing...")
                params = optimizer.tune(X, y, 0.03)
                # save optimized params for later (full training)
                member['params'] = params
                print("Optimal hyperparams: {}".format(params))

            model_names.append(trainer.get_model_name())

            # train the model using early stopping (validation data is not used during training)
            model = trainer.train_model(train_X, train_y, val_X, val_y, params=params)
            # process validation data
            processed_val_X = trainer.get_pipeline().transform(val_X)
            # make predictions and store them
            predictions_array.append(model.predict(processed_val_X))

        # optimize weights to maximize prediction accuracy
        raw_weights = self._optuna_weight_study(val_y, predictions_array)
        self.weights: Series = Series(raw_weights, index=model_names)

    def _optuna_weight_study(self, real_values: Series, predictions_array: list) -> list:
        """
        Uses Optuna to optimize the weights of the ensemble and returns them.
        :param real_values:
        :param predictions_array:
        :return:
        """
        # set log level
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        optuna_distributed.config.disable_logging()
        # create a study
        sampler = optuna.samplers.CmaEsSampler(seed=0)
        pruner = optuna.pruners.HyperbandPruner()
        study = optuna.create_study(sampler=sampler, pruner=pruner, study_name="OptWeights", direction='minimize')
        # leverage distributed training on linux
        if platform.system() != 'Windows':
            study = optuna_distributed.from_study(study)
        # define an objective and start the study
        objective_partial = partial(self._objective, real_values=real_values, predictions_array=predictions_array)
        study.optimize(objective_partial, n_trials=4000, n_jobs=-1)
        # get weights
        return [study.best_params[f"weight{n}"] for n in range(len(predictions_array))]

    def _objective(self, trial, real_values, predictions_array):
        """
        Optimization objective for optuna.
        :param trial:
        :param real_values:
        :param predictions_array:
        :return:
        """
        # Define the weights for the predictions from each model
        weights = [trial.suggest_float(f"weight{n}", 1e-14, 1) for n in range(len(predictions_array))]

        # Calculate the weighted prediction
        predictions = np.average(np.array(predictions_array).T, axis=1, weights=weights)

        # Calculate the score for the weighted prediction
        score = self.members[0]['trainer'].calculate_accuracy(predictions, real_values)
        return score

    def predict(self, X: DataFrame) -> Series:
        if len(self.weights) == 0:
            print("Weights are still not optimized."
                  "This probably means the validation was skipped or the callback is set incorrectly.")
            return Series([])

        if len(self.models) == 0:
            print("No models trained, use train() first")
            return Series([])

        predictions_array = []

        # for each trained model
        for model in self.models:
            # make prediction and add to the prediction list
            predictions_array.append(model.predict(X))

        weighted_pred = np.average(np.array(predictions_array).T, axis=1, weights=self.weights)
        return weighted_pred
