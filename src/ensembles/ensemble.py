import optuna
import numpy as np
import matplotlib.pyplot as plt
from typing import TypedDict
from functools import partial
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split

from src.enums.accuracy_metric import AccuracyMetric
from src.hyperparameter_optimizers.hp_optimizer import HyperparameterOptimizer
from src.models.model_wrapper import ModelWrapper
from src.trainers.trainer import Trainer

EnsembleMember = TypedDict('EnsembleMember', {'trainer': Trainer, 'params': dict})
LeaderboardEntry = TypedDict('LeaderboardEntry', {'model_name': str, 'accuracy': float, 'iterations': int})


class Ensemble:

    def __init__(self, members: list[EnsembleMember], optimizer: HyperparameterOptimizer = None):
        self.members = members
        self.optimizer = optimizer
        self.accuracy_metric = None
        self.leaderboard = None
        self.weights = None
        self.models: list[ModelWrapper] = []

    def show_weights(self):
        # Pie chart
        weights = self.weights[self.weights >= 0.005]  # hide small weights in pie chart
        plt.pie(weights, labels=weights.index, autopct="%.0f%%")
        plt.title('Ensemble weights')
        plt.show()

    def show_leaderboard(self, X: DataFrame, y: Series) -> float:
        """
        Trains each model of the ensemble and optimizes params if needed.
        Computes an individual model leaderboard, and returns the mean accuracy value across all models.
        :param X:
        :param y:
        :return:
        """
        # if we didn't compute a leaderboard before
        if self.leaderboard is None:
            leaderboardList: list[LeaderboardEntry] = []
            # train each model in the ensemble
            for member in self.members:
                # get the trainer and the params
                trainer = member['trainer']
                params = member['params']

                # check accuracy metric consistency
                if self.accuracy_metric is None:
                    self.accuracy_metric: AccuracyMetric = trainer.metric
                if self.accuracy_metric != trainer.metric:
                    print("Accuracy metric is different across trainers")

                print("Training {}...".format(trainer.get_model_name()))

                # if we have an optimizer set and no params are provided, calculate optimal params
                if self.optimizer is not None and params is None:
                    print("No hyperparams provided, auto-optimizing...")
                    params = self.optimizer.tune(X, y, 0.03)
                    # save optimized params for later (full training)
                    member['params'] = params
                    print("Optimal hyperparams: {}".format(params))

                # train model
                accuracy, iterations = trainer.validate_model(X, y, log_level=0, params=params)
                # append model results to the leaderboard
                leaderboardList.append(
                    LeaderboardEntry(model_name=trainer.get_model_name(), accuracy=accuracy, iterations=iterations)
                )

            self.leaderboard: DataFrame = DataFrame.from_records(leaderboardList)
            self.leaderboard.sort_values('accuracy', ascending=True, inplace=True)

        # print leaderboard and return mean accuracy
        print(self.leaderboard)
        return np.mean(self.leaderboard['accuracy'])

    def optimize_weights(self, X: DataFrame, y: Series) -> Series:
        """
        Trains each model of the ensemble on half of the provided data, and calculates the optimal ensemble weights.
        :param X:
        :param y:
        :return:
        """
        if self.weights is None:
            # Split into validation and training data
            train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

            predictions_array = []
            model_names = []

            # train each model in the ensemble
            for member in self.members:
                trainer = member['trainer']
                params = member['params']

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
        return self.weights

    def _optuna_weight_study(self, real_values: Series, predictions_array: list) -> list:
        """
        Uses Optuna to optimize the weights of the ensemble and returns them.
        :param real_values:
        :param predictions_array:
        :return:
        """
        # set log level
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        # create a study
        sampler = optuna.samplers.CmaEsSampler(seed=0)
        pruner = optuna.pruners.HyperbandPruner()
        study = optuna.create_study(sampler=sampler, pruner=pruner, study_name="OptWeights", direction='minimize')
        # define an objective and start the study
        objective_partial = partial(self._objective, real_values=real_values, predictions_array=predictions_array)
        study.optimize(objective_partial, n_trials=4000)
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

    def train(self, X: DataFrame, y: Series):
        """
        Trains each model of the ensemble on the whole data.
        :param X:
        :param y:
        :return:
        """

        # train each model in the ensemble
        for member in self.members:
            # get the trainer and the params
            trainer = member['trainer']
            params = member['params']

            print("Training {}...".format(trainer.get_model_name()))

            # get iterations from leaderboard
            iterations = \
                self.leaderboard.loc[self.leaderboard['model_name'] == trainer.get_model_name()].iterations.values[0]
            # train model
            model = trainer.train_model(X, y, iterations=iterations, params=params)
            self.models.append(model)

    def predict(self, X: DataFrame) -> Series:

        if len(self.weights) == 0:
            print("No weights provided, use optimize_weights() first")
            return Series([])

        predictions_array = []

        # for each trained model
        for model in self.models:
            # make prediction and add to the prediction list
            predictions_array.append(model.predict(X))

        weighted_pred = np.average(np.array(predictions_array).T, axis=1, weights=self.weights)
        return weighted_pred
