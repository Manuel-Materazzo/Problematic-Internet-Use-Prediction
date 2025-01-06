import pandas as pd
import time
import re
import optuna
from functools import partial

from sklearn.metrics import cohen_kappa_score

from src.enums.accuracy_metric import AccuracyMetric
from src.enums.optimization_direction import OptimizationDirection
from src.models.xgb_regressor import XGBRegressorWrapper
from src.pipelines.dt_pipeline import save_data_model
from src.pipelines.problematic_internet_use_dt_pipeline import ProblematicInternetUseDTPipeline
from src.preprocessors.problematic_internet_usage_preprocessor import ProblematicInternetUsagePreprocessor
from src.trainers.simple_trainer import SimpleTrainer
from src.trainers.accurate_cross_trainer import AccurateCrossTrainer
from src.trainers.cached_accurate_cross_trainer import CachedAccurateCrossTrainer
from src.hyperparameter_optimizers.custom_grid_optimizer import CustomGridOptimizer
from src.hyperparameter_optimizers.default_grid_optimizer import DefaultGridOptimizer
from src.hyperparameter_optimizers.hyperopt_bayesian_optimizer import HyperoptBayesianOptimizer
from src.hyperparameter_optimizers.optuna_optimizer import OptunaOptimizer
from src.trainers.trainer import save_model


def load_data():
    # Load the data
    file_path = '../resources/cmi-problematic-internet-usage/train.csv'
    data = pd.read_csv(file_path, index_col='id')

    # Remove rows with missing target, separate target from predictors
    data.dropna(axis=0, subset=['sii'], inplace=True)
    y = data['sii']
    X = data.drop(['sii'], axis=1)

    # standardize column names
    data = data.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '_', x))
    X = X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '_', x))
    return X, y


def calculate_sii(pciat_scores, weight):
    # define a utility method to calculate sii
    pciat_scores = pciat_scores * weight
    bins = pd.cut(pciat_scores, bins=[-float('inf'), 30, 50, 80, float('inf')], labels=[0, 1, 2, 3], right=False)
    return bins.astype(int)


def objective(trial, real_sii_values, pciat_scores):
    # define weight optimization objective

    # Define the weights for the predictions from each model
    weight = trial.suggest_float("weight", 0.8, 1.5)

    # Calculate the weighted prediction
    sii_predictions = calculate_sii(pciat_scores, weight)

    # Calculate the score for the weighted prediction
    score = cohen_kappa_score(sii_predictions, real_sii_values)
    return score


# set log level
optuna.logging.set_verbosity(optuna.logging.ERROR)

print("Loading data...")
X, y = load_data()

# save model file for current dataset on target directory
print("Saving data model...")
save_data_model(X)

# extract the regression target
pciat_regression_y = X['PCIAT_PCIAT_Total']

# instantiate data preprocessor
preprocessor = ProblematicInternetUsagePreprocessor()

# preprocess data
preprocessor.preprocess_data(X)

# instantiate data pipeline
pipeline = ProblematicInternetUseDTPipeline(X)

# pick a model, and a trainer
model_type = XGBRegressorWrapper(early_stopping_rounds=50)
trainer = AccurateCrossTrainer(pipeline, model_type, metric=AccuracyMetric.RMSE)
optimizer = CustomGridOptimizer(trainer, model_type, direction=OptimizationDirection.MINIMIZE)

# optimize parameters
print("Tuning Hyperparameters...")
start = time.time()
optimized_params = optimizer.tune(X, y, 0.03)
end = time.time()
print("Tuning took {} seconds".format(end - start))

# Train model
print("Training and evaluating model...")
accuracy, iterations, oof_prediction_comparisons = trainer.validate_model(X, pciat_regression_y, log_level=0,
                                                                          params=optimized_params,
                                                                          output_prediction_comparison=True)
print(f"Cross RMSE score: {accuracy}")

# add real values to oof prediction dataframe (leverage indexes)
oof_prediction_comparisons['real_sii'] = y

# optimize sii weight
print("Optimize sii weights...")
# create a study
sampler = optuna.samplers.CmaEsSampler(seed=0)
pruner = optuna.pruners.HyperbandPruner()
study = optuna.create_study(sampler=sampler, pruner=pruner, study_name="OptSiiWeight", direction='maximize')

# define an objective and start the study
objective_partial = partial(objective, real_sii_values=oof_prediction_comparisons['real_sii'],
                            pciat_scores=oof_prediction_comparisons['predictions'])
study.optimize(objective_partial, n_trials=4000)

print("Best weights:")
print(study.best_params)
pciat_weight = study.best_params['weight']

# calculate sii and add it to oof prediction dataframe (leverage indexes)
oof_prediction_comparisons['predicted_sii'] = calculate_sii(oof_prediction_comparisons['predictions'], pciat_weight)

# compute QWK score
regression_qwk_score = cohen_kappa_score(
    oof_prediction_comparisons['real_sii'],
    oof_prediction_comparisons['predicted_sii'],
    weights='quadratic'
)
print(f"Cross QWK score: {regression_qwk_score}")

# fit complete_model on all data from the training data
print("Fitting complete model...")
complete_model = trainer.train_model(X, y, iterations=iterations, params=optimized_params)

# save preprocessor on target directory
print("Saving preprocessor...")
preprocessor.save_preprocessor()

# save trained pipeline on target directory
print("Saving pipeline...")
pipeline.save_pipeline()

# save model on target directory
print("Saving fitted model...")
save_model(complete_model)
