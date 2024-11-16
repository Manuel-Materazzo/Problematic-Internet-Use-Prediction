import pandas as pd
import time

from src.enums.accuracy_metric import AccuracyMetric
from src.models.xgb_regressor import XGBRegressorWrapper
from src.pipelines.dt_pipeline import save_data_model
from src.pipelines.housing_prices_competition_dt_pipeline import HousingPricesCompetitionDTPipeline
from src.trainers.simple_trainer import SimpleTrainer
from src.trainers.fast_cross_trainer import FastCrossTrainer
from src.trainers.accurate_cross_trainer import AccurateCrossTrainer
from src.trainers.cached_accurate_cross_trainer import CachedAccurateCrossTrainer
from src.hyperparameter_optimizers.custom_grid_optimizer import CustomGridOptimizer
from src.hyperparameter_optimizers.default_grid_optimizer import DefaultGridOptimizer
from src.hyperparameter_optimizers.hyperopt_bayesian_optimizer import HyperoptBayesianOptimizer
from src.hyperparameter_optimizers.optuna_optimizer import OptunaOptimizer
from src.trainers.trainer import save_model


def load_data():
    # Load the data
    iowa_file_path = '../resources/train.csv'
    home_data = pd.read_csv(iowa_file_path)

    # Remove rows with missing target, separate target from predictors
    pruned_home_data = home_data.dropna(axis=0, subset=['SalePrice'])
    y = pruned_home_data.SalePrice
    X = pruned_home_data.drop(['SalePrice'], axis=1)
    return X, y


print("Loading data...")
X, y = load_data()

# save model file for current dataset on target directory
print("Saving data model...")
save_data_model(X)

# instantiate data pipeline
pipeline = HousingPricesCompetitionDTPipeline(X, True)

# pick a model, a trainer and an optimizer
model_type = XGBRegressorWrapper()
trainer = CachedAccurateCrossTrainer(pipeline, model_type, X, y)
optimizer = DefaultGridOptimizer(trainer, model_type)

# optimize parameters
print("Tuning Hyperparameters...")
start = time.time()
optimized_params = optimizer.tune(X, y, 0.03)
end = time.time()

print("Tuning took {} seconds".format(end - start))

print("Training and evaluating model...")
_, boost_rounds = trainer.validate_model(X, y, log_level=1, params=optimized_params)
print()

# fit complete_model on all data from the training data
print("Fitting complete model...")
complete_model = trainer.train_model(X, y, iterations=boost_rounds, params=optimized_params)

# save trained pipeline on target directory
print("Saving pipeline...")
pipeline.save_pipeline()

# save model on target directory
print("Saving fitted model...")
save_model(complete_model)
