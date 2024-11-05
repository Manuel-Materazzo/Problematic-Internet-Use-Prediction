import pandas as pd
import time

from src.enums.accuracy_metric import AccuracyMetric
from src.pipelines.dt_pipeline import save_data_model
from src.pipelines.housing_prices_competition_dt_pipeline import HousingPricesCompetitionDTPipeline
from src.trainers.simple_trainer import SimpleTrainer
from src.trainers.fast_cross_trainer import FastCrossTrainer
from src.trainers.accurate_cross_trainer import AccurateCrossTrainer
from src.trainers.cached_accurate_cross_trainer import CachedAccurateCrossTrainer
from src.hyperparameter_optimizers.custom_grid_optimizer import CustomGridOptimizer
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


X, y = load_data()

# save model file for current dataset on target directory
save_data_model(X)

pipeline = HousingPricesCompetitionDTPipeline(X, True)

trainer = CachedAccurateCrossTrainer(pipeline, X, y, metric=AccuracyMetric.RMSE)

optimizer = CustomGridOptimizer(trainer)

# optimize parameters
start = time.time()
optimized_params = optimizer.tune(X, y, 0.03)
end = time.time()

print("Tuning took {} seconds".format(end - start))

_, boost_rounds = trainer.validate_model(X, y, log_level=1, **optimized_params)
print()

# fit complete_model on all data from the training data
complete_model = trainer.train_model(X, y, rounds=boost_rounds, **optimized_params)

# save trained pipeline on target directory
pipeline.save_pipeline()

# save model on target directory
save_model(complete_model)
