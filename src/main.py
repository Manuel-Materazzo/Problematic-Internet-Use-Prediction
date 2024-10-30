import pandas as pd

from src.pipelines.housing_prices_competition_dt_pipeline import HousingPricesCompetitionDTPipeline
from src.trainer import Trainer
from src.hyperparameter_optimizers.accurate_grid_optimizer import AccurateGridOptimizer


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

pipeline = HousingPricesCompetitionDTPipeline(X, True)

trainer = Trainer(pipeline)

params = {
    'n_jobs': -1,  # Use all available cores
    'objective': 'reg:squarederror',
    'learning_rate': 0.03,
}

optimizer = AccurateGridOptimizer(trainer)

# optimize parameters
optimized_params = optimizer.tune(X, y, 0.03)

# Train model baseline
# print()
# print("Before optimization:")
# _, boost_rounds = trainer.cross_validation(X, y, log_level=1, **params)
# print()

print("After optimization:")
_, boost_rounds = trainer.cross_validation(X, y, log_level=1, **optimized_params)
print()

# fit complete_model on all data from the training data
complete_model = trainer.train_model(X, y, rounds=boost_rounds, **params)
