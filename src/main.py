import pandas as pd

from src.data_trasformation_pipeline import DataTrasformationPipeline
from src.trainer import Trainer
#from src.grid_optimizer import GridOptimizer


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

pipeline = DataTrasformationPipeline(X, True)

trainer = Trainer(pipeline)


params = {
    'n_jobs': -1,  # Use all available cores
    'objective': 'reg:squarederror',
    'learning_rate': 0.03,
}

#optimizer = GridOptimizer(pipeline, trainer)

# Train model baseline
print("Manual Baseline:")
boost_rounds = trainer.cross_validation(X, y, **params)

# optimize parameters
#optimizer.tune(X, y)

# fit complete_model on all data from the training data
#complete_model = trainer.train_model(X, y, rounds=boost_rounds, **params)