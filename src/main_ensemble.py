import pandas as pd
import re
import optuna

import torch
from sklearn.metrics import cohen_kappa_score

from src.ensembles.weighted_ensemble import WeightedEnsemble
from src.enums.accuracy_metric import AccuracyMetric
from src.enums.optimization_direction import OptimizationDirection
from src.models.xgb_regressor import XGBRegressorWrapper
from src.models.catboost_regressor import CatBoostRegressorWrapper
from src.models.lgbm_regressor import LGBMRegressorWrapper
from src.models.tabnet_regressor import TabNetRegressorWrapper
from src.pipelines.dt_pipeline import save_data_model
from src.pipelines.problematic_internet_use_dt_pipeline import ProblematicInternetUseDTPipeline
from src.preprocessors.problematic_internet_usage_preprocessor import ProblematicInternetUsagePreprocessor
from src.trainers.accurate_cross_trainer import AccurateCrossTrainer
from src.trainers.cached_accurate_cross_trainer import CachedAccurateCrossTrainer
from src.hyperparameter_optimizers.custom_grid_optimizer import CustomGridOptimizer
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

# instantiate data pipeline and preprocessor
preprocessor = ProblematicInternetUsagePreprocessor()
pipeline = ProblematicInternetUseDTPipeline(X)

# extract the regression target
pciat_regression_y = X['PCIAT_PCIAT_Total']

# preprocess data
preprocessor.preprocess_data(X)

# XGB model, trainer and optimizer
xgb_model_type = XGBRegressorWrapper(early_stopping_rounds=50)
xgb_trainer = CachedAccurateCrossTrainer(pipeline, xgb_model_type, X, pciat_regression_y, metric=AccuracyMetric.RMSE)
xgb_optimizer = CustomGridOptimizer(xgb_trainer, xgb_model_type, direction=OptimizationDirection.MINIMIZE)

# Catboost model, trainer and optimizer
catboost_model_type = CatBoostRegressorWrapper(early_stopping_rounds=50)
catboost_trainer = CachedAccurateCrossTrainer(pipeline, catboost_model_type, X, pciat_regression_y,
                                              metric=AccuracyMetric.RMSE)
catboost_optimizer = OptunaOptimizer(catboost_trainer, catboost_model_type, direction=OptimizationDirection.MINIMIZE)

# LGBM model, trainer and optimizer
lgbm_model_type = LGBMRegressorWrapper(early_stopping_rounds=50)
lgbm_trainer = CachedAccurateCrossTrainer(pipeline, lgbm_model_type, X, pciat_regression_y, metric=AccuracyMetric.RMSE)
lgbm_optimizer = CustomGridOptimizer(lgbm_trainer, lgbm_model_type, direction=OptimizationDirection.MINIMIZE)

# TabNet model,trainer and optimizer
tabnet_model_type = TabNetRegressorWrapper(early_stopping_rounds=200)
tabnet_trainer = AccurateCrossTrainer(pipeline, tabnet_model_type, metric=AccuracyMetric.RMSE)
tabnet_optimizer = OptunaOptimizer(tabnet_trainer, tabnet_model_type, direction=OptimizationDirection.MINIMIZE)

# Define ensemble
# to speed up submission, optimizer was provided on the first run only
ensemble = WeightedEnsemble(members=[
    {
        'trainer': xgb_trainer,
        'params': {
            'objective': 'reg:squarederror',
            'learning_rate': 0.03,
            'max_depth': 3,
            'min_child_weight': 4,
            'gamma': 0.0,
            'subsample': 0.6,
            'colsample_bytree': 0.7,
            'n_jobs': -1,
            'reg_alpha': 100
        },
        'optimizer': None
    },
    {
        'trainer': catboost_trainer,
        'params': {
            'loss_function': 'RMSE',
            'grow_policy': 'SymmetricTree',
            'bagging_temperature': 1.0573025149057662,
            'learning_rate': 0.03,
            'depth': 7,
            'l2_leaf_reg': 5.283378489381951,
            'min_data_in_leaf': 37,
            'thread_count': -1,
            'colsample_bylevel': 0.831591113092739,
            'random_strength': 2.683007858572137,
            'max_bin': 315
        },
        'optimizer': None
    },
    {
        'trainer': lgbm_trainer,
        'params': {
            'boosting_type': 'gbdt',
            'force_col_wise': True,
            'num_leaves': 64,
            'max_depth': 8,
            'learning_rate': 0.03,
            'n_estimators': 20000,
            'min_child_samples': 20,
            'reg_alpha': 10,
            'reg_lambda': 0.05,
            'colsample_bytree': 0.75,
            'colsample_bynode': 0.6,
            'extra_trees': True,
            'max_bin': 255,
            'subsample': 0.6,
            'n_jobs': -1,
            'random_state': 0
        },
        'optimizer': None
    },
    {
        'trainer': tabnet_trainer,
        # optimization took more than 12 hours, trying with a set of premade params
        'params': {
            'n_d': 64,  # Width of the decision prediction layer
            'n_a': 64,  # Width of the attention embedding for each step
            'n_steps': 5,  # Number of steps in the architecture
            'gamma': 1.5,  # Coefficient for feature selection regularization
            'n_independent': 2,  # Number of independent GLU layer in each GLU block
            'n_shared': 2,  # Number of shared GLU layer in each GLU block
            'lambda_sparse': 1e-4,  # Sparsity regularization
            'optimizer_fn': torch.optim.Adam,
            'optimizer_params': dict(lr=2e-2, weight_decay=1e-5),
            'mask_type': 'entmax',
            'scheduler_params': dict(mode="min", patience=10, min_lr=1e-5, factor=0.5),
            'scheduler_fn': torch.optim.lr_scheduler.ReduceLROnPlateau,
            'verbose': 1,
        },
        'optimizer': None
    }
])

# train models and compute a cross-validation score leaderboard
print("Training and evaluating ensemble...")
accuracy = ensemble.validate_models_and_show_leaderboard(X, pciat_regression_y)

print(f"Ensemble Cross RMSE score: {accuracy}")

# fit complete_model on all data from the training data
print("Fitting complete model...")
complete_model = ensemble.train(X, pciat_regression_y)

# save preprocessor on target directory
print("Saving preprocessor...")
preprocessor.save_preprocessor()

# save trained pipeline on target directory
print("Saving pipeline...")
pipeline.save_pipeline()

# save model on target directory
print("Saving fitted model...")
save_model(complete_model)
