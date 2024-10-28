import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin


class FunctionalImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        try:
            X['Functional'] = X['Functional'].fillna('Typ')  # Assume typical unless deductions are warranted
        except KeyError:
            print('KeyError')

        return X


class DataTrasformationPipeline:
    def __init__(self, X, imputation_enabled):
        self.imputation_enabled = imputation_enabled
        self.X = X
        self.pipeline = self.build_pipeline()

    def build_pipeline(self):
        #TODO: extract categorical selection to static method
        # Select categorical columns
        categorical_cols = [cname for cname in self.X.columns if self.X[cname].dtype == "object"]

        # Select numerical columns
        numerical_cols = [cname for cname in self.X.columns if self.X[cname].dtype in ['int64', 'float64']]

        # Encoding for categorical data
        # categorical_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        categorical_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        # if imputation is disabled, just encode categorical columns
        if not self.imputation_enabled:
            return ColumnTransformer(transformers=[
                ('cat', categorical_encoder, categorical_cols)
            ])

        # Preprocessing for numerical data
        numerical_transformer = SimpleImputer(strategy='median')

        # Preprocessing for categorical data
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', categorical_encoder)
        ])

        preprocessor = ColumnTransformer(transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
        ])

        # Bundle preprocessing
        return Pipeline(steps=[
            ('functional_imputer', FunctionalImputer()),
            ('preprocessor', preprocessor)
        ])

    def fit_transform(self, dataframe):
        return pd.DataFrame(self.pipeline.fit_transform(dataframe))

    def transform(self, dataframe):
        return pd.DataFrame(self.pipeline.transform(dataframe))

    def get_pipeline_with_training(self, model):
        # Create a complete pipeline, including model training
        return Pipeline(steps=[
            ('preprocessor', self.pipeline),
            ('model', model)
        ])