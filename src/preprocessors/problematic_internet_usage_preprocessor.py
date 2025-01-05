from pandas import DataFrame

from src.preprocessors.data_preprocessor import DataPreprocessor

seasons_dictionary = {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3}
half_missing = []
null_filling_cols = [
    'Basic_Demos_Sex', 'FGC_FGC_CU', 'FGC_FGC_CU_Zone',
    'FGC_FGC_GSD_Zone', 'FGC_FGC_PU_Zone', 'FGC_FGC_SRL_Zone',
    'FGC_FGC_GSND_Zone', 'FGC_FGC_SRR_Zone', 'FGC_FGC_TL_Zone'
]


def encode_and_fill(X):
    # encode seasons
    for col in X.columns:
        if '_Season' in col:
            # Assign values based on the season dictionary and set -1 for empty values
            X[col] = X[col].apply(lambda x: seasons_dictionary.get(x, -1) if x != '' else -1)

    # for col in null_filling_cols:
    # Explicitely set -1 to mark a missing value
    # X[col] = X[col].fillna(-1)


def select_columns(X):
    for column in X.columns:
        # drop PCIAT columns, the test was almost always administered AFTER the diagnosis and it's not on the test set
        # i'm not using pciat_indexes_columns, because could be other PCIAT  (not indexes) on the test set
        # also drop features with over 50% of missing values
        if 'PCIAT_' in column or column in half_missing:
            X.drop([column], axis=1, inplace=True)


def feature_engineering(df):
    df['BMI_Age'] = df['Physical_BMI'] * df['Basic_Demos_Age']
    df['Internet_Hours_Age'] = df['PreInt_EduHx_computerinternet_hoursday'] * df['Basic_Demos_Age']
    df['BMI_Internet_Hours'] = df['Physical_BMI'] * df['PreInt_EduHx_computerinternet_hoursday']
    df['BFP_BMI'] = df['BIA_BIA_Fat'] / df['BIA_BIA_BMI']
    df['FFMI_BFP'] = df['BIA_BIA_FFMI'] / df['BIA_BIA_Fat']
    df['FMI_BFP'] = df['BIA_BIA_FMI'] / df['BIA_BIA_Fat']
    df['LST_TBW'] = df['BIA_BIA_LST'] / df['BIA_BIA_TBW']
    df['BFP_BMR'] = df['BIA_BIA_Fat'] * df['BIA_BIA_BMR']
    df['BFP_DEE'] = df['BIA_BIA_Fat'] * df['BIA_BIA_DEE']
    df['BMR_Weight'] = df['BIA_BIA_BMR'] / df['Physical_Weight']
    df['DEE_Weight'] = df['BIA_BIA_DEE'] / df['Physical_Weight']
    df['SMM_Height'] = df['BIA_BIA_SMM'] / df['Physical_Height']
    df['Muscle_to_Fat'] = df['BIA_BIA_SMM'] / df['BIA_BIA_FMI']
    df['Hydration_Status'] = df['BIA_BIA_TBW'] / df['Physical_Weight']
    df['ICW_TBW'] = df['BIA_BIA_ICW'] / df['BIA_BIA_TBW']
    df['BMI_PHR'] = df['Physical_BMI'] * df['Physical_HeartRate']


class ProblematicInternetUsagePreprocessor(DataPreprocessor):

    def preprocess_data(self, X: DataFrame):
        global half_missing
        half_missing = [val for val in X.columns[(X.isnull().sum() * 100 / len(X)) > 60]]
        encode_and_fill(X)
        select_columns(X)
        feature_engineering(X)
