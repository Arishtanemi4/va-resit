import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler, LabelEncoder



def load_and_clean_anomalies(filepath):
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    
    df = pd.read_csv(filepath, header=None, names=column_names, skipinitialspace=True)
    
    df.replace('?', np.nan, inplace=True)
    
    print("--- Missing Values Count per Column ---")
    print(df.isnull().sum())
    
    return df


def drop_redundant_features(df):
    if 'education' in df.columns:
        df = df.drop(columns=['education'])
    return df


def encode_categorical_data(df):
    X = df.drop(columns=['income'])
    y = df['income']
    
    le = LabelEncoder()
    y_encoded = pd.Series(le.fit_transform(y), name='income')
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    return X_encoded, y_encoded


def impute_missing_data_bayesian(X):

    bayesian_imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
    
    X_imputed = pd.DataFrame(bayesian_imputer.fit_transform(X), columns=X.columns)
    
    print(X_imputed.isnull().sum().sum(), "total missing values remain.")
    
    return X_imputed


def scale_data(X):

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled