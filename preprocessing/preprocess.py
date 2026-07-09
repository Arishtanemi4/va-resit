import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler, LabelEncoder



# Reads the raw census file and swaps the '?' placeholders for real
# missing values, so later steps can actually detect what's missing.
def load_and_clean_anomalies(filepath):
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]

    df = pd.read_csv(filepath, header=None, names=column_names, skipinitialspace=True)

    df.replace('?', np.nan, inplace=True)

    print("--- Missing Values Count per Column ---")
    print(df.isnull().sum())  # count blanks per column, then just print those counts

    return df


# Drops the 'education' text column because 'education-num' already
# stores the same information as a number, so keeping both is redundant.
def drop_redundant_features(df):
    if 'education' in df.columns:
        df = df.drop(columns=['education'])
    return df


# Turns each text category (e.g. job type) into its own 0/1 column so a
# model/PCA can use it, and separates out the income label being described.
def encode_categorical_data(df):
    X = df.drop(columns=['income'])
    y = df['income']

    le = LabelEncoder()
    y_encoded = pd.Series(le.fit_transform(y), name='income')

    categorical_cols = X.select_dtypes(include=['object']).columns
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    return X_encoded, y_encoded


# Converts each category into a single number code (blanks stay blank) so
# the imputer below can treat a missing category like a missing number.
def encode_ordinal_categoricals(df):
    X = df.drop(columns=['income'])
    categorical_cols = X.select_dtypes(include=['object']).columns

    X_ordinal = X.copy()
    category_maps = {}
    for col in categorical_cols:  # go through each text column one at a time, to build its own code mapping
        categories = sorted(X[col].dropna().unique())  # drop blanks, then list each unique category once
        code_to_label = dict(enumerate(categories))
        label_to_code = {label: code for code, label in code_to_label.items()}  # flip the map so we can look up a code from a label
        X_ordinal[col] = X[col].map(label_to_code)
        category_maps[col] = code_to_label

    return X_ordinal, category_maps


# Fills in the missing values by predicting them from a person's other
# answers, using a Bayesian statistical model rather than a simple average.
def impute_missing_data_bayesian(X):

    bayesian_imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)

    X_imputed = pd.DataFrame(bayesian_imputer.fit_transform(X), columns=X.columns)

    print(X_imputed.isnull().sum().sum(), "total missing values remain.")  # count remaining blanks, then total them into one number

    return X_imputed


# Turns the imputer's filled-in number codes back into readable category
# names, so the exported data stays human-friendly for Tableau.
def decode_imputed_categoricals(df, X_imputed, category_maps):
    df_filled = df.copy()
    for col, code_to_label in category_maps.items():  # go through each column that had missing values, to decode it
        max_code = len(code_to_label) - 1
        rounded_codes = X_imputed[col].round().clip(0, max_code).astype(int)  # round to the nearest code, keep it in range, then make it a whole number
        df_filled[col] = rounded_codes.map(code_to_label)
    return df_filled


# Rescales every number column onto the same scale, so a column like
# income in dollars doesn't dominate columns like age just by being bigger.
def scale_data(X):

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled