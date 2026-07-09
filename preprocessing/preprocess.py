import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler



# Reads the raw census file, swaps the '?' placeholders for real missing
# values, and drops 'education-num' since it's just a numeric restatement
# of 'education' (checked below, not assumed) — we keep the readable
# 'education' label instead since that's more useful for Tableau.
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

    labels_per_code = df.groupby('education-num')['education'].nunique()  # count how many different education labels share each education-num code
    if not (labels_per_code == 1).all():
        raise ValueError("education-num no longer matches education 1-to-1, so it can't be safely dropped")
    df = df.drop(columns=['education-num'])

    return df


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


# Buckets each person's age into a readable range (e.g. "30-39"), so the
# dashboard can show income by age group instead of by exact age.
def add_age_group(df):
    df = df.copy()
    bins = [0, 20, 30, 40, 50, 60, 70, 150]
    labels = ['<20', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
    df['age-group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    return df


# For each dashboard column given, adds how many rows share that value
# (absolute) and what percentage of the whole dataset that is, so Tableau
# can show either number without needing its own calculated field.
def add_percentage_columns(df, dimension_cols):
    df = df.copy()
    total_rows = len(df)
    for col in dimension_cols:  # go through each dashboard column, to add its own count/percentage pair
        counts = df.groupby(col)[col].transform('count')  # count how many rows share this row's category
        df[f'{col}-count'] = counts
        df[f'{col}-pct'] = (counts / total_rows * 100).round(2)
    return df