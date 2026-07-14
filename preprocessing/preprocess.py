import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler



# Reads BOTH raw census splits (adult.data + adult.test) and stacks them
# into one table, swaps the '?' placeholders for real missing values, and
# drops two columns: 'education-num' (a numeric restatement of 'education',
# checked below, not assumed — we keep the readable 'education' label since
# it's more useful for Tableau) and 'fnlwgt' (a Census survey sampling
# weight, not a person's attribute — see docs/REPLAN.md §9).
def load_and_clean_anomalies(data_path, test_path):
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]

    df_data = pd.read_csv(data_path, header=None, names=column_names, skipinitialspace=True)
    df_test = pd.read_csv(test_path, header=None, names=column_names, skipinitialspace=True, skiprows=1)  # skip the '|1x3 Cross validator' junk header line
    df_data['split'] = 'train'  # remember which raw file each row came from, so the forecast step can train on adult.data and predict on the held-out adult.test rows
    df_test['split'] = 'test'
    df = pd.concat([df_data, df_test], ignore_index=True)  # stack the two splits into one table, renumbering the rows

    df['income'] = df['income'].str.rstrip('.')  # adult.test writes income as '<=50K.'/'>50K.' — strip the trailing dot so both splits share one label

    df.replace('?', np.nan, inplace=True)

    print("--- Missing Values Count per Column ---")
    print(df.isnull().sum())  # count blanks per column, then just print those counts

    labels_per_code = df.groupby('education-num')['education'].nunique()  # count how many different education labels share each education-num code
    if not (labels_per_code == 1).all():
        raise ValueError("education-num no longer matches education 1-to-1, so it can't be safely dropped")
    df = df.drop(columns=['education-num', 'fnlwgt'])

    return df


# Converts each category into a single number code (blanks stay blank) so
# the imputer below can treat a missing category like a missing number.
def encode_ordinal_categoricals(df):
    X = df.drop(columns=['income', 'split'])  # 'split' is a pipeline bookkeeping column (which raw file a row came from), not a person's attribute — keep it out of the imputation/PCA/t-SNE feature matrix, same as 'income'
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


# Adds a True/False column for each field that had missing values, marking
# which rows the Bayesian imputer filled in, so the dashboard can show the
# imputation's footprint even though the exported values are now complete.
def add_imputed_flags(df, df_raw):
    df = df.copy()
    missing_cols = df_raw.columns[df_raw.isna().any()]  # list only the columns that actually had blanks in the raw data
    for col in missing_cols:  # go through each of those columns, to flag its imputed rows
        df[f'{col}-imputed'] = df_raw[col].isna().values  # a row is 'imputed' wherever the raw value was blank; .values keeps it aligned by position
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