from pathlib import Path

import pandas as pd

from preprocess import *
from dim_reduce import *


def main():
    base_dir = Path(__file__).resolve().parent.parent
    raw_path = base_dir / 'data' / 'raw' / 'adult.data'
    processed_dir = base_dir / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / 'adult_tableau.csv'

    df_raw = load_and_clean_anomalies(str(raw_path))

    df_reduced = drop_redundant_features(df_raw)

    X_enc, y_enc = encode_categorical_data(df_reduced)

    X_imputed = impute_missing_data_bayesian(X_enc)

    X_scaled = scale_data(X_imputed)

    pca_df, loadings_df = apply_pca(X_scaled)
    tsne_df = apply_tsne(X_scaled)

    final_df = pd.concat([df_reduced.reset_index(drop=True), pca_df, tsne_df], axis=1)

    final_df.to_csv(out_path, index=False)
    print(f"Preprocessed data saved to '{out_path}'.")


if __name__ == "__main__":
    main()