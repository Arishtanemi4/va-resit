from pathlib import Path

import pandas as pd

from preprocess import *
from dim_reduce import *


# Runs every pipeline step in order, from the raw census file to the
# finished, Tableau-ready CSVs.
def main():
    base_dir = Path(__file__).resolve().parent.parent
    raw_path = base_dir / 'data' / 'raw' / 'adult.data'
    processed_dir = base_dir / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / 'adult_tableau.csv'

    df_raw = load_and_clean_anomalies(str(raw_path))

    df_reduced = drop_redundant_features(df_raw)

    X_ordinal, category_maps = encode_ordinal_categoricals(df_reduced)

    X_ordinal_imputed = impute_missing_data_bayesian(X_ordinal)

    df_imputed = decode_imputed_categoricals(df_reduced, X_ordinal_imputed, category_maps)

    X_enc, y_enc = encode_categorical_data(df_imputed)

    X_scaled = scale_data(X_enc)

    pca_df, loadings_df = apply_pca(X_scaled)
    tsne_df = apply_tsne(X_scaled)

    df_final_cols = add_age_group(df_imputed)
    df_final_cols = add_percentage_columns(df_final_cols, ['education-num', 'occupation', 'age-group', 'income'])

    final_df = pd.concat([df_final_cols.reset_index(drop=True), pca_df, tsne_df], axis=1)
    loadings_df.to_csv(processed_dir / 'pca_loadings.csv', index=True)

    final_df.to_csv(out_path, index=False)
    print(f"Preprocessed data saved to '{out_path}'.")


if __name__ == "__main__":
    main()