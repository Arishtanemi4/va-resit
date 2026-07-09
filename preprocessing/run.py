from pathlib import Path

import pandas as pd

from preprocess import *
from dim_reduce import *


# Runs every pipeline step in order, from the raw census file to the
# finished, Tableau-ready CSVs.
def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / 'data' / 'raw' / 'adult.data'
    test_path = base_dir / 'data' / 'raw' / 'adult.test'
    processed_dir = base_dir / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / 'adult_tableau.csv'

    df_raw = load_and_clean_anomalies(str(data_path), str(test_path))

    X_ordinal, category_maps = encode_ordinal_categoricals(df_raw)

    X_ordinal_imputed = impute_missing_data_bayesian(X_ordinal)

    df_imputed = decode_imputed_categoricals(df_raw, X_ordinal_imputed, category_maps)

    X_for_pca, _ = encode_ordinal_categoricals(df_imputed)  # re-encode the now-complete data, to build the PCA/t-SNE input matrix

    X_scaled = scale_data(X_for_pca)

    pca_df, loadings_df = apply_pca(X_scaled)
    tsne_df = apply_tsne(X_scaled)

    df_final_cols = add_age_group(df_imputed)
    df_final_cols = add_percentage_columns(df_final_cols, ['education', 'occupation', 'age-group', 'income'])
    df_final_cols = add_imputed_flags(df_final_cols, df_raw)  # mark which rows were filled by the Bayesian imputer, for the QA panel

    final_df = pd.concat([df_final_cols.reset_index(drop=True), pca_df, tsne_df], axis=1)
    loadings_df.to_csv(processed_dir / 'pca_loadings.csv', index=True)

    final_df.to_csv(out_path, index=False)
    print(f"Preprocessed data saved to '{out_path}'.")


if __name__ == "__main__":
    main()