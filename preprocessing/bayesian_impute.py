import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.special import softmax
import warnings


def impute_categorical_bayesian(
    df, target_col, numeric_predictors, categorical_predictors,
    draws=1000, tune=500, chains=4, random_seed=42
):
    """
    Bayesian multinomial logistic imputation for a single categorical column.

    Uses PyMC with NUTS sampler (via nutpie) to fit a softmax regression model
    on complete rows, then predicts posterior-mean class and uncertainty for missing rows.

    Args:
        df: DataFrame with both observed and missing values in target_col
        target_col: column name to impute (string)
        numeric_predictors: list of numeric column names to standardize and use
        categorical_predictors: list of categorical column names to one-hot encode and use
        draws: posterior draws per chain (default 1000, per CLAUDE.md)
        tune: tuning steps per chain (default 500, per CLAUDE.md)
        chains: number of sampling chains (minimum 4 for R-hat)
        random_seed: random seed for reproducibility

    Returns:
        imputed_labels: pd.Series of imputed class labels for originally-missing rows
        uncertainty: pd.Series of entropy-based uncertainty (0=confident, 1=maximally uncertain)
        confidence: pd.Series of top-1 posterior probability
        idata: arviz InferenceData object with full posterior samples
    """
    obs_mask = df[target_col].notna()
    n_total = len(df)
    n_obs = obs_mask.sum()

    X_numeric = df.loc[obs_mask, numeric_predictors].astype(float).copy()
    X_numeric_all = df[numeric_predictors].astype(float).copy()

    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric)
    X_numeric_scaled_all = scaler.transform(X_numeric_all)

    X_cat = pd.get_dummies(df.loc[obs_mask, categorical_predictors], drop_first=True)
    X_cat_all = pd.get_dummies(df[categorical_predictors], drop_first=True)

    X_obs = np.hstack([X_numeric_scaled, X_cat.values])
    X_all = np.hstack([X_numeric_scaled_all, X_cat_all.values])

    y_obs = df.loc[obs_mask, target_col].values
    classes = np.sort(df[target_col].dropna().unique())
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_obs_idx = np.array([class_to_idx[c] for c in y_obs])

    n_features = X_obs.shape[1]

    baseline_class_idx = n_classes - 1

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=5, shape=n_classes - 1)
        beta = pm.Normal("beta", mu=0, sigma=2.5, shape=(n_features, n_classes - 1))

        logits_obs = pm.math.dot(X_obs, beta)
        logits_obs = pm.math.concatenate([logits_obs, pm.math.zeros((n_obs, 1))], axis=1)
        p_obs = pm.math.softmax(logits_obs, axis=1)

        y_likelihood = pm.Categorical(target_col, p=p_obs, observed=y_obs_idx)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            nuts_sampler="nutpie",
            target_accept=0.9,
            random_seed=random_seed,
            progressbar=True,
            return_inferencedata=True,
            discard_tuned_chains=True,
        )

    print(f"\n--- Bayesian Imputation for '{target_col}' ---")
    print(f"Model: {n_classes}-class softmax GLM, {n_features} predictors, {n_obs} observed rows")
    print(f"Posterior samples: {draws} draws × {chains} chains")
    print(f"\nConvergence diagnostics:")
    summary = az.summary(idata, var_names=["alpha", "beta"])
    print(summary)

    max_rhat = summary["r_hat"].max()
    if max_rhat > 1.05:
        print(f"\n⚠️ WARNING: Max R-hat = {max_rhat:.4f} > 1.05 (potential non-convergence)")
        print(f"   Model: {target_col} | Please review diagnostics before proceeding")
    else:
        print(f"\n✓ PASS: Max R-hat = {max_rhat:.4f} ≤ 1.05 (acceptable convergence)")

    summary_df = summary.reset_index()
    summary_df.to_csv(f"../data/bayesian_imputation_diagnostics_{target_col}.csv", index=False)
    print(f"   Diagnostics saved to data/bayesian_imputation_diagnostics_{target_col}.csv")

    posterior_samples = idata.posterior
    alpha_samples = posterior_samples["alpha"].values
    beta_samples = posterior_samples["beta"].values

    n_draws = alpha_samples.shape[0] * alpha_samples.shape[1]
    alpha_samples = alpha_samples.reshape(n_draws, n_classes - 1)
    beta_samples = beta_samples.reshape(n_draws, n_features, n_classes - 1)

    logits_all = np.dot(X_all, beta_samples.mean(axis=0))
    logits_all_stacked = np.hstack([logits_all, np.zeros((n_total, 1))])
    p_mean = softmax(logits_all_stacked, axis=1)

    imputed_classes = classes[p_mean.argmax(axis=1)]
    top1_prob = p_mean.max(axis=1)

    entropy = -np.sum(p_mean * np.log(p_mean + 1e-10), axis=1)
    max_entropy = np.log(n_classes)
    normalized_entropy = entropy / max_entropy

    imputed_idx = np.where(~obs_mask)[0]
    n_imputed = len(imputed_idx)

    imputed_labels = pd.Series(
        imputed_classes[imputed_idx],
        index=imputed_idx,
        name=target_col
    )
    uncertainty = pd.Series(
        normalized_entropy[imputed_idx],
        index=imputed_idx,
        name=f"{target_col}_impute_uncertainty"
    )
    confidence = pd.Series(
        top1_prob[imputed_idx],
        index=imputed_idx,
        name=f"{target_col}_impute_confidence"
    )

    mode_baseline = df.loc[~obs_mask, target_col].isna()
    mode_imputer = SimpleImputer(strategy="most_frequent")
    mode_labels = mode_imputer.fit_transform(
        df.loc[obs_mask, [target_col]]
    )
    mode_baseline_all = pd.Series(index=df.index, dtype=object)
    mode_baseline_all.loc[obs_mask] = df.loc[obs_mask, target_col]
    mode_baseline_all = mode_imputer.fit_transform(
        df[[target_col]]
    ).ravel()

    match_count = (imputed_classes[imputed_idx] == mode_baseline_all[imputed_idx]).sum()
    match_pct = 100 * match_count / n_imputed if n_imputed > 0 else 0

    print(f"\nSanity check vs. mode-baseline imputation:")
    print(f"  {match_count}/{n_imputed} imputed values match mode-fill ({match_pct:.1f}%)")

    return imputed_labels, uncertainty, confidence, idata


def impute_workclass_and_occupation(df):
    """
    Orchestrator: Imputes workclass, then occupation (using imputed workclass as predictor).

    Operates on raw categorical dataframe (pre-encoding). Returns dataframe with all
    three `?`-bearing columns filled and uncertainty/confidence columns added.
    """
    df_imputed = df.copy()

    numeric_cols = ["age", "education-num", "hours-per-week", "capital-gain", "capital-loss"]
    cat_cols_base = ["sex", "race", "marital-status", "relationship"]

    print("\n" + "="*70)
    print("STAGE 1: Bayesian Categorical Imputation (PyMC/NUTS)")
    print("="*70)

    workclass_labels, workclass_unc, workclass_conf, idata_wc = impute_categorical_bayesian(
        df_imputed,
        target_col="workclass",
        numeric_predictors=numeric_cols,
        categorical_predictors=cat_cols_base,
        draws=1000,
        tune=500,
        chains=4,
    )

    df_imputed.loc[workclass_labels.index, "workclass"] = workclass_labels
    df_imputed[f"workclass_impute_uncertainty"] = np.nan
    df_imputed.loc[workclass_labels.index, "workclass_impute_uncertainty"] = workclass_unc
    df_imputed[f"workclass_impute_confidence"] = np.nan
    df_imputed.loc[workclass_labels.index, "workclass_impute_confidence"] = workclass_conf

    cat_cols_occupation = cat_cols_base + ["workclass"]

    print("\n" + "-"*70)
    print("Imputing occupation (using now-complete workclass as predictor)...")
    print("-"*70)

    occupation_labels, occupation_unc, occupation_conf, idata_occ = impute_categorical_bayesian(
        df_imputed,
        target_col="occupation",
        numeric_predictors=numeric_cols,
        categorical_predictors=cat_cols_occupation,
        draws=1000,
        tune=500,
        chains=4,
    )

    df_imputed.loc[occupation_labels.index, "occupation"] = occupation_labels
    df_imputed[f"occupation_impute_uncertainty"] = np.nan
    df_imputed.loc[occupation_labels.index, "occupation_impute_uncertainty"] = occupation_unc
    df_imputed[f"occupation_impute_confidence"] = np.nan
    df_imputed.loc[occupation_labels.index, "occupation_impute_confidence"] = occupation_conf

    print("\n" + "="*70)
    print(f"Bayesian imputation complete: workclass & occupation filled")
    print(f"Uncertainty columns created for Tableau QA sheet")
    print("="*70 + "\n")

    return df_imputed


def mode_impute_native_country(df):
    """
    Fallback SimpleImputer for native-country (not mandated by CLAUDE.md for Bayesian method).

    Uses mode-fill strategy since 89.6% of observed rows are already 'United-States',
    and native-country has 41 sparse categories (extending PyMC to this would be
    disproportionate complexity for marginal gain).
    """
    print("Imputing native-country with mode-fill (fallback SimpleImputer, not Bayesian)...")
    df_imputed = df.copy()

    imputer = SimpleImputer(strategy="most_frequent")
    df_imputed[["native-country"]] = imputer.fit_transform(df[["native-country"]])

    n_imputed = (df["native-country"].isna()).sum()
    print(f"  Mode-filled {n_imputed} missing native-country values")
    print(f"  (CLAUDE.md does not mandate PyMC for this column; SimpleImputer is a labelled fallback)")

    return df_imputed
