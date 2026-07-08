"""Stage 1: Cleaning & Bayesian imputation.

Imputes `workclass` then `occupation` (sequentially — imputed workclass feeds
the occupation model) with Bayesian multinomial logistic regression sampled
via NUTS (nutpie backend; no g++ in this env, so PyTensor's C backend is
unavailable). `native_country` '?' becomes an explicit 'Unknown' category
(857 rows, 41-level nominal — a 41-class Bayesian model is disproportionate
and the column is not a projection input). The 10 `Never-worked` rows get
occupation 'None (never worked)' directly: structurally, no occupation exists.

CLAUDE.md §3 phrases imputation for a continuous target ("posterior mean /
posterior std"). Categorical adaptation, flagged in the report: we impute the
posterior-predictive modal category and store the modal probability plus the
predictive entropy as uncertainty columns.

Models fit on a stratified subsample of observed rows (FIT_N per model) to
keep NUTS tractable; predictions cover every missing row. Flagged in report.

Verify gate: zero '?' remaining; R-hat <= 1.05 (hard flag otherwise);
imputed distribution printed against observed marginal and mode baseline.
Output: data/adult_imputed.csv, data/impute_summary.txt.
"""

import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

SEED = 42
DRAWS, TUNE, CHAINS = 1000, 500, 4
FIT_N = 8000          # stratified fit subsample per model
MIN_PER_CLASS = 15    # top-up floor so tiny classes aren't lost
RHAT_LIMIT = 1.05
NEVER_WORKED_OCC = "None (never worked)"

MARRIED = {"Married-civ-spouse", "Married-AF-spouse"}
PREVIOUSLY = {"Divorced", "Separated", "Widowed", "Married-spouse-absent"}

log_lines = []


def log(msg=""):
    print(msg)
    log_lines.append(str(msg))


def design_matrix(df, workclass_dummies=None):
    """Predictors: standardized age/education_num/hours_per_week, sex,
    collapsed marital status; optionally one-hot workclass (ref: Private)."""
    cols = {}
    for c in ["age", "education_num", "hours_per_week"]:
        cols[c] = (df[c] - MEANS[c]) / STDS[c]
    cols["sex_male"] = (df["sex"] == "Male").astype(float)
    cols["marital_married"] = df["marital_status"].isin(MARRIED).astype(float)
    cols["marital_prev"] = df["marital_status"].isin(PREVIOUSLY).astype(float)
    if workclass_dummies is not None:
        for wc in workclass_dummies:  # reference class: Private
            cols[f"wc_{wc}"] = (df["workclass"] == wc).astype(float)
    X = pd.DataFrame(cols, index=df.index)
    X.insert(0, "intercept", 1.0)
    return X


def stratified_fit_sample(df, target, n, rng):
    frac = n / len(df)
    parts = []
    for _, grp in df.groupby(target):
        take = max(int(round(frac * len(grp))), min(len(grp), MIN_PER_CLASS))
        parts.append(grp.sample(n=take, random_state=rng.integers(2**31)))
    return pd.concat(parts)


def fit_multinomial(name, X, y_codes, n_classes):
    """Softmax regression, reference class 0 pinned to zero logits."""
    with pm.Model():
        B = pm.Normal("B", 0.0, 2.5, shape=(X.shape[1], n_classes - 1))
        B_full = pt.concatenate([pt.zeros((X.shape[1], 1)), B], axis=1)
        pm.Categorical("y", logit_p=pt.dot(pt.as_tensor(X), B_full), observed=y_codes)
        idata = pm.sample(
            draws=DRAWS, tune=TUNE, chains=CHAINS, random_seed=SEED,
            nuts_sampler="nutpie", progressbar=False,
        )
    summ = az.summary(idata, var_names=["B"])
    max_rhat = float(summ["r_hat"].max())
    min_ess = float(summ["ess_bulk"].min())
    log(f"[{name}] az.summary: {len(summ)} params, max R-hat = {max_rhat:.4f}, "
        f"min ESS_bulk = {min_ess:.0f}")
    if max_rhat > RHAT_LIMIT:
        log(f"[{name}] *** CONVERGENCE FLAG: max R-hat {max_rhat:.4f} > {RHAT_LIMIT} ***")
        sys.exit(f"R-hat limit exceeded for {name}; not proceeding (CLAUDE.md §3).")
    return idata, summ


def predict_probs(idata, X_miss, n_classes, thin_to=400):
    B = idata.posterior["B"].stack(s=("chain", "draw")).transpose("s", ...).values
    step = max(1, B.shape[0] // thin_to)
    B = B[::step]                                     # (S, P, K-1)
    B = np.concatenate([np.zeros((B.shape[0], B.shape[1], 1)), B], axis=2)
    logits = np.einsum("mp,spk->smk", X_miss, B)      # (S, M, K)
    logits -= logits.max(axis=2, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=2, keepdims=True)
    return p.mean(axis=0)                             # (M, K)


def impute_column(df, target, fit_pool, miss_mask, wc_dummies, rng):
    classes = sorted(fit_pool[target].unique())
    log(f"\n=== {target}: {miss_mask.sum()} missing, {len(fit_pool)} observed, "
        f"{len(classes)} classes ===")
    fit_df = stratified_fit_sample(fit_pool, target, FIT_N, rng)
    log(f"[{target}] fit subsample: {len(fit_df)} rows (stratified, floor "
        f"{MIN_PER_CLASS}/class)")

    X_fit = design_matrix(fit_df, wc_dummies).values
    y_fit = fit_df[target].map({c: i for i, c in enumerate(classes)}).values
    idata, summ = fit_multinomial(target, X_fit, y_fit, len(classes))

    X_miss = design_matrix(df.loc[miss_mask], wc_dummies).values
    probs = predict_probs(idata, X_miss, len(classes))
    modal_idx = probs.argmax(axis=1)
    modal = np.array(classes)[modal_idx]
    modal_p = probs[np.arange(len(probs)), modal_idx]
    entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1)

    df.loc[miss_mask, target] = modal
    df.loc[miss_mask, f"{target}_imputed_prob"] = np.round(modal_p, 4)
    df.loc[miss_mask, f"{target}_entropy"] = np.round(entropy, 4)
    log(f"[{target}] imputed modal prob: mean {modal_p.mean():.3f}, "
        f"min {modal_p.min():.3f}; entropy mean {entropy.mean():.3f} nats "
        f"(max possible {np.log(len(classes)):.3f})")
    return summ, classes


def distribution_qa(df, target, miss_mask, observed_marginal):
    imputed = df.loc[miss_mask, target].value_counts(normalize=True)
    qa = pd.DataFrame({"observed_%": observed_marginal * 100,
                       "imputed_%": imputed * 100}).fillna(0).round(2)
    mode = observed_marginal.idxmax()
    log(f"\n[{target}] imputed vs observed marginal (mode baseline would put "
        f"100% on '{mode}'):\n{qa.to_string()}")


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(DATA_DIR / "adult_raw.csv", index_col="index")

    global MEANS, STDS
    MEANS = df[["age", "education_num", "hours_per_week"]].mean()
    STDS = df[["age", "education_num", "hours_per_week"]].std()

    df["native_country_was_unknown"] = df["native_country"] == "?"
    df["native_country"] = df["native_country"].replace("?", "Unknown")
    log(f"native_country: {df['native_country_was_unknown'].sum()} '?' -> 'Unknown'")

    wc_miss = df["workclass"] == "?"
    never_worked = df["workclass"] == "Never-worked"
    occ_miss = df["occupation"] == "?"
    df["workclass_was_imputed"] = wc_miss
    df["occupation_was_imputed"] = occ_miss & ~never_worked
    for c in ["workclass_imputed_prob", "workclass_entropy",
              "occupation_imputed_prob", "occupation_entropy"]:
        df[c] = np.nan

    # structural: Never-worked rows have no occupation to impute
    df.loc[never_worked, "occupation"] = NEVER_WORKED_OCC
    log(f"occupation: {never_worked.sum()} Never-worked rows set to "
        f"'{NEVER_WORKED_OCC}' (structural)")

    wc_marginal = df.loc[~wc_miss, "workclass"].value_counts(normalize=True)
    wc_summ, _ = impute_column(df, "workclass", df.loc[~wc_miss], wc_miss,
                               None, rng)
    distribution_qa(df, "workclass", wc_miss, wc_marginal)

    # occupation model uses workclass (now fully observed/imputed) as predictor
    occ_pool = df.loc[~occ_miss & (df["occupation"] != NEVER_WORKED_OCC)]
    wc_dummies = [c for c in sorted(df["workclass"].unique()) if c != "Private"]
    occ_marginal = occ_pool["occupation"].value_counts(normalize=True)
    occ_summ, _ = impute_column(df, "occupation", occ_pool,
                                occ_miss & ~never_worked, wc_dummies, rng)
    distribution_qa(df, "occupation", occ_miss & ~never_worked, occ_marginal)

    # mode baseline for the Bayesian_Imputation_QA sheet (fallback comparison only)
    df["workclass_mode_baseline"] = df["workclass"]
    df.loc[wc_miss, "workclass_mode_baseline"] = wc_marginal.idxmax()
    df["occupation_mode_baseline"] = df["occupation"]
    df.loc[occ_miss & ~never_worked, "occupation_mode_baseline"] = occ_marginal.idxmax()

    remaining = int(df.isin(["?"]).sum().sum())
    assert remaining == 0, f"'?' sentinels remaining: {remaining}"
    assert len(df) == 48_842
    log(f"\n[ok] zero '?' remaining; {len(df)} rows")

    out = DATA_DIR / "adult_imputed.csv"
    df.to_csv(out, index=True, index_label="index")
    log(f"[ok] wrote {out}")

    with open(DATA_DIR / "impute_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
        f.write(f"\n\nsampler: nutpie NUTS, draws={DRAWS}, tune={TUNE}, "
                f"chains={CHAINS}, seed={SEED}\n")
        f.write("\naz.summary — workclass model:\n" + wc_summ.to_string())
        f.write("\n\naz.summary — occupation model:\n" + occ_summ.to_string())
    print(f"[ok] wrote {DATA_DIR / 'impute_summary.txt'}")


if __name__ == "__main__":
    main()
