import numpy as np
import pandas as pd
import pymc as pm

AGE_GROUP_ORDER = ['<20', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']  # same life-cycle order used throughout the dashboard (preprocess.py::add_age_group)


# Fits a Bayesian logistic regression on the training rows to predict whether
# income is above $50K, then predicts on the held-out test rows so the
# dashboard can compare a genuine prediction against data the model never saw.
def fit_income_forecast(X_train, y_train, X_test, test_meta):
    coords = {"feature": list(X_train.columns)}
    with pm.Model(coords=coords) as model:
        X_data = pm.Data("X_data", X_train.values, mutable=True)
        intercept = pm.Normal("intercept", mu=0, sigma=2)
        coefs = pm.Normal("coefs", mu=0, sigma=1, dims="feature")
        logits = intercept + pm.math.dot(X_data, coefs)
        p = pm.Deterministic("p", pm.math.sigmoid(logits))
        pm.Bernoulli("y_obs", p=p, observed=y_train.values)

        trace = pm.sample(1000, tune=1000, chains=2, random_seed=42, nuts_sampler="nutpie", progressbar=False)

    # Swap in the held-out test rows, then sample the model's predicted
    # probability for each one from the posterior it just learned.
    with model:
        pm.set_data({"X_data": X_test.values})
        posterior_pred = pm.sample_posterior_predictive(
            trace, var_names=["p"], predictions=True, random_seed=42, progressbar=False
        )

    # (chain, draw, n_test_rows) -> (n_draws, n_test_rows): one predicted probability per test row per posterior draw
    p_draws = posterior_pred.predictions["p"].stack(sample=("chain", "draw")).values.T

    return aggregate_by_age_group(p_draws, test_meta)


# Averages each posterior draw's predicted probabilities within an age group
# first, then takes quantiles across draws for that group's credible interval
# — this keeps the interval tied to the model's actual uncertainty about the
# group's rate, instead of just averaging pre-computed per-row intervals.
def aggregate_by_age_group(p_draws, test_meta):
    test_meta = test_meta.reset_index(drop=True)
    rows = []
    for age_group in AGE_GROUP_ORDER:  # go through each age bucket in life-cycle order, to summarize that group's rows
        mask = (test_meta['age-group'] == age_group).values
        if not mask.any():
            continue
        group_draws = p_draws[:, mask].mean(axis=1)  # this age group's mean predicted probability, one value per posterior draw
        actual = (test_meta.loc[mask, 'income'] == '>50K').mean()
        rows.append({
            'age-group': age_group,
            'actual_pct_above_50k': actual,
            'predicted_pct_above_50k': group_draws.mean(),
            'ci_lower': np.quantile(group_draws, 0.025),
            'ci_upper': np.quantile(group_draws, 0.975),
        })
    return pd.DataFrame(rows)
