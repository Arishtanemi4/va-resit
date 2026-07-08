"""Stage 2: Feature encoding for projections.

Builds the numeric feature matrix used by BOTH projections (PCA and t-SNE
share one feature set for comparability): standardized age, education_num,
hours_per_week, log1p(capital_gain), log1p(capital_loss), plus binary sex.

Excluded, per report justification: `income` (the label being verified
against), `fnlwgt` (survey design weight, not a person attribute),
`education` (redundant text form of education_num), and the high-cardinality
nominals (occupation/workclass/native_country etc.) which are surfaced via
tooltips instead of being one-hot inflated into the distance space.

Verify gate: all numeric, zero NaNs, shape printed.
Output: data/features.csv (index + 6 features + variables_used string).
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

FEATURES = ["age", "education_num", "hours_per_week",
            "capital_gain_log", "capital_loss_log", "sex_male"]


def main():
    df = pd.read_csv(DATA_DIR / "adult_imputed.csv", index_col="index")

    X = pd.DataFrame(index=df.index)
    X["age"] = df["age"]
    X["education_num"] = df["education_num"]
    X["hours_per_week"] = df["hours_per_week"]
    X["capital_gain_log"] = np.log1p(df["capital_gain"])
    X["capital_loss_log"] = np.log1p(df["capital_loss"])
    X["sex_male"] = (df["sex"] == "Male").astype(float)

    X = (X - X.mean()) / X.std()

    assert list(X.columns) == FEATURES
    assert X.notna().all().all(), "NaNs in feature matrix"
    assert X.shape == (48_842, 6), f"unexpected shape {X.shape}"
    print(f"feature matrix shape: {X.shape}")
    print(f"features (standardized): {FEATURES}")
    print(X.describe().loc[["mean", "std", "min", "max"]].round(3).to_string())

    X["variables_used"] = ", ".join(FEATURES)
    X.to_csv(DATA_DIR / "features.csv", index=True, index_label="index")
    print(f"[ok] wrote {DATA_DIR / 'features.csv'}")


if __name__ == "__main__":
    main()
