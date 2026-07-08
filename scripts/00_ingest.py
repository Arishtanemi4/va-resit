"""Stage 0: Raw ingest & audit of the UCI Adult dataset.

Downloads adult.data / adult.test / adult.names, combines them into one
DataFrame, and audits row count, dtypes, missing-value sentinels and
duplicates. Writes data/adult_raw.csv and data/audit.txt.

Verify gate (CLAUDE.md Stage 0): row count == 48,842; missing counts printed.
"""

from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DATA_DIR = ROOT / "data"

BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult"
FILES = ["adult.data", "adult.test", "adult.names"]

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]

EXPECTED_ROWS = 48_842


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = RAW_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[skip] {name} already present ({dest.stat().st_size:,} bytes)")
            continue
        resp = requests.get(f"{BASE_URL}/{name}", timeout=60)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        print(f"[ok]   downloaded {name} ({len(resp.content):,} bytes)")


def load() -> pd.DataFrame:
    read_kwargs = dict(header=None, names=COLUMNS, skipinitialspace=True)
    train = pd.read_csv(RAW_DIR / "adult.data", **read_kwargs)
    # adult.test line 1 is '|1x3 Cross validator'; labels have trailing periods
    test = pd.read_csv(RAW_DIR / "adult.test", skiprows=1, **read_kwargs)
    test["income"] = test["income"].str.rstrip(".")
    df = pd.concat([train, test], ignore_index=True)
    for col in df.select_dtypes(include="str").columns:
        df[col] = df[col].str.strip()
    return df


def audit(df: pd.DataFrame) -> str:
    lines = []
    lines.append(f"rows: {len(df)} (expected {EXPECTED_ROWS})")
    lines.append(f"columns: {len(df.columns)} = {list(df.columns)}")
    lines.append("\ndtypes:\n" + df.dtypes.to_string())

    missing = df.isin(["?"]).sum()
    lines.append("\n'?' sentinel counts per column:\n" + missing.to_string())

    # Structural vs. ordinary missingness (CLAUDE.md §1)
    wc_q = df["workclass"] == "?"
    occ_q = df["occupation"] == "?"
    nc_q = df["native_country"] == "?"
    never_worked = df["workclass"] == "Never-worked"
    lines.append(
        "\nmissingness structure:"
        f"\n  workclass == '?':                     {wc_q.sum()}"
        f"\n  occupation == '?':                    {occ_q.sum()}"
        f"\n  occupation '?' AND workclass '?':     {(occ_q & wc_q).sum()} (structural: co-missing)"
        f"\n  occupation '?' AND Never-worked:      {(occ_q & never_worked).sum()} (structural: no occupation exists)"
        f"\n  occupation '?' AND workclass observed:{(occ_q & ~wc_q & ~never_worked).sum()}"
        f"\n  native_country == '?':                {nc_q.sum()} (ordinary missingness)"
    )

    dupes = df.duplicated().sum()
    lines.append(f"\nfull-row duplicates: {dupes} (kept: plausible identical census records)")

    lines.append(f"\nincome label counts:\n{df['income'].value_counts().to_string()}")
    edu_pairs = df.groupby("education_num")["education"].nunique()
    lines.append(
        "\neducation vs education_num: 1:1 mapping = "
        f"{bool((edu_pairs == 1).all())} (redundant; education_num kept, education "
        "retained only as tooltip text)"
    )
    return "\n".join(lines)


def main() -> None:
    download()
    df = load()
    report = audit(df)
    print("\n" + report)

    assert len(df) == EXPECTED_ROWS, f"row count {len(df)} != {EXPECTED_ROWS}"
    assert df["income"].isin(["<=50K", ">50K"]).all(), "unexpected income labels"

    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(DATA_DIR / "adult_raw.csv", index=True, index_label="index")
    (DATA_DIR / "audit.txt").write_text(report, encoding="utf-8")
    print(f"\n[ok] wrote {DATA_DIR / 'adult_raw.csv'} and {DATA_DIR / 'audit.txt'}")


if __name__ == "__main__":
    main()
