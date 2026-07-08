"""Stages 3-4: PCA and t-SNE projections.

PCA (sklearn): prints explained variance ratios, saves figures/scree.png.
t-SNE (sklearn Barnes-Hut): full 48,842 rows, perplexity/max_iter/seed
logged to params.json.

Both exports carry the CLAUDE.md §4 required columns:
index, <2D coords>, income, age, education, occupation, hours_per_week, sex,
variables_used.

Verify gate: CSVs contain all required columns; explained variance printed;
params.json written.
"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

SEED = 42
PERPLEXITY = 30
MAX_ITER = 1000

TOOLTIP_COLS = ["income", "age", "education", "occupation", "hours_per_week", "sex"]


def export(coords, names, tooltips, variables_used, path):
    out = pd.DataFrame(coords.round(4), columns=names, index=tooltips.index)
    out = out.join(tooltips)
    out["variables_used"] = variables_used
    out.to_csv(path, index=True, index_label="index")
    required = ["index"] + names + TOOLTIP_COLS + ["variables_used"]
    have = ["index"] + list(out.columns)
    missing = [c for c in required if c not in have]
    assert not missing, f"missing required columns in {path.name}: {missing}"
    print(f"[ok] wrote {path} ({len(out)} rows, cols: {have})")


def main():
    feats = pd.read_csv(DATA_DIR / "features.csv", index_col="index")
    variables_used = feats.pop("variables_used").iloc[0]
    df = pd.read_csv(DATA_DIR / "adult_imputed.csv", index_col="index")
    tooltips = df[TOOLTIP_COLS]
    X = feats.values
    print(f"input: {X.shape}, variables_used: {variables_used}")

    # Stage 3: PCA
    pca = PCA(n_components=X.shape[1], random_state=SEED)
    pcs = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_
    print(f"PCA explained variance ratio: PC1={evr[0]:.4f}, PC2={evr[1]:.4f} "
          f"(PC1+PC2={evr[:2].sum():.4f}); full: {[round(v, 4) for v in evr]}")

    FIG_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(1, len(evr) + 1), evr, color="#4C78A8")
    ax.plot(range(1, len(evr) + 1), evr.cumsum(), "o-", color="#F58518",
            label="cumulative")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA scree plot — Adult dataset (6 standardized features)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "scree.png", dpi=150)
    print(f"[ok] wrote {FIG_DIR / 'scree.png'}")

    export(pcs[:, :2], ["PC1", "PC2"], tooltips, variables_used,
           DATA_DIR / "adult_pca_2d.csv")

    # Stage 4: t-SNE (Barnes-Hut), full dataset
    t0 = time.time()
    tsne = TSNE(n_components=2, perplexity=PERPLEXITY, max_iter=MAX_ITER,
                init="pca", random_state=SEED, verbose=1)
    ts = tsne.fit_transform(X)
    elapsed = time.time() - t0
    print(f"t-SNE done in {elapsed:.0f}s, KL divergence: {tsne.kl_divergence_:.3f}")

    export(ts, ["TSNE1", "TSNE2"], tooltips, variables_used,
           DATA_DIR / "adult_tsne_2d.csv")

    params = {
        "features": variables_used.split(", "),
        "n_rows": int(X.shape[0]),
        "pca": {
            "library": "sklearn.decomposition.PCA",
            "n_components": int(X.shape[1]),
            "explained_variance_ratio": [round(float(v), 4) for v in evr],
            "random_state": SEED,
        },
        "tsne": {
            "library": "sklearn.manifold.TSNE",
            "n_components": 2,
            "perplexity": PERPLEXITY,
            "max_iter": MAX_ITER,
            "init": "pca",
            "random_state": SEED,
            "kl_divergence": round(float(tsne.kl_divergence_), 3),
            "runtime_seconds": round(elapsed),
        },
    }
    (ROOT / "params.json").write_text(json.dumps(params, indent=2),
                                      encoding="utf-8")
    print(f"[ok] wrote {ROOT / 'params.json'}")


if __name__ == "__main__":
    main()
