"""Stage 6 (packaging): bundle adult_dashboard.twb + CSVs into a .twbx.

A .twbx is a plain zip: the .twb at the archive root plus its data files at
the paths the .twb's textscan connections reference (Data/...). Run after the
.twb is finalized. Verify gate remains manual: the .twbx must open in Tableau
Desktop with all data sources resolving.
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TWB = ROOT / "adult_dashboard.twb"
CSVS = ["adult_pca_2d.csv", "adult_tsne_2d.csv", "adult_imputed.csv"]
OUT = ROOT / "adult_dashboard.twbx"


def main():
    assert TWB.exists(), f"{TWB} missing — build the workbook first"
    twb_text = TWB.read_text(encoding="utf-8")
    for csv in CSVS:
        assert (ROOT / "data" / csv).exists(), f"data/{csv} missing"
        assert csv in twb_text, f"{csv} not referenced by the .twb"

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(TWB, TWB.name)
        for csv in CSVS:
            z.write(ROOT / "data" / csv, f"Data/{csv}")
    print(f"[ok] wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB) "
          f"containing {TWB.name} + {len(CSVS)} CSVs under Data/")


if __name__ == "__main__":
    main()
