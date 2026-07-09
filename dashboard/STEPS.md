# STEPS.md — Building `dashboard/2701553.twb` sheet by sheet

This is a build guide for the Tableau workbook, written against the data as it exists today in
`data/processed/`. It assumes Tableau Desktop is installed and `2701553.twb` is open.

---

## 0. Data footnotes — read before building

The exported CSV doesn't fully match the CLAUDE.md §4 spec yet. Rather than block on a pipeline
fix, every sheet below works around these gaps in Tableau. Fix upstream later if you want the
pipeline itself to be spec-compliant.

| Gap | Where it shows up | Workaround used below |
|---|---|---|
| No `variables_used` column | Proj_PCA / Proj_tSNE tooltips | Static calculated field (§1), since each projection was fit on one fixed feature set |
| No `index` column | — | Not needed — none of the 7 sheets require a row id |
| `workclass`/`occupation`/`native-country` still have blank cells | Any sheet using these fields | `IFNULL(..., "Unknown")` calculated fields (§1) |
| Only 32,561 of 48,842 rows (`adult.test` never merged) | All sheets | Known limitation — not fixable from Tableau, just be aware row counts will look low |
| Bayesian imputation only fed the PCA/t-SNE math, was never written back to the categorical columns | Bayesian_Imputation_QA | That sheet is redefined as a **missingness audit**, not a true imputed-vs-raw comparison — see §7 |
| `pca_loadings.csv` auto-connected to `adult_tableau.csv` as a union | Data Source tab | Split them apart — see §1 |

---

## 1. Prerequisite check (do this once)

1. Open `2701553.twb`, go to the **Data Source** tab.
2. Check whether `adult_tableau` and `pca_loadings` were combined as a union/collection (two
   connections stacked into one logical table). They have incompatible schemas — 18 columns of
   per-person data vs. 89 columns of PCA loading weights — so unioning them silently produces a
   couple of mostly-null garbage rows. **Split them into two separate connections.** None of the
   7 required sheets need `pca_loadings.csv`; leave it disconnected from the sheets (keep it
   around only if you later want a bonus loadings/biplot view).
3. Create these calculated fields once, up front (Data pane → right-click → Create Calculated
   Field). They're reused by name in the sheet steps below:

   - `Workclass (Clean)`
     ```
     IFNULL([Workclass], "Unknown")
     ```
   - `Occupation (Clean)`
     ```
     IFNULL([Occupation], "Unknown")
     ```
   - `Native Country (Clean)`
     ```
     IFNULL([Native Country], "Unknown")
     ```
   - `Variables Used (PCA)` — hardcoded string; this is the actual feature list read off
     `pca_loadings.csv`'s columns (everything except the label and the dropped `education`):
     ```
     "age, workclass, fnlwgt, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country"
     ```
   - `Variables Used (tSNE)` — same string (t-SNE was fit on the same encoded matrix as PCA):
     ```
     "age, workclass, fnlwgt, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country"
     ```
4. Create a parameter called `Show As` (Data pane → right-click → Create Parameter):
   - Data type: String
   - Allowable values: List → `Count`, `% of Total`
5. Create the calculated field `Income Measure` (this is the Abs/Proportion toggle logic — one
   calculated field reused across all three `Income_by_*` sheets, per CLAUDE.md §5):
   ```
   IF [Show As] = "Count" THEN
       COUNT([Income])
   ELSE
       COUNT([Income]) / TOTAL(COUNT([Income]))
   END
   ```
   Right-click it → **Default Table Calculation** → compute using the dimension you'll put on
   the opposite shelf (e.g. Occupation for Income_by_Occupation) so percentages sum to 100%
   within each chart, not across the whole table.

**Verify:** Data pane shows 3 "(Clean)" fields, 2 "Variables Used" fields, 1 parameter, 1
`Income Measure` field, all with no red exclamation/error icons.

---

## 2. Proj_PCA

1. New worksheet, rename to `Proj_PCA`.
2. Drag `PC1` to **Columns**, `PC2` to **Rows**. Both should be continuous (green) pills.
3. Mark type: **Circle**.
4. Drag `Income` to **Color**.
5. Drag onto **Tooltip**, in this order: `Age`, `Education-num`, `Occupation`, `Income`,
   `Variables Used (PCA)`. Edit the tooltip text so each field is labeled (e.g. "Age: <Age>").
6. Optional: add axis captions noting PC1/PC2 explained variance once `run.py` prints/logs it
   (not currently logged anywhere — see pipeline TODO, out of scope for this doc).

**Verify:** hover 5 random points — tooltip shows all five fields populated (Occupation may show
blank if the `Clean` version wasn't used; swap in `Occupation (Clean)` if so).

---

## 3. Proj_tSNE

1. New worksheet, rename to `Proj_tSNE`.
2. Drag `tSNE1` to **Columns**, `tSNE2` to **Rows** (continuous).
3. Mark type: **Circle**. Color: `Income`.
4. Tooltip: `Age`, `Education-num`, `Occupation (Clean)`, `Income`, `Variables Used (tSNE)`.
5. Add a text caption on the sheet (Worksheet → Show Caption, or a floating text object) noting
   the hyperparameters actually used: `perplexity=30, random_state=42`. `n_iter`/`max_iter` was
   never explicitly set in `preprocessing/dim_reduce.py` — flag this in the caption as a pipeline
   gap rather than inventing a value.

**Verify:** same hover check as Proj_PCA; caption text is visible when the sheet is viewed at
normal zoom.

---

## 4. Income_by_Education

1. New worksheet, rename to `Income_by_Education`.
2. Drag `Education-num` to **Columns** (treat as discrete — right-click the pill → Discrete).
3. Drag `Income Measure` to **Rows**.
4. Mark type: **Bar**.
5. Drag `Income` to **Color**.
6. Right-click the `Show As` parameter → **Show Parameter Control**, place it near the chart.

**Verify:** toggling `Show As` between "Count" and "% of Total" switches the bar heights from raw
counts to percentages that sum to 100% within each education-num group.

---

## 5. Income_by_Occupation

1. New worksheet, rename to `Income_by_Occupation`.
2. Drag `Occupation (Clean)` to **Rows**.
3. Drag `Income Measure` to **Columns**.
4. Mark type: **Bar**. Color: `Income`.
5. Show the `Show As` parameter control on this sheet too (each sheet needs its own visible
   control, or place one on the dashboard and hide it per-sheet — see §8).

**Verify:** an "Unknown" bar appears instead of a blank/null row; toggle works as in §4.

---

## 6. Income_by_Age

1. New worksheet, rename to `Income_by_Age`.
2. Create a calculated field `Age Group`:
   ```
   IF [Age] < 20 THEN "<20"
   ELSEIF [Age] < 30 THEN "20-29"
   ELSEIF [Age] < 40 THEN "30-39"
   ELSEIF [Age] < 50 THEN "40-49"
   ELSEIF [Age] < 60 THEN "50-59"
   ELSEIF [Age] < 70 THEN "60-69"
   ELSE "70+"
   END
   ```
3. Drag `Age Group` to **Columns**, `Income Measure` to **Rows**.
4. Mark type: **Bar**. Color: `Income`.
5. Show the `Show As` parameter control.

**Verify:** bars appear left-to-right in age order (`<20` … `70+`); if Tableau sorts them
alphabetically instead, right-click the `Age Group` axis → Sort → set manual order.

---

## 7. Bayesian_Imputation_QA

**Scope note:** the pipeline's Bayesian-ridge imputation currently only feeds the PCA/t-SNE
input matrix — it was never written back into the exported `workclass`/`occupation`/
`native-country` columns, so there is no "imputed value" column to compare against "raw value."
A true imputed-vs-raw QA view is **not buildable from the current CSV**. This sheet is
re-scoped as a **missingness audit** instead, until the pipeline is patched to export both.

1. New worksheet, rename to `Bayesian_Imputation_QA`.
2. Create calculated fields:
   ```
   Workclass Missing = IIF(ISNULL([Workclass]), 1, 0)
   Occupation Missing = IIF(ISNULL([Occupation]), 1, 0)
   Native Country Missing = IIF(ISNULL([Native Country]), 1, 0)
   ```
3. Build a bar chart: one bar per column (use a manually-pivoted long-form calculated field, or
   three separate reference lines) showing `SUM(Workclass Missing)`, `SUM(Occupation Missing)`,
   `SUM(Native Country Missing)`.
4. Add a text object on the sheet: *"Shows raw missingness only — the pipeline's Bayesian
   imputation currently feeds PCA/t-SNE inputs but has not been written back to these columns.
   True imputed-vs-raw comparison requires a pipeline change (see STEPS.md §0)."*

**Verify:** the three missing-count bars are non-zero and roughly match the counts noted during
data exploration (workclass ≈1,836, occupation ≈1,843, native-country ≈583); the caption text is
visible.

---

## 8. Overview Dashboard

1. New Dashboard, rename to `Overview Dashboard`. Set size to a fixed dashboard size (e.g.
   1400×900) rather than "Automatic".
2. Drag a **Horizontal** container onto the dashboard. Drop `Proj_PCA` and `Proj_tSNE` into it
   side by side.
3. Below that, drag another **Horizontal** container. Drop `Income_by_Education`,
   `Income_by_Occupation`, `Income_by_Age` into it.
4. Below that, drop `Bayesian_Imputation_QA`.
5. Drag the `Show As` parameter control onto the dashboard once (top or side) instead of showing
   it on every sheet individually — it will drive all three `Income_by_*` sheets simultaneously
   since they share the same `Income Measure` field.
6. Add a dashboard title text object: "Adult Census Income — Overview".

**Verify:** close and reopen the workbook; the dashboard renders with no broken data source
icons, and toggling `Show As` once updates all three Income_by_* charts at the same time.

---

## 9. Packaging

1. **Preferences.tps** — create a custom color palette so `Income` (`<=50K` / `>50K`) always
   renders with consistent colors across every sheet:
   - Locate your Tableau Repository folder (Documents\My Tableau Repository on Windows).
   - Create/edit `Preferences.tps` there with a `<color-palette>` entry, e.g.:
     ```xml
     <?xml version='1.0'?>
     <workbook>
       <preferences>
         <color-palette name="Adult Income" type="regular">
           <color>#4C78A8</color>
           <color>#E45756</color>
         </color-palette>
       </preferences>
     </workbook>
     ```
   - Restart Tableau Desktop so it picks up the new palette.
   - On each sheet, right-click the `Income` color legend → **Edit Colors** → select the
     "Adult Income" palette from the dropdown → assign `<=50K` and `>50K` explicitly.
   - Copy `Preferences.tps` to the repo root (required deliverable per CLAUDE.md §10).
2. **Package as .twbx** — `File → Export Packaged Workbook`, save as `adult_dashboard.twbx` at
   the repo root (not inside `dashboard/`). This embeds `adult_tableau.csv` so the workbook is
   self-contained.
3. Final check before zipping for submission: close `adult_dashboard.twbx`, reopen it fresh, and
   confirm every sheet and the dashboard render immediately with no "cannot connect to data
   source" prompts. `report.pdf`, `data/adult_imputed.csv`, and `params.json` are separate
   deliverables not covered by this document.
