"""Stage 5a: Plotly HTML prototype of the Tableau dashboard.

Design-validation artefact only (not submitted). Mirrors the planned workbook:
Proj_PCA / Proj_tSNE scatters (colour = income, blue/orange CVD-validated
pair, 5-field tooltips), Income_by_Education / Occupation / Age bars with an
Absolute<->Proportion toggle (mirrors the Tableau 'Show As' parameter), and a
Bayesian imputation QA panel (observed marginal vs Bayesian vs mode baseline).

Light theme only: the prototype mirrors Tableau Desktop's light workbook.

Verify gate: opens in browser; tooltips show age, education, occupation,
income, variables_used; toggle switches counts <-> percent.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT = ROOT / "prototype" / "dashboard.html"

# dataviz-validated palette (light surface #fcfcfb)
C_LOW, C_HIGH = "#2a78d6", "#eb6834"          # income <=50K / >50K
C_OBS, C_BAYES, C_MODE = "#898781", "#2a78d6", "#eda100"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
INCOME_COLORS = {"<=50K": C_LOW, ">50K": C_HIGH}

LAYOUT = dict(
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
    font=dict(family='system-ui, "Segoe UI", sans-serif', color=INK, size=13),
    margin=dict(l=60, r=20, t=60, b=50),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor="#c3c2b7"),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor="#c3c2b7"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)

TOOLTIP = ("age: %{customdata[0]}<br>education: %{customdata[1]}<br>"
           "occupation: %{customdata[2]}<br>income: %{customdata[3]}<br>"
           "hours/week: %{customdata[4]}<br>sex: %{customdata[5]}"
           "<br><i>variables used: %{customdata[6]}</i><extra></extra>")


def projection_fig(df, x, y, title, subtitle):
    fig = go.Figure()
    for income in ["<=50K", ">50K"]:
        sub = df[df["income"] == income]
        fig.add_trace(go.Scattergl(
            x=sub[x], y=sub[y], mode="markers", name=income,
            marker=dict(color=INCOME_COLORS[income], size=3, opacity=0.35),
            customdata=sub[["age", "education", "occupation", "income",
                            "hours_per_week", "sex", "variables_used"]].values,
            hovertemplate=TOOLTIP,
        ))
    fig.update_layout(
        **LAYOUT, height=560,
        title=dict(text=f"{title}<br><sup>{subtitle}</sup>"),
    )
    fig.update_xaxes(title_text=x)
    fig.update_yaxes(title_text=y)
    return fig


def bars_fig(df, key, title, order=None, height=440):
    counts = (df.groupby([key, "income"], observed=True).size()
              .unstack(fill_value=0).reindex(columns=["<=50K", ">50K"]))
    if order is not None:
        counts = counts.reindex(order)
    fig = go.Figure()
    for income in ["<=50K", ">50K"]:
        fig.add_trace(go.Bar(
            x=counts.index.astype(str), y=counts[income], name=income,
            marker=dict(color=INCOME_COLORS[income],
                        line=dict(color=SURFACE, width=1)),
            hovertemplate=f"%{{x}}<br>income {income}: %{{y:,.0f}}"
                          "<extra></extra>",
        ))
    fig.update_layout(
        **LAYOUT, height=height, barmode="stack",
        title=dict(text=title),
        updatemenus=[dict(
            type="buttons", direction="right", x=1, xanchor="right",
            y=1.18, yanchor="top", showactive=True,
            buttons=[
                dict(label="Absolute", method="relayout",
                     args=[{"barnorm": "", "yaxis.title.text": "people"}]),
                dict(label="Proportion", method="relayout",
                     args=[{"barnorm": "percent",
                            "yaxis.title.text": "% within group"}]),
            ],
        )],
    )
    fig.update_yaxes(title_text="people")
    return fig


def qa_fig(imp, target):
    observed = imp.loc[~imp[f"{target}_was_imputed"], target]
    bayes = imp.loc[imp[f"{target}_was_imputed"], target]
    mode = imp.loc[imp[f"{target}_was_imputed"], f"{target}_mode_baseline"]
    cats = observed.value_counts().index.tolist()
    series = [("Observed marginal", observed, C_OBS),
              ("Bayesian imputed", bayes, C_BAYES),
              ("Mode baseline", mode, C_MODE)]
    fig = go.Figure()
    for name, s, color in series:
        pct = (s.value_counts(normalize=True).reindex(cats).fillna(0) * 100)
        fig.add_trace(go.Bar(
            x=cats, y=pct.values, name=name,
            marker=dict(color=color, line=dict(color=SURFACE, width=1)),
            hovertemplate="%{x}<br>" + name + ": %{y:.1f}%<extra></extra>",
        ))
    fig.update_layout(
        **LAYOUT, height=420, barmode="group",
        title=dict(text=f"Bayesian_Imputation_QA — {target}"
                        f"<br><sup>share of rows per category (%): "
                        f"{len(bayes):,} imputed rows vs "
                        f"{len(observed):,} observed</sup>"),
    )
    fig.update_yaxes(title_text="% of rows")
    return fig


def main():
    pca = pd.read_csv(DATA_DIR / "adult_pca_2d.csv", index_col="index")
    tsne = pd.read_csv(DATA_DIR / "adult_tsne_2d.csv", index_col="index")
    imp = pd.read_csv(DATA_DIR / "adult_imputed.csv", index_col="index")
    params = json.loads((ROOT / "params.json").read_text())

    evr = params["pca"]["explained_variance_ratio"]
    figs = [
        ("Projections", projection_fig(
            pca, "PC1", "PC2", "Proj_PCA",
            f"explained variance: PC1 {evr[0]:.1%}, PC2 {evr[1]:.1%} — "
            "only relative distances are meaningful")),
        (None, projection_fig(
            tsne, "TSNE1", "TSNE2", "Proj_tSNE",
            f"perplexity {params['tsne']['perplexity']}, "
            f"max_iter {params['tsne']['max_iter']}, "
            f"seed {params['tsne']['random_state']} — "
            "only relative distances are meaningful")),
    ]

    edu_order = (imp[["education", "education_num"]].drop_duplicates()
                 .sort_values("education_num")["education"].tolist())
    occ_order = imp["occupation"].value_counts().index.tolist()
    imp["age_band"] = (imp["age"] // 5 * 5).astype(str) + "s"
    age_order = sorted(imp["age_band"].unique(), key=lambda s: int(s[:-1]))

    figs += [
        ("Income by group", bars_fig(imp, "education",
                                     "Income_by_Education (ordered by years of education)",
                                     order=edu_order)),
        (None, bars_fig(imp, "occupation",
                        "Income_by_Occupation (value-sorted)", order=occ_order)),
        (None, bars_fig(imp, "age_band", "Income_by_Age (5-year bands)",
                        order=age_order)),
        ("Imputation QA", qa_fig(imp, "workclass")),
        (None, qa_fig(imp, "occupation")),
    ]

    blocks, first = [], True
    for heading, fig in figs:
        if heading:
            blocks.append(f"<h2>{heading}</h2>")
        blocks.append(pio.to_html(
            fig, full_html=False, include_plotlyjs=("inline" if first else False),
            config={"displaylogo": False}))
        first = False

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Adult dashboard prototype</title>
<style>
 body {{ background:#f9f9f7; color:{INK}; margin:0 auto; max-width:1200px;
        padding:24px; font-family:system-ui,"Segoe UI",sans-serif; }}
 h1 {{ font-size:22px; margin-bottom:4px; }}
 h2 {{ font-size:16px; color:{INK2}; border-bottom:1px solid {GRID};
      padding-bottom:4px; margin-top:32px; }}
 p.note {{ color:{INK2}; font-size:13px; }}
 .card {{ background:{SURFACE}; border:1px solid {GRID}; margin:12px 0;
         padding:8px; }}
</style></head><body>
<h1>UCI Adult (1994 census) — dashboard prototype</h1>
<p class="note">Design-validation prototype for the Tableau workbook.
{len(imp):,} records; income encoded blue (&le;50K) / orange (&gt;50K),
CVD-validated. Bar charts: Absolute&harr;Proportion toggle top-right
(mirrors the Tableau <i>Show As</i> parameter). Projection tooltips carry
age, education, occupation, income and the variables used.</p>
{"".join(f'<div class="card">{b}</div>' if b.startswith('<div') else b for b in blocks)}
</body></html>"""
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"[ok] wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
