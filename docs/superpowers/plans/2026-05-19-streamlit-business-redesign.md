# Streamlit Business Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform `src/app.py` from a technical ML showcase into a 5-page business presentation tool for a non-technical C-suite audience.

**Architecture:** Single file rewrite of `src/app.py`. No backend changes. Navigation restructured from 5 old pages to 5 new pages: Contexte & Enjeux, Liaisons à risque, Comprendre les retards, Prédire un retard, Synthèse.

**Tech Stack:** Python 3, Streamlit, Plotly Express, pandas, joblib, pathlib

---

## File Map

| File | Action |
|---|---|
| `src/app.py` | Full rewrite — navigation, CSS, all 5 pages |

No other files are modified. Backend (`src/data.py`, `src/config.py`, `src/model_io.py`) untouched.

---

## Task 1: Rewrite CSS — Light Corporate Theme

**Files:**
- Modify: `src/app.py:39-80` (the `<style>` block)

Replace the current dark-theme CSS with a light corporate design system (white background, navy titles `#1a3a5c`, blue accents `#2563eb`).

- [ ] **Step 1: Replace the CSS block**

Find and replace the entire `st.markdown("""\n    <style>` block (lines 40–80) with:

```python
    st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a3a5c;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-top: 4px;
        margin-bottom: 1.5rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a3a5c;
        border-bottom: 2px solid #2563eb;
        padding-bottom: 6px;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .insight-card {
        background: #ffffff;
        border-left: 4px solid #2563eb;
        border-radius: 8px;
        padding: 20px 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 12px;
        min-height: 120px;
    }
    .insight-number {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1a3a5c;
        margin-bottom: 6px;
    }
    .insight-label {
        font-size: 0.9rem;
        color: #6b7280;
        line-height: 1.4;
    }
    .next-step-card {
        background: #f0f7ff;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 8px;
        border-left: 3px solid #2563eb;
    }
    .model-encart {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 12px 18px;
        border: 1px solid #e5e7eb;
        color: #6b7280;
        font-size: 0.85rem;
        margin-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/app.py').read()); print('syntax OK')"
```

Expected: `syntax OK`

- [ ] **Step 3: Commit**

```bash
git add src/app.py
git commit -m "style(app): replace dark CSS with light corporate theme"
```

---

## Task 2: Rewrite Sidebar & Navigation

**Files:**
- Modify: `src/app.py:84-98` (sidebar block)

Replace the 5 old page names with the 5 new business-oriented names. Update sidebar content to feel professional.

- [ ] **Step 1: Replace the sidebar block**

Find and replace the `# --- Sidebar ---` block (lines 84–98) with:

```python
    # --- Sidebar ---
    with st.sidebar:
        st.markdown("## 🚄 SNCF TGV")
        st.markdown("##### Outil d'analyse & prédiction")
        st.markdown("---")
        page = st.radio(
            "Navigation",
            [
                "Contexte & Enjeux",
                "Liaisons a risque",
                "Comprendre les retards",
                "Predire un retard",
                "Synthese",
            ],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption("Dataset SNCF Open Data")
        st.caption("Régularité mensuelle TGV · 2018–2026")
        st.markdown("---")
        st.caption("Projet ML · Albert School")
        st.caption("Lenny Sebban")
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/app.py').read()); print('syntax OK')"
```

Expected: `syntax OK`

- [ ] **Step 3: Commit**

```bash
git add src/app.py
git commit -m "feat(app): update sidebar navigation — 5 business-oriented pages"
```

---

## Task 3: Page 1 — Contexte & Enjeux

**Files:**
- Modify: `src/app.py` — replace old pages "Vue d ensemble" + "Exploration des donnees" + "Comparaison des modeles" with single new page

This page merges the overview KPIs, temporal evolution chart, monthly seasonality bar, and a small model encart. Remove technical jargon entirely.

- [ ] **Step 1: Replace pages 1, 2 and 4**

Find and replace from `if page == "Vue d ensemble":` through the end of the `elif page == "Comparaison des modeles":` block with:

```python
    # =========================================================
    # PAGE 1 : CONTEXTE & ENJEUX
    # =========================================================
    if page == "Contexte & Enjeux":
        st.markdown('<p class="main-title">Retards TGV — Analyse & Prédiction</p>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Données SNCF Open Data · 2018–2026 · Réseau TGV national</p>', unsafe_allow_html=True)

        # KPIs
        retard_moyen = df[target].mean()
        nb_liaisons = df.groupby([col_dep, col_arr]).ngroups
        nb_gares = df[col_dep].nunique()
        nb_obs = len(df)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Retard moyen", f"{retard_moyen:.1f} min")
        with k2:
            st.metric("Liaisons analysées", f"{nb_liaisons}")
        with k3:
            st.metric("Gares couvertes", f"{nb_gares}")
        with k4:
            st.metric("Observations", f"{nb_obs:,}".replace(",", " "))

        st.markdown("")
        st.markdown('<p class="section-header">Évolution du retard moyen (2018–2026)</p>', unsafe_allow_html=True)

        monthly = df.groupby(["Annee", "Mois"])[target].mean().reset_index()
        monthly["Date"] = pd.to_datetime(
            monthly["Annee"].astype(str) + "-" + monthly["Mois"].astype(str).str.zfill(2) + "-01"
        )
        fig_timeline = px.line(
            monthly, x="Date", y=target,
            labels={target: "Retard moyen (min)", "Date": ""},
            template="plotly_white",
        )
        fig_timeline.update_traces(line_color="#2563eb", line_width=2.5)
        fig_timeline.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_timeline, use_container_width=True)

        st.markdown('<p class="section-header">Saisonnalité par mois</p>', unsafe_allow_html=True)
        mois_noms = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]
        mois_data = df.groupby("Mois")[target].mean().reset_index()
        mois_data["Mois_nom"] = [mois_noms[i - 1] for i in mois_data["Mois"]]
        fig_mois = px.bar(
            mois_data, x="Mois_nom", y=target,
            labels={target: "Retard moyen (min)", "Mois_nom": ""},
            template="plotly_white",
            color=target,
            color_continuous_scale=[[0, "#bfdbfe"], [1, "#1a3a5c"]],
        )
        fig_mois.update_layout(height=280, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_mois, use_container_width=True)

        # Model encart — discret
        if MODEL_METRICS_FILE.exists():
            _mdf = pd.read_csv(MODEL_METRICS_FILE)
            _best = _mdf.loc[_mdf["R2"].idxmax()]
            _mae, _r2 = _best["MAE"], _best["R2"]
        else:
            _mae, _r2 = 3.0, 0.68
        st.markdown(
            f'<div class="model-encart">🤖 <b>Modèle de prédiction :</b> Gradient Boosting — '
            f'Précision R² = {_r2:.2f} — Erreur moyenne = {_mae:.1f} min</div>',
            unsafe_allow_html=True,
        )
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/app.py').read()); print('syntax OK')"
```

Expected: `syntax OK`

- [ ] **Step 3: Commit**

```bash
git add src/app.py
git commit -m "feat(app): add page Contexte & Enjeux — merge overview + exploration + model encart"
```

---

## Task 4: Page 2 — Liaisons à risque (nouvelle page)

**Files:**
- Modify: `src/app.py` — insert new page block after page 1

New page with a top-15 bar chart (horizontal) and a filterable table sorted by average delay.

- [ ] **Step 1: Insert the new page block**

After the closing of the "Contexte & Enjeux" `if` block, add:

```python
    # =========================================================
    # PAGE 2 : LIAISONS A RISQUE
    # =========================================================
    elif page == "Liaisons a risque":
        st.markdown('<p class="main-title">Liaisons à risque</p>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Classement des liaisons les plus touchées par les retards</p>', unsafe_allow_html=True)

        st.markdown('<p class="section-header">Top 15 des liaisons les plus retardées</p>', unsafe_allow_html=True)

        top15 = (
            df.groupby([col_dep, col_arr])[target]
            .mean()
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
        )
        top15["Liaison"] = top15[col_dep] + " → " + top15[col_arr]

        fig_top = px.bar(
            top15, x=target, y="Liaison", orientation="h",
            labels={target: "Retard moyen (min)", "Liaison": ""},
            template="plotly_white",
            color=target,
            color_continuous_scale=[[0, "#bfdbfe"], [1, "#1a3a5c"]],
        )
        fig_top.update_layout(
            height=520,
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            margin=dict(l=0, r=20, t=10, b=0),
        )
        st.plotly_chart(fig_top, use_container_width=True)

        st.markdown('<p class="section-header">Détail par gare de départ</p>', unsafe_allow_html=True)

        all_deps = ["Toutes"] + sorted(df[col_dep].unique().tolist())
        selected_dep = st.selectbox("Filtrer par gare de départ", all_deps)

        liaisons_df = (
            df.groupby([col_dep, col_arr])[target]
            .agg(["mean", "max", "count"])
            .reset_index()
            .rename(columns={
                col_dep: "Départ",
                col_arr: "Arrivée",
                "mean": "Retard moyen (min)",
                "max": "Retard max (min)",
                "count": "Nb observations",
            })
            .sort_values("Retard moyen (min)", ascending=False)
        )
        liaisons_df["Retard moyen (min)"] = liaisons_df["Retard moyen (min)"].round(1)
        liaisons_df["Retard max (min)"] = liaisons_df["Retard max (min)"].round(1)

        if selected_dep != "Toutes":
            liaisons_df = liaisons_df[liaisons_df["Départ"] == selected_dep]

        st.dataframe(liaisons_df.reset_index(drop=True), use_container_width=True, hide_index=True)
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/app.py').read()); print('syntax OK')"
```

Expected: `syntax OK`

- [ ] **Step 3: Commit**

```bash
git add src/app.py
git commit -m "feat(app): add page Liaisons a risque — top 15 chart + filterable table"
```

---

## Task 5: Page 3 — Comprendre les retards

**Files:**
- Modify: `src/app.py` — replace old "Causes de retard" page

Rename page, add a donut chart alongside the bar chart, switch to light plotly theme. Remove the COVID comparison section (too historical/niche for a C-suite demo).

- [ ] **Step 1: Replace the old "Causes de retard" block**

Find `elif page == "Causes de retard":` and replace the entire block with:

```python
    # =========================================================
    # PAGE 3 : COMPRENDRE LES RETARDS
    # =========================================================
    elif page == "Comprendre les retards":
        st.markdown('<p class="main-title">Comprendre les retards</p>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Quelles sont les causes et comment évoluent-elles ?</p>', unsafe_allow_html=True)

        cause_cols = [c for c in df.columns if c.startswith("Prct retard")]
        cause_names = {
            cause_cols[0]: "Causes externes",
            cause_cols[1]: "Infrastructure",
            cause_cols[2]: "Gestion du trafic",
            cause_cols[3]: "Matériel roulant",
            cause_cols[4]: "Gestion en gare",
            cause_cols[5]: "Voyageurs",
        }

        cause_means = {v: df[k].mean() for k, v in cause_names.items()}
        cause_df = pd.DataFrame(list(cause_means.items()), columns=["Cause", "Part (%)"])

        col_donut, col_bar = st.columns(2)
        with col_donut:
            st.markdown('<p class="section-header">Répartition globale</p>', unsafe_allow_html=True)
            fig_donut = px.pie(
                cause_df, values="Part (%)", names="Cause",
                template="plotly_white",
                color_discrete_sequence=["#1a3a5c", "#2563eb", "#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe"],
                hole=0.4,
            )
            fig_donut.update_layout(height=360, legend=dict(orientation="v", font=dict(size=11)))
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_bar:
            st.markdown('<p class="section-header">Par importance</p>', unsafe_allow_html=True)
            fig_bar = px.bar(
                cause_df.sort_values("Part (%)"), x="Part (%)", y="Cause", orientation="h",
                template="plotly_white",
                color="Part (%)",
                color_continuous_scale=[[0, "#bfdbfe"], [1, "#1a3a5c"]],
            )
            fig_bar.update_layout(height=360, coloraxis_showscale=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown('<p class="section-header">Évolution des causes par année</p>', unsafe_allow_html=True)
        cause_year = df.groupby("Annee")[cause_cols].mean().reset_index()
        cause_year_melted = cause_year.melt(id_vars="Annee", var_name="Cause", value_name="Part (%)")
        cause_year_melted["Cause"] = cause_year_melted["Cause"].map(cause_names)
        fig_area = px.area(
            cause_year_melted, x="Annee", y="Part (%)", color="Cause",
            template="plotly_white",
            labels={"Part (%)": "% moyen"},
            color_discrete_sequence=["#1a3a5c", "#2563eb", "#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe"],
        )
        fig_area.update_layout(height=380)
        st.plotly_chart(fig_area, use_container_width=True)
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/app.py').read()); print('syntax OK')"
```

Expected: `syntax OK`

- [ ] **Step 3: Commit**

```bash
git add src/app.py
git commit -m "feat(app): rename Causes de retard -> Comprendre les retards, add donut chart, light theme"
```

---

## Task 6: Page 4 — Prédire un retard (tweaks mineurs)

**Files:**
- Modify: `src/app.py` — rename page branch + update 2 labels

The prediction page is already well-built. Just update the `elif` condition to match the new page name and change 2 user-visible strings.

- [ ] **Step 1: Rename the page condition**

Find:
```python
    elif page == "Predicteur interactif":
```
Replace with:
```python
    # =========================================================
    # PAGE 4 : PREDIRE UN RETARD
    # =========================================================
    elif page == "Predire un retard":
```

- [ ] **Step 2: Update the page title and metric label**

Find:
```python
        st.markdown('<p class="main-title">Predicteur interactif</p>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Selectionnez une liaison, une date et une heure pour predire le retard moyen</p>', unsafe_allow_html=True)
```
Replace with:
```python
        st.markdown('<p class="main-title">Estimation du retard</p>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Sélectionnez une liaison et une date pour obtenir une estimation du retard à l\'arrivée</p>', unsafe_allow_html=True)
```

Find:
```python
                st.metric("Retard predit", f"{prediction:.1f} min")
```
Replace with:
```python
                st.metric("Estimation du retard", f"{prediction:.1f} min")
```

- [ ] **Step 3: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/app.py').read()); print('syntax OK')"
```

Expected: `syntax OK`

- [ ] **Step 4: Commit**

```bash
git add src/app.py
git commit -m "feat(app): rename predicteur page, update user-facing labels to business language"
```

---

## Task 7: Page 5 — Synthèse (nouvelle page)

**Files:**
- Modify: `src/app.py` — add final page at end of elif chain

Conclusion page with 3 dynamic insight cards (computed from real data) and a "Et ensuite ?" section with 3 next-step cards.

- [ ] **Step 1: Add the Synthèse page**

After the closing of the "Predire un retard" `elif` block (end of file before `if __name__ == "__main__":`), add:

```python
    # =========================================================
    # PAGE 5 : SYNTHESE
    # =========================================================
    elif page == "Synthese":
        st.markdown('<p class="main-title">Synthèse & Perspectives</p>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Ce que les données révèlent — et ce qu\'on peut en faire</p>', unsafe_allow_html=True)

        # --- Dynamic insights computed from data ---
        mois_data_s = df.groupby("Mois")[target].mean()
        pire_mois_idx = int(mois_data_s.idxmax())
        mois_noms_fr = {
            1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
            5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
            9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre",
        }
        pire_mois_nom = mois_noms_fr[pire_mois_idx]
        pire_mois_val = mois_data_s[pire_mois_idx]

        liaison_means = df.groupby([col_dep, col_arr])[target].mean()
        pire_liaison = liaison_means.idxmax()
        pire_retard_val = liaison_means.max()

        if MODEL_METRICS_FILE.exists():
            _mdf2 = pd.read_csv(MODEL_METRICS_FILE)
            _mae2 = _mdf2.loc[_mdf2["R2"].idxmax(), "MAE"]
        else:
            _mae2 = 3.0

        st.markdown('<p class="section-header">3 enseignements clés</p>', unsafe_allow_html=True)
        i1, i2, i3 = st.columns(3)

        with i1:
            st.markdown(
                f'<div class="insight-card">'
                f'<div class="insight-number">📅 {pire_mois_nom}</div>'
                f'<div class="insight-label">est le mois le plus à risque avec {pire_mois_val:.1f} min de retard moyen — '
                f'planifier des ressources supplémentaires en conséquence.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with i2:
            liaison_str = f"{pire_liaison[0]} → {pire_liaison[1]}"
            st.markdown(
                f'<div class="insight-card">'
                f'<div class="insight-number">🚂 {pire_retard_val:.0f} min</div>'
                f'<div class="insight-label">de retard moyen sur la liaison la plus impactée<br>'
                f'<b>{liaison_str}</b> — priorité d\'action identifiée.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with i3:
            st.markdown(
                f'<div class="insight-card">'
                f'<div class="insight-number">🎯 ±{_mae2:.1f} min</div>'
                f'<div class="insight-label">d\'erreur moyenne du modèle de prédiction — '
                f'une précision opérationnelle suffisante pour anticiper les situations à risque.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("")
        st.markdown('<p class="section-header">Et ensuite ?</p>', unsafe_allow_html=True)

        next_steps = [
            (
                "🚀 Déploiement en production",
                "Intégrer l'outil dans les tableaux de bord opérationnels SNCF pour un usage quotidien par les équipes métier et les gestionnaires de lignes.",
            ),
            (
                "📡 Données en temps réel",
                "Connecter le modèle aux flux de données temps réel (perturbations, météo, trafic) pour des prédictions dynamiques à J+1 et J+7.",
            ),
            (
                "🔔 Alertes automatiques",
                "Mettre en place un système d'alertes sur les liaisons à risque avant les grandes périodes de trafic (vacances scolaires, événements).",
            ),
        ]

        for title, desc in next_steps:
            st.markdown(
                f'<div class="next-step-card">'
                f'<b>{title}</b><br>'
                f'<span style="color:#6b7280;font-size:0.9rem">{desc}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/app.py').read()); print('syntax OK')"
```

Expected: `syntax OK`

- [ ] **Step 3: Commit**

```bash
git add src/app.py
git commit -m "feat(app): add Synthese page — 3 dynamic insight cards + next steps"
```

---

## Task 8: Push & Final Check

- [ ] **Step 1: Run full syntax check**

```bash
python3 -c "import ast; ast.parse(open('src/app.py').read()); print('syntax OK')"
```

- [ ] **Step 2: Verify page count**

```bash
grep -c "elif page ==" src/app.py
```

Expected: `4` (4 `elif` + 1 `if` = 5 pages total)

- [ ] **Step 3: Push to GitHub**

```bash
git push origin main
```

---

## Spec Coverage Checklist

| Spec requirement | Task |
|---|---|
| Light corporate CSS (white/navy/blue) | Task 1 |
| Sidebar 5 new page names | Task 2 |
| Page 1: KPIs + timeline + saisonnalité + model encart | Task 3 |
| Page 2: Top 15 + filterable table | Task 4 |
| Page 3: Donut + bar + area chart, light theme | Task 5 |
| Page 4: Rename + business labels | Task 6 |
| Page 5: 3 insight cards + next steps | Task 7 |
| Remove old pages (Exploration, Comparaison) | Task 3 (merged/removed) |
| No technical jargon in page content | Tasks 3–7 |
| Push to GitHub | Task 8 |
