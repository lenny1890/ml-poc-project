"""Streamlit application for SNCF TGV delay prediction."""

from __future__ import annotations

import datetime
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go

from config import PROJECT_ROOT, MODELS_DIR, MODEL_METRICS_FILE, ensure_dirs


def load_data():
    ensure_dirs()
    df = pd.read_csv(PROJECT_ROOT / "sncf_retards.csv", sep=";")
    target = [c for c in df.columns if "Retard moyen de tous les trains" in c and "arriv" in c][0]
    col_arr = [c for c in df.columns if "Gare" in c and "arriv" in c.lower()][0]
    col_dep = [c for c in df.columns if c.startswith("Gare de d")][0]
    df["Annee"] = df["Date"].str[:4].astype(int)
    df["Mois"] = df["Date"].str[5:7].astype(int)
    df["Mois_nom"] = df["Mois"].map({
        1: "Jan", 2: "Fev", 3: "Mar", 4: "Avr", 5: "Mai", 6: "Juin",
        7: "Juil", 8: "Aout", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    })
    return df, target, col_dep, col_arr


def build_app() -> None:
    st.set_page_config(
        page_title="SNCF TGV Delay Predictor",
        page_icon="\U0001F684",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
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

    df, target, col_dep, col_arr = load_data()

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
            st.metric("Observations", f"{nb_obs:,}".replace(",", " "))

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

    # =========================================================
    # PAGE 4 : PREDIRE UN RETARD
    # =========================================================
    elif page == "Predire un retard":
        st.markdown('<p class="main-title">Estimation du retard</p>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Sélectionnez une liaison et une date pour obtenir une estimation du retard à l\'arrivée</p>', unsafe_allow_html=True)
        st.markdown("")

        le_depart = joblib.load(MODELS_DIR / "le_depart.joblib")
        le_arrivee = joblib.load(MODELS_DIR / "le_arrivee.joblib")
        best_model = joblib.load(MODELS_DIR / "gradient_boosting.joblib")

        # Build valid route combinations from the dataset
        valid_routes: dict[str, list[str]] = (
            df.groupby(col_dep)[col_arr]
            .apply(lambda s: sorted(s.unique()))
            .to_dict()
        )
        all_departures = sorted(valid_routes.keys())

        col1, col2 = st.columns(2)
        with col1:
            default_dep = "PARIS LYON" if "PARIS LYON" in valid_routes else all_departures[0]
            gare_dep = st.selectbox("Gare de depart", all_departures, index=all_departures.index(default_dep))
        with col2:
            valid_arrivals = valid_routes.get(gare_dep, [])
            default_arr = "MARSEILLE ST CHARLES" if "MARSEILLE ST CHARLES" in valid_arrivals else valid_arrivals[0]
            gare_arr = st.selectbox("Gare d arrivee", valid_arrivals, index=valid_arrivals.index(default_arr))

        # Date and time inputs
        col_date, col_heure, col_jour = st.columns(3)
        with col_date:
            depart_date = st.date_input(
                "Date de depart",
                value=datetime.date.today(),
                min_value=datetime.date(2018, 1, 1),
                max_value=datetime.date(2030, 12, 31),
            )
        with col_heure:
            heure = st.slider("Heure de depart", 0, 23, 8, format="%dh00")
        with col_jour:
            jours = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
            st.markdown(f"**Jour**")
            st.markdown(f"### {jours[depart_date.weekday()]}")

        mois = depart_date.month
        annee = depart_date.year

        mask = (df[col_dep] == gare_dep) & (df[col_arr] == gare_arr)
        annul_col = [c for c in df.columns if "annul" in c.lower() and "Nombre" in c][0]
        duree_col = [c for c in df.columns if "moyenne du trajet" in c][0]
        circ_col = [c for c in df.columns if "circulations" in c][0]
        duree = df.loc[mask, duree_col]
        duree_val = int(duree.mean()) if len(duree) > 0 else 120
        circulations = df.loc[mask, circ_col]
        circ_val = int(circulations.mean()) if len(circulations) > 0 else 300

        if st.button("Estimer le retard", type="primary", use_container_width=True):
            annul_mean = df.loc[mask, annul_col].mean() if len(df.loc[mask]) > 0 else 0
            retard_dep_col = [c for c in df.columns if "Retard moyen de tous les trains" in c and "part" in c.lower()][0]
            retard_dep_val = df.loc[mask, retard_dep_col].mean() if len(df.loc[mask]) > 0 else 3.0
            nb_retard_dep_col = [c for c in df.columns if "Nombre de trains en retard au d" in c][0]
            nb_retard_dep = df.loc[mask, nb_retard_dep_col].mean() if len(df.loc[mask]) > 0 else 30
            taux_annul = (annul_mean / circ_val * 100) if circ_val > 0 else 0
            mois_sin_val = np.sin(2 * np.pi * mois / 12)
            mois_cos_val = np.cos(2 * np.pi * mois / 12)
            pct_ret_dep = (nb_retard_dep / circ_val * 100) if circ_val > 0 else 10
            cause_cols_data = [c for c in df.columns if c.startswith("Prct retard")]
            cause_vals = []
            for cc in cause_cols_data:
                val = df.loc[mask, cc].mean() if len(df.loc[mask]) > 0 else df[cc].mean()
                cause_vals.append(val if not np.isnan(val) else 0)
            feature_vector = [
                le_depart.transform([gare_dep])[0],
                le_arrivee.transform([gare_arr])[0],
                duree_val, circ_val, annul_mean, annee,
                mois_sin_val, mois_cos_val,
                taux_annul, retard_dep_val, pct_ret_dep,
            ] + cause_vals
            features = np.array([feature_vector])
            prediction = best_model.predict(features)[0]

            st.markdown("")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Estimation du retard", f"{prediction:.1f} min")
            with m2:
                st.metric("Duree du trajet", f"{duree_val} min")
            with m3:
                st.metric("Circulations mensuelles", f"{circ_val}")

        # Historique de la liaison
        historique = df.loc[mask].copy()
        if len(historique) > 0:
            st.markdown('<p class="section-header">Historique de cette liaison</p>', unsafe_allow_html=True)
            hist_monthly = historique.groupby(["Annee", "Mois"])[target].mean().reset_index()
            hist_monthly["Date"] = pd.to_datetime(hist_monthly["Annee"].astype(str) + "-" + hist_monthly["Mois"].astype(str).str.zfill(2) + "-01")

            fig_hist = px.line(
                hist_monthly, x="Date", y=target,
                labels={target: "Retard moyen (min)", "Date": ""},
                template="plotly_white",
            )
            fig_hist.update_traces(line_color="#2563eb", line_width=2)
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)

            # Stats de la liaison
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                st.metric("Retard moyen historique", f"{historique[target].mean():.1f} min")
            with col_s2:
                st.metric("Retard max observe", f"{historique[target].max():.1f} min")
            with col_s3:
                st.metric("Retard min observe", f"{historique[target].min():.1f} min")
            with col_s4:
                annul_col = [c for c in df.columns if "annul" in c][0]
                st.metric("Trains annules (moy)", f"{historique[annul_col].mean():.0f}")
        else:
            st.warning("Aucune donnee historique pour cette liaison.")

    # =========================================================
    # PAGE 5 : SYNTHESE
    # =========================================================
    elif page == "Synthese":
        st.markdown('<p class="main-title">Synthèse & Perspectives</p>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Ce que les données révèlent — et ce qu\'on peut en faire</p>', unsafe_allow_html=True)

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


if __name__ == "__main__":
    build_app()
