"""Streamlit application for SNCF TGV delay prediction."""

from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go

from config import DATA_DIR, MODELS_DIR, MODEL_METRICS_FILE


def load_data():
    df = pd.read_csv(DATA_DIR / "sncf_retards.csv", sep=";")
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
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1e3a5f, #4a90d9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #888;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #4a90d9;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4a90d9;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        border-bottom: 2px solid #4a90d9;
        padding-bottom: 8px;
        margin-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

    df, target, col_dep, col_arr = load_data()

    # --- Sidebar ---
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/SNCF_Logo.svg/240px-SNCF_Logo.svg.png", width=120)
        st.markdown("### Navigation")
        page = st.radio(
            "Section",
            ["Vue d ensemble", "Exploration des donnees", "Causes de retard", "Comparaison des modeles", "Predicteur interactif"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown("**Dataset SNCF Open Data**")
        st.markdown("Regularite mensuelle TGV par liaisons (2018-2026)")
        st.markdown("---")
        st.markdown("Projet ML - Albert School")
        st.markdown("Lenny Sebban")

    # =========================================================
    # PAGE 1 : VUE D'ENSEMBLE
    # =========================================================
    if page == "Vue d ensemble":
        st.markdown('<p class="main-title">Prediction des retards TGV</p>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Analyse et prediction des retards moyens a l arrivee des TGV en France</p>', unsafe_allow_html=True)
        st.markdown("")

        # KPI Cards
        retard_moyen = df[target].mean()
        nb_liaisons = df.groupby([col_dep, col_arr]).ngroups
        nb_gares = df[col_dep].nunique()
        nb_obs = len(df)
        pire_liaison = df.groupby([col_dep, col_arr])[target].mean().idxmax()
        pire_retard = df.groupby([col_dep, col_arr])[target].mean().max()

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Retard moyen global", f"{retard_moyen:.1f} min")
        with k2:
            st.metric("Liaisons analysees", f"{nb_liaisons}")
        with k3:
            st.metric("Gares couvertes", f"{nb_gares}")
        with k4:
            st.metric("Observations", f"{nb_obs:,}".replace(",", " "))

        st.markdown("")

        # Evolution temporelle
        st.markdown('<p class="section-header">Evolution du retard moyen dans le temps</p>', unsafe_allow_html=True)
        monthly = df.groupby(["Annee", "Mois"])[target].mean().reset_index()
        monthly["Date"] = pd.to_datetime(monthly["Annee"].astype(str) + "-" + monthly["Mois"].astype(str).str.zfill(2) + "-01")
        fig_timeline = px.line(
            monthly, x="Date", y=target,
            labels={target: "Retard moyen (min)", "Date": ""},
            template="plotly_dark",
        )
        fig_timeline.update_traces(line_color="#4a90d9", line_width=2)
        fig_timeline.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_timeline, use_container_width=True)

        # Contexte
        st.markdown('<p class="section-header">A propos du projet</p>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                "La SNCF publie chaque mois les donnees de regularite de ses TGV "
                "par liaison gare a gare. Ce projet exploite ces donnees Open Data "
                "pour entrainer des modeles de Machine Learning capables de predire "
                "le retard moyen a l arrivee."
            )
        with col_b:
            st.markdown(
                "**Approche** : 4 algorithmes de regression sont compares "
                "(Linear Regression, Random Forest, Gradient Boosting, KNN). "
                "Le meilleur modele (Gradient Boosting, R2=0.50) est utilise "
                "dans le predicteur interactif."
            )

    # =========================================================
    # PAGE 2 : EXPLORATION DES DONNEES
    # =========================================================
    elif page == "Exploration des donnees":
        st.markdown('<p class="main-title">Exploration des donnees</p>', unsafe_allow_html=True)
        st.markdown("")

        tab1, tab2, tab3 = st.tabs(["Saisonnalite", "Par annee", "Top liaisons"])

        with tab1:
            mois_data = df.groupby("Mois")[target].agg(["mean", "std", "count"]).reset_index()
            mois_noms = ["Jan", "Fev", "Mar", "Avr", "Mai", "Juin", "Juil", "Aout", "Sep", "Oct", "Nov", "Dec"]
            mois_data["Mois_nom"] = [mois_noms[i-1] for i in mois_data["Mois"]]

            fig_mois = px.bar(
                mois_data, x="Mois_nom", y="mean",
                error_y="std",
                labels={"mean": "Retard moyen (min)", "Mois_nom": "Mois"},
                template="plotly_dark",
                color="mean",
                color_continuous_scale="RdYlGn_r",
            )
            fig_mois.update_layout(height=500, showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig_mois, use_container_width=True)

            st.markdown(
                "Les mois d ete (juillet) et d hiver (janvier, fevrier) "
                "affichent les retards les plus eleves, en raison de l affluence "
                "estivale et des conditions meteorologiques hivernales."
            )

        with tab2:
            annee_data = df.groupby("Annee")[target].mean().reset_index()
            fig_annee = px.bar(
                annee_data, x="Annee", y=target,
                labels={target: "Retard moyen (min)", "Annee": "Annee"},
                template="plotly_dark",
                color=target,
                color_continuous_scale="Blues",
            )
            fig_annee.update_layout(height=500, showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig_annee, use_container_width=True)

        with tab3:
            n_top = st.slider("Nombre de liaisons a afficher", 5, 20, 10)
            top = (
                df.groupby([col_dep, col_arr])[target]
                .mean()
                .sort_values(ascending=False)
                .head(n_top)
                .reset_index()
            )
            top["Liaison"] = top[col_dep] + " -> " + top[col_arr]

            fig_top = px.bar(
                top, x=target, y="Liaison", orientation="h",
                labels={target: "Retard moyen (min)"},
                template="plotly_dark",
                color=target,
                color_continuous_scale="Reds",
            )
            fig_top.update_layout(height=max(400, n_top * 40), yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
            st.plotly_chart(fig_top, use_container_width=True)

    # =========================================================
    # PAGE 3 : CAUSES DE RETARD
    # =========================================================
    elif page == "Causes de retard":
        st.markdown('<p class="main-title">Analyse des causes de retard</p>', unsafe_allow_html=True)
        st.markdown("")

        cause_cols = [c for c in df.columns if c.startswith("Prct retard")]
        cause_names = {
            cause_cols[0]: "Causes externes",
            cause_cols[1]: "Infrastructure",
            cause_cols[2]: "Gestion trafic",
            cause_cols[3]: "Materiel roulant",
            cause_cols[4]: "Gestion gare",
            cause_cols[5]: "Voyageurs",
        }

        cause_means = {v: df[k].mean() for k, v in cause_names.items()}
        cause_df = pd.DataFrame(list(cause_means.items()), columns=["Cause", "Pourcentage moyen"])
        cause_df = cause_df.sort_values("Pourcentage moyen", ascending=True)

        fig_causes = px.bar(
            cause_df, x="Pourcentage moyen", y="Cause", orientation="h",
            template="plotly_dark",
            color="Pourcentage moyen",
            color_continuous_scale="Viridis",
        )
        fig_causes.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig_causes, use_container_width=True)

        # Evolution des causes par annee
        st.markdown('<p class="section-header">Evolution des causes par annee</p>', unsafe_allow_html=True)
        cause_year = df.groupby("Annee")[cause_cols].mean().reset_index()
        cause_year_melted = cause_year.melt(id_vars="Annee", var_name="Cause", value_name="Pourcentage")
        cause_year_melted["Cause"] = cause_year_melted["Cause"].map(cause_names)

        fig_cause_year = px.area(
            cause_year_melted, x="Annee", y="Pourcentage", color="Cause",
            template="plotly_dark",
            labels={"Pourcentage": "% moyen"},
        )
        fig_cause_year.update_layout(height=500)
        st.plotly_chart(fig_cause_year, use_container_width=True)

        # Avant/apres COVID
        st.markdown('<p class="section-header">Impact COVID : avant vs apres 2020</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        avant = df[df["Annee"] < 2020][target].mean()
        apres = df[df["Annee"] >= 2021][target].mean()
        with col1:
            st.metric("Retard moyen avant 2020", f"{avant:.1f} min")
        with col2:
            delta = apres - avant
            st.metric("Retard moyen apres 2020", f"{apres:.1f} min", delta=f"{delta:+.1f} min", delta_color="inverse")

    # =========================================================
    # PAGE 4 : COMPARAISON DES MODELES
    # =========================================================
    elif page == "Comparaison des modeles":
        st.markdown('<p class="main-title">Comparaison des modeles</p>', unsafe_allow_html=True)
        st.markdown("")

        if MODEL_METRICS_FILE.exists():
            metrics_df = pd.read_csv(MODEL_METRICS_FILE)

            # Highlight best model
            best_idx = metrics_df["R2"].idxmax()
            best_name = metrics_df.loc[best_idx, "model_name"]
            best_r2 = metrics_df.loc[best_idx, "R2"]
            best_mae = metrics_df.loc[best_idx, "MAE"]

            st.success(f"Meilleur modele : **{best_name}** avec un R2 de {best_r2:.4f} et une MAE de {best_mae:.2f} min")

            tab1, tab2 = st.tabs(["Tableau comparatif", "Visualisations"])

            with tab1:
                display_df = metrics_df[["model_name", "MAE", "RMSE", "R2"]].copy()
                display_df.columns = ["Modele", "MAE (min)", "RMSE (min)", "R2"]
                display_df = display_df.sort_values("R2", ascending=False)
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                st.markdown(
                    "**MAE** = erreur moyenne en minutes. "
                    "**RMSE** = erreur penalisant les gros ecarts. "
                    "**R2** = qualite globale (1 = parfait, 0 = aussi bon que la moyenne)."
                )

            with tab2:
                fig_r2 = px.bar(
                    metrics_df.sort_values("R2"), x="R2", y="model_name",
                    orientation="h",
                    labels={"R2": "R2 Score", "model_name": ""},
                    template="plotly_dark",
                    color="R2",
                    color_continuous_scale="Viridis",
                )
                fig_r2.update_layout(height=350, coloraxis_showscale=False)
                st.plotly_chart(fig_r2, use_container_width=True)

                fig_mae = px.bar(
                    metrics_df.sort_values("MAE"), x="MAE", y="model_name",
                    orientation="h",
                    labels={"MAE": "MAE (minutes)", "model_name": ""},
                    template="plotly_dark",
                    color="MAE",
                    color_continuous_scale="Reds_r",
                )
                fig_mae.update_layout(height=350, coloraxis_showscale=False)
                st.plotly_chart(fig_mae, use_container_width=True)
        else:
            st.info("Lancez python scripts/main.py pour generer les resultats.")

    # =========================================================
    # PAGE 5 : PREDICTEUR INTERACTIF
    # =========================================================
    elif page == "Predicteur interactif":
        st.markdown('<p class="main-title">Predicteur interactif</p>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Selectionnez une liaison et un mois pour predire le retard moyen</p>', unsafe_allow_html=True)
        st.markdown("")

        le_depart = joblib.load(MODELS_DIR / "le_depart.joblib")
        le_arrivee = joblib.load(MODELS_DIR / "le_arrivee.joblib")
        best_model = joblib.load(MODELS_DIR / "gradient_boosting.joblib")

        col1, col2 = st.columns(2)
        with col1:
            gare_dep = st.selectbox("Gare de depart", sorted(le_depart.classes_), index=list(sorted(le_depart.classes_)).index("PARIS LYON") if "PARIS LYON" in le_depart.classes_ else 0)
        with col2:
            gare_arr = st.selectbox("Gare d arrivee", sorted(le_arrivee.classes_), index=list(sorted(le_arrivee.classes_)).index("MARSEILLE ST CHARLES") if "MARSEILLE ST CHARLES" in le_arrivee.classes_ else 0)

        mois = st.select_slider(
            "Mois",
            options=list(range(1, 13)),
            format_func=lambda x: ["Jan", "Fev", "Mar", "Avr", "Mai", "Juin", "Juil", "Aout", "Sep", "Oct", "Nov", "Dec"][x-1],
            value=6,
        )

        mask = (df[col_dep] == gare_dep) & (df[col_arr] == gare_arr)
        duree_col = [c for c in df.columns if "moyenne du trajet" in c][0]
        circ_col = [c for c in df.columns if "circulations" in c][0]
        duree = df.loc[mask, duree_col]
        duree_val = int(duree.mean()) if len(duree) > 0 else 120
        circulations = df.loc[mask, circ_col]
        circ_val = int(circulations.mean()) if len(circulations) > 0 else 300

        if st.button("Predire le retard", type="primary", use_container_width=True):
            features = np.array([[
                le_depart.transform([gare_dep])[0],
                le_arrivee.transform([gare_arr])[0],
                duree_val,
                circ_val,
                0,
                2025,
                mois,
            ]])
            prediction = best_model.predict(features)[0]

            st.markdown("")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Retard predit", f"{prediction:.1f} min")
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
                template="plotly_dark",
            )
            fig_hist.update_traces(line_color="#4a90d9", line_width=2)
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


if __name__ == "__main__":
    build_app()
