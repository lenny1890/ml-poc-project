"""Streamlit application for SNCF TGV delay prediction.

Enhanced dashboard with premium design, rich analytics,
and interactive prediction capabilities.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go

from config import DATA_DIR, MODELS_DIR, MODEL_METRICS_FILE


# ─────────────────────────────────────────────
# DESIGN SYSTEM
# ─────────────────────────────────────────────

COLORS = {
    "primary": "#0057B8",
    "primary_light": "#3B82F6",
    "accent": "#F59E0B",
    "accent_red": "#EF4444",
    "accent_green": "#10B981",
    "bg_dark": "#0A0E1A",
    "bg_card": "#111827",
    "bg_card_hover": "#1F2937",
    "text_primary": "#F9FAFB",
    "text_secondary": "#9CA3AF",
    "border": "#1F2937",
}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Global overrides */
.stApp {
    font-family: 'DM Sans', sans-serif;
}

/* Hero header */
.hero-container {
    background: linear-gradient(135deg, #0A0E1A 0%, #111827 50%, #0A0E1A 100%);
    border: 1px solid #1F2937;
    border-radius: 16px;
    padding: 48px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(0,87,184,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-container::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(245,158,11,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    color: #F9FAFB;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin: 0 0 8px 0;
    position: relative;
    z-index: 1;
}
.hero-title span {
    background: linear-gradient(135deg, #3B82F6, #0057B8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    font-size: 1.15rem;
    color: #9CA3AF;
    font-weight: 400;
    line-height: 1.5;
    max-width: 600px;
    position: relative;
    z-index: 1;
}

/* KPI Cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 32px;
}
.kpi-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 12px;
    padding: 24px;
    transition: all 0.3s ease;
}
.kpi-card:hover {
    border-color: #3B82F6;
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(59, 130, 246, 0.1);
}
.kpi-icon {
    font-size: 1.5rem;
    margin-bottom: 12px;
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #F9FAFB;
    margin: 0;
    line-height: 1.2;
}
.kpi-label {
    font-size: 0.8rem;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 6px;
    font-weight: 500;
}
.kpi-delta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    margin-top: 8px;
    padding: 3px 8px;
    border-radius: 6px;
    display: inline-block;
}
.kpi-delta.positive { background: rgba(16,185,129,0.15); color: #10B981; }
.kpi-delta.negative { background: rgba(239,68,68,0.15); color: #EF4444; }

/* Section headers */
.section-tag {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    color: #3B82F6;
    margin-bottom: 4px;
}
.section-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #F9FAFB;
    letter-spacing: -0.01em;
    margin: 0 0 8px 0;
}
.section-desc {
    font-size: 0.95rem;
    color: #9CA3AF;
    line-height: 1.6;
    margin-bottom: 24px;
    max-width: 700px;
}
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, #1F2937, #3B82F6, #1F2937);
    margin: 40px 0;
    border: none;
}

/* Insight cards */
.insight-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    border-left: 3px solid #3B82F6;
}
.insight-card.warning { border-left-color: #F59E0B; }
.insight-card.danger { border-left-color: #EF4444; }
.insight-card.success { border-left-color: #10B981; }
.insight-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #F9FAFB;
    margin-bottom: 6px;
}
.insight-text {
    font-size: 0.88rem;
    color: #9CA3AF;
    line-height: 1.5;
}

/* Prediction result */
.prediction-result {
    background: linear-gradient(135deg, #0A0E1A, #111827);
    border: 1px solid #3B82F6;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    margin: 24px 0;
}
.prediction-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 3.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #3B82F6, #0057B8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.prediction-unit {
    font-size: 1.1rem;
    color: #9CA3AF;
    margin-top: 4px;
}
.prediction-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #3B82F6;
    margin-bottom: 8px;
    font-weight: 600;
}

/* Model badge */
.model-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.model-badge.best { background: rgba(16,185,129,0.15); color: #10B981; border: 1px solid rgba(16,185,129,0.3); }
.model-badge.good { background: rgba(59,130,246,0.15); color: #3B82F6; border: 1px solid rgba(59,130,246,0.3); }
.model-badge.baseline { background: rgba(156,163,175,0.15); color: #9CA3AF; border: 1px solid rgba(156,163,175,0.3); }

/* Footer */
.footer {
    text-align: center;
    padding: 32px 0;
    margin-top: 48px;
    border-top: 1px solid #1F2937;
    color: #6B7280;
    font-size: 0.85rem;
}
</style>
"""

PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#9CA3AF"),
    margin=dict(l=20, r=20, t=40, b=20),
)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

@st.cache_data
def load_data():
    """Load and prepare the SNCF dataset for display."""
    df = pd.read_csv(DATA_DIR / "sncf_retards.csv", sep=";", encoding="utf-8-sig")

    target = [c for c in df.columns if "Retard moyen de tous les trains" in c and "arriv" in c][0]
    col_arr = [c for c in df.columns if "Gare" in c and "arriv" in c.lower()][0]
    col_dep = [c for c in df.columns if c.startswith("Gare de d")][0]

    df["Annee"] = df["Date"].str[:4].astype(int)
    df["Mois"] = df["Date"].str[5:7].astype(int)

    mois_map = {1:"Jan",2:"Fev",3:"Mar",4:"Avr",5:"Mai",6:"Juin",
                7:"Juil",8:"Aout",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    df["Mois_nom"] = df["Mois"].map(mois_map)

    cause_cols = [c for c in df.columns if c.startswith("Prct retard")]
    return df, target, col_dep, col_arr, cause_cols


def kpi_card(icon, value, label, delta=None, delta_type="positive"):
    """Render a single KPI card."""
    delta_html = ""
    if delta is not None:
        delta_html = f'<div class="kpi-delta {delta_type}">{delta}</div>'
    return f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <p class="kpi-value">{value}</p>
        <p class="kpi-label">{label}</p>
        {delta_html}
    </div>
    """


def section_header(tag, title, desc=""):
    """Render a section header with tag, title, and description."""
    desc_html = f'<p class="section-desc">{desc}</p>' if desc else ""
    st.markdown(f"""
    <div style="margin-top: 16px;">
        <p class="section-tag">{tag}</p>
        <h2 class="section-title">{title}</h2>
        {desc_html}
    </div>
    """, unsafe_allow_html=True)


def insight_card(title, text, variant=""):
    """Render an insight card."""
    cls = f"insight-card {variant}" if variant else "insight-card"
    st.markdown(f"""
    <div class="{cls}">
        <div class="insight-title">{title}</div>
        <div class="insight-text">{text}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────

def page_overview(df, target, col_dep, col_arr):
    """Dashboard overview with KPIs and timeline."""

    # Hero
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">Prediction des retards <span>TGV</span></h1>
        <p class="hero-subtitle">
            Dashboard analytique et predictif base sur les donnees Open Data SNCF.
            12 000+ observations de regularite mensuelles, 4 modeles de Machine Learning compares.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    retard_moyen = df[target].mean()
    nb_liaisons = df.groupby([col_dep, col_arr]).ngroups
    nb_gares = df[col_dep].nunique()

    # Trend: compare last year vs previous
    last_year = df["Annee"].max()
    prev_year = last_year - 1
    retard_last = df[df["Annee"] == last_year][target].mean()
    retard_prev = df[df["Annee"] == prev_year][target].mean()
    trend = retard_last - retard_prev
    trend_pct = (trend / retard_prev * 100) if retard_prev != 0 else 0

    # Best and worst month
    mois_stats = df.groupby("Mois")[target].mean()
    best_month_idx = mois_stats.idxmin()
    worst_month_idx = mois_stats.idxmax()
    mois_noms = {1:"Janvier",2:"Fevrier",3:"Mars",4:"Avril",5:"Mai",6:"Juin",
                 7:"Juillet",8:"Aout",9:"Septembre",10:"Octobre",11:"Novembre",12:"Decembre"}

    st.markdown(f"""
    <div class="kpi-grid">
        {kpi_card("&#9201;", f"{retard_moyen:.1f} min", "Retard moyen global")}
        {kpi_card("&#128652;", f"{nb_liaisons}", "Liaisons analysees")}
        {kpi_card("&#127971;", f"{nb_gares}", "Gares couvertes")}
        {kpi_card("&#128200;", f"{trend_pct:+.1f}%", f"Tendance {last_year} vs {prev_year}",
                  delta=f"{trend:+.1f} min", delta_type="negative" if trend > 0 else "positive")}
    </div>
    """, unsafe_allow_html=True)

    # Timeline
    section_header("TEMPORALITE", "Evolution mensuelle des retards",
                   "Retard moyen a l arrivee sur l ensemble des liaisons TGV, agreges par mois.")

    monthly = df.groupby(["Annee", "Mois"])[target].mean().reset_index()
    monthly["Date"] = pd.to_datetime(
        monthly["Annee"].astype(str) + "-" + monthly["Mois"].astype(str).str.zfill(2) + "-01"
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["Date"], y=monthly[target],
        mode="lines",
        line=dict(color="#3B82F6", width=2),
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.08)",
        hovertemplate="<b>%{x|%b %Y}</b><br>Retard: %{y:.1f} min<extra></extra>",
    ))

    # Add rolling average
    monthly["MA6"] = monthly[target].rolling(6, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=monthly["Date"], y=monthly["MA6"],
        mode="lines",
        line=dict(color="#F59E0B", width=2, dash="dot"),
        name="Moyenne mobile 6 mois",
        hovertemplate="<b>Moyenne mobile</b><br>%{y:.1f} min<extra></extra>",
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=420,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Retard moyen (min)", gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Insights
    col1, col2 = st.columns(2)
    with col1:
        insight_card(
            f"Pire mois : {mois_noms[worst_month_idx]}",
            f"Avec un retard moyen de {mois_stats[worst_month_idx]:.1f} minutes, "
            f"{mois_noms[worst_month_idx]} est le mois le plus critique pour les retards TGV. "
            f"Cela s explique par les conditions meteorologiques et l affluence.",
            "danger"
        )
    with col2:
        insight_card(
            f"Meilleur mois : {mois_noms[best_month_idx]}",
            f"Le mois de {mois_noms[best_month_idx]} presente le retard moyen le plus bas "
            f"({mois_stats[best_month_idx]:.1f} min), beneficiant d un trafic plus faible "
            f"et de conditions favorables.",
            "success"
        )


def page_exploration(df, target, col_dep, col_arr):
    """Data exploration with multiple visualization types."""

    section_header("EXPLORATION", "Analyse des donnees",
                   "Plongee dans les patterns de retard par saisonnalite, annee et liaison.")

    tab1, tab2, tab3, tab4 = st.tabs(["Saisonnalite", "Tendance annuelle", "Top liaisons", "Distribution"])

    mois_noms_short = ["Jan","Fev","Mar","Avr","Mai","Juin","Juil","Aout","Sep","Oct","Nov","Dec"]

    with tab1:
        mois_data = df.groupby("Mois")[target].agg(["mean","median","std","count"]).reset_index()
        mois_data["Mois_nom"] = [mois_noms_short[i-1] for i in mois_data["Mois"]]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=mois_data["Mois_nom"], y=mois_data["mean"],
            marker=dict(
                color=mois_data["mean"],
                colorscale=[[0, "#10B981"], [0.5, "#F59E0B"], [1, "#EF4444"]],
                line=dict(width=0),
                cornerradius=6,
            ),
            error_y=dict(type="data", array=mois_data["std"], color="rgba(255,255,255,0.2)", thickness=1),
            hovertemplate="<b>%{x}</b><br>Moyenne: %{y:.1f} min<br>Ecart-type: %{error_y.array:.1f}<extra></extra>",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=450,
                          xaxis=dict(showgrid=False),
                          yaxis=dict(title="Retard moyen (min)", gridcolor="rgba(255,255,255,0.05)"))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            insight_card("Saisonnalite marquee",
                         "L ete (juillet) et l hiver (janvier-fevrier) concentrent les pics de retard. "
                         "L affluence estivale et les intemperies hivernales en sont les causes principales.",
                         "warning")
        with col2:
            # Median vs mean comparison
            ecart_max = mois_data.loc[mois_data["mean"].idxmax()]
            insight_card("Forte dispersion en " + ecart_max["Mois_nom"],
                         f"L ecart-type atteint {ecart_max['std']:.1f} min en {ecart_max['Mois_nom']}, "
                         "ce qui indique une grande variabilite des retards ce mois-la. "
                         "Certaines liaisons sont beaucoup plus touchees que d autres.",
                         "danger")

    with tab2:
        annee_data = df.groupby("Annee")[target].agg(["mean","count"]).reset_index()

        fig = go.Figure()
        colors = ["#3B82F6" if y != df["Annee"].max() else "#F59E0B" for y in annee_data["Annee"]]
        fig.add_trace(go.Bar(
            x=annee_data["Annee"], y=annee_data["mean"],
            marker=dict(color=colors, line=dict(width=0), cornerradius=6),
            hovertemplate="<b>%{x}</b><br>Retard moyen: %{y:.1f} min<extra></extra>",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=450,
                          xaxis=dict(showgrid=False, dtick=1),
                          yaxis=dict(title="Retard moyen (min)", gridcolor="rgba(255,255,255,0.05)"))
        st.plotly_chart(fig, use_container_width=True)

        # COVID analysis
        avant_covid = df[df["Annee"] < 2020][target].mean()
        apres_covid = df[df["Annee"] >= 2021][target].mean()
        delta = apres_covid - avant_covid
        insight_card("Impact post-COVID",
                     f"Le retard moyen est passe de {avant_covid:.1f} min (pre-2020) a "
                     f"{apres_covid:.1f} min (post-2020), soit une hausse de {delta:+.1f} min. "
                     "La reprise du trafic et les travaux d infrastructure expliquent cette degradation.",
                     "warning" if delta > 0 else "success")

    with tab3:
        n_top = st.slider("Nombre de liaisons", 5, 25, 10, key="top_slider")
        top = (df.groupby([col_dep, col_arr])[target]
               .agg(["mean","count","std"])
               .sort_values("mean", ascending=False)
               .head(n_top)
               .reset_index())
        top["Liaison"] = top[col_dep] + " \\u2192 " + top[col_arr]
        top["label"] = top.apply(lambda r: f"{r['mean']:.1f} min ({r['count']} obs)", axis=1)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=top["Liaison"], x=top["mean"],
            orientation="h",
            marker=dict(
                color=top["mean"],
                colorscale=[[0, "#F59E0B"], [1, "#EF4444"]],
                line=dict(width=0),
                cornerradius=6,
            ),
            text=top["label"],
            textposition="inside",
            textfont=dict(color="white", size=11),
            hovertemplate="<b>%{y}</b><br>Retard: %{x:.1f} min<extra></extra>",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=max(400, n_top * 45),
                          yaxis=dict(autorange="reversed", showgrid=False),
                          xaxis=dict(title="Retard moyen (min)", gridcolor="rgba(255,255,255,0.05)"))
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df[target],
            nbinsx=80,
            marker=dict(color="#3B82F6", line=dict(width=0)),
            hovertemplate="Retard: %{x:.1f} min<br>Frequence: %{y}<extra></extra>",
        ))
        fig.add_vline(x=df[target].median(), line_dash="dash", line_color="#F59E0B",
                      annotation_text=f"Mediane: {df[target].median():.1f} min",
                      annotation_font_color="#F59E0B")
        fig.update_layout(**PLOTLY_LAYOUT, height=450,
                          xaxis=dict(title="Retard moyen (min)", showgrid=False),
                          yaxis=dict(title="Frequence", gridcolor="rgba(255,255,255,0.05)"))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Moyenne", f"{df[target].mean():.1f} min")
        with col2:
            st.metric("Mediane", f"{df[target].median():.1f} min")
        with col3:
            st.metric("Ecart-type", f"{df[target].std():.1f} min")


def page_causes(df, target, col_dep, col_arr, cause_cols):
    """Analysis of delay causes."""

    section_header("DIAGNOSTIC", "Causes de retard",
                   "Decomposition des retards par type de cause pour comprendre les leviers d amelioration.")

    labels = ["Causes externes", "Infrastructure", "Gestion trafic",
              "Materiel roulant", "Gestion gare", "Voyageurs"]
    cause_names = {}
    for i, col in enumerate(cause_cols):
        if i < len(labels):
            cause_names[col] = labels[i]

    # Donut chart
    cause_means = {v: df[k].mean() for k, v in cause_names.items()}
    cause_df = pd.DataFrame(list(cause_means.items()), columns=["Cause", "Pct"])
    cause_df = cause_df.sort_values("Pct", ascending=False)

    col1, col2 = st.columns([1, 1])

    with col1:
        colors_donut = ["#EF4444", "#3B82F6", "#F59E0B", "#8B5CF6", "#10B981", "#EC4899"]
        fig = go.Figure(data=[go.Pie(
            labels=cause_df["Cause"], values=cause_df["Pct"],
            hole=0.55,
            marker=dict(colors=colors_donut[:len(cause_df)], line=dict(width=2, color="#0A0E1A")),
            textinfo="label+percent",
            textfont=dict(size=11),
            hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
        )])
        fig.update_layout(**PLOTLY_LAYOUT, height=420, showlegend=False,
                          annotations=[dict(text="Causes", x=0.5, y=0.5,
                                            font_size=14, font_color="#9CA3AF", showarrow=False)])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top_cause = cause_df.iloc[0]
        second_cause = cause_df.iloc[1]
        insight_card(f"Cause principale : {top_cause['Cause']}",
                     f"Avec {top_cause['Pct']:.1f}% des retards en moyenne, "
                     f"les {top_cause['Cause'].lower()} representent la premiere source de perturbation. "
                     f"Suivies par {second_cause['Cause'].lower()} ({second_cause['Pct']:.1f}%).",
                     "danger")
        insight_card("Leviers d amelioration",
                     "Les causes liees a l infrastructure et a la gestion du trafic sont sous le controle "
                     "de SNCF Reseau. Les causes externes (meteo, actes de malveillance) sont plus difficiles "
                     "a anticiper mais peuvent etre integrees dans un modele predictif.",
                     "")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Evolution by year
    section_header("EVOLUTION", "Causes de retard par annee")
    cause_year = df.groupby("Annee")[cause_cols].mean().reset_index()
    cause_year_melted = cause_year.melt(id_vars="Annee", var_name="Cause_raw", value_name="Pct")
    cause_year_melted["Cause"] = cause_year_melted["Cause_raw"].map(cause_names)

    fig = px.area(cause_year_melted, x="Annee", y="Pct", color="Cause",
                  template=PLOTLY_TEMPLATE,
                  color_discrete_sequence=colors_donut[:len(cause_names)],
                  labels={"Pct": "% moyen", "Annee": ""})
    fig.update_layout(**PLOTLY_LAYOUT, height=450)
    st.plotly_chart(fig, use_container_width=True)

    # COVID comparison
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    section_header("COMPARAISON", "Avant vs apres COVID")

    col1, col2, col3 = st.columns([1, 1, 1])
    avant = df[df["Annee"] < 2020][target].mean()
    apres = df[df["Annee"] >= 2021][target].mean()
    delta = apres - avant

    with col1:
        st.metric("Retard moyen pre-2020", f"{avant:.1f} min")
    with col2:
        st.metric("Retard moyen post-2020", f"{apres:.1f} min",
                  delta=f"{delta:+.1f} min", delta_color="inverse")
    with col3:
        pct_change = (delta / avant * 100) if avant != 0 else 0
        st.metric("Variation", f"{pct_change:+.1f}%", delta_color="inverse")


def page_models(df, target):
    """Model comparison dashboard."""

    section_header("PERFORMANCE", "Comparaison des modeles",
                   "4 algorithmes de regression evalues sur le meme jeu de test (20% des donnees, random_state=42).")

    if not MODEL_METRICS_FILE.exists():
        st.info("Lancez `python scripts/main.py` pour generer les resultats.")
        return

    metrics_df = pd.read_csv(MODEL_METRICS_FILE)
    best_idx = metrics_df["R2"].idxmax()

    # Model cards
    for _, row in metrics_df.sort_values("R2", ascending=False).iterrows():
        is_best = row.name == best_idx
        badge = "best" if is_best else ("good" if row["R2"] > 0.3 else "baseline")
        badge_text = "MEILLEUR" if is_best else ("BON" if row["R2"] > 0.3 else "BASELINE")

        st.markdown(f"""
        <div class="insight-card {'success' if is_best else ''}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div class="insight-title">{row['model_name']}
                        <span class="model-badge {badge}">{badge_text}</span>
                    </div>
                    <div class="insight-text" style="margin-top: 8px;">
                        MAE: <b style="color:#F9FAFB">{row['MAE']:.2f} min</b> &nbsp;|&nbsp;
                        RMSE: <b style="color:#F9FAFB">{row['RMSE']:.2f} min</b> &nbsp;|&nbsp;
                        R2: <b style="color:#F9FAFB">{row['R2']:.4f}</b>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        ms = metrics_df.sort_values("R2")
        colors_r2 = ["#10B981" if i == ms["R2"].idxmax() else "#3B82F6" for i in ms.index]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=ms["model_name"], x=ms["R2"], orientation="h",
            marker=dict(color=colors_r2, line=dict(width=0), cornerradius=6),
            hovertemplate="<b>%{y}</b><br>R2: %{x:.4f}<extra></extra>",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=300, title="R2 Score (plus haut = meilleur)",
                          xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                          yaxis=dict(showgrid=False))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        ms2 = metrics_df.sort_values("MAE", ascending=False)
        colors_mae = ["#10B981" if i == ms2["MAE"].idxmin() else "#EF4444" for i in ms2.index]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=ms2["model_name"], x=ms2["MAE"], orientation="h",
            marker=dict(color=colors_mae, line=dict(width=0), cornerradius=6),
            hovertemplate="<b>%{y}</b><br>MAE: %{x:.2f} min<extra></extra>",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=300, title="MAE en minutes (plus bas = meilleur)",
                          xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                          yaxis=dict(showgrid=False))
        st.plotly_chart(fig, use_container_width=True)

    # Metrics explanation
    st.markdown("")
    insight_card("Comprendre les metriques",
                 "**MAE** (Mean Absolute Error) : erreur moyenne en minutes, facile a interpreter. "
                 "**RMSE** (Root Mean Squared Error) : penalise davantage les grosses erreurs. "
                 "**R2** : proportion de la variance expliquee (1 = parfait, 0 = aussi bon que la moyenne).",
                 "")


def page_predictor(df, target, col_dep, col_arr):
    """Interactive delay predictor."""

    section_header("PREDICTION", "Predicteur interactif",
                   "Selectionnez une liaison et un mois pour obtenir une estimation du retard moyen.")

    le_depart = joblib.load(MODELS_DIR / "le_depart.joblib")
    le_arrivee = joblib.load(MODELS_DIR / "le_arrivee.joblib")
    best_model = joblib.load(MODELS_DIR / "gradient_boosting.joblib")

    dep_list = sorted(le_depart.classes_)
    arr_list = sorted(le_arrivee.classes_)
    default_dep = dep_list.index("PARIS LYON") if "PARIS LYON" in dep_list else 0
    default_arr = arr_list.index("MARSEILLE ST CHARLES") if "MARSEILLE ST CHARLES" in arr_list else 0

    col1, col2 = st.columns(2)
    with col1:
        gare_dep = st.selectbox("Gare de depart", dep_list, index=default_dep)
    with col2:
        gare_arr = st.selectbox("Gare d arrivee", arr_list, index=default_arr)

    mois_noms = ["Janvier","Fevrier","Mars","Avril","Mai","Juin",
                 "Juillet","Aout","Septembre","Octobre","Novembre","Decembre"]
    mois = st.select_slider("Mois", options=list(range(1,13)),
                            format_func=lambda x: mois_noms[x-1], value=6)

    mask = (df[col_dep] == gare_dep) & (df[col_arr] == gare_arr)
    duree_col = [c for c in df.columns if "moyenne du trajet" in c][0]
    circ_col = [c for c in df.columns if "circulations" in c][0]
    annul_col = [c for c in df.columns if "annul" in c.lower()][0]

    duree = df.loc[mask, duree_col]
    duree_val = int(duree.mean()) if len(duree) > 0 else 120
    circulations = df.loc[mask, circ_col]
    circ_val = int(circulations.mean()) if len(circulations) > 0 else 300

    if st.button("Lancer la prediction", type="primary", use_container_width=True):
        features = np.array([[
            le_depart.transform([gare_dep])[0],
            le_arrivee.transform([gare_arr])[0],
            duree_val, circ_val, 0, 2025, mois,
        ]])
        prediction = best_model.predict(features)[0]

        st.markdown(f"""
        <div class="prediction-result">
            <p class="prediction-label">Retard moyen predit</p>
            <p class="prediction-value">{prediction:.1f}</p>
            <p class="prediction-unit">minutes de retard estime</p>
        </div>
        """, unsafe_allow_html=True)

        # Context metrics
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Duree du trajet", f"{duree_val} min")
        with c2:
            st.metric("Circulations mensuelles", f"{circ_val}")
        with c3:
            historique_retard = df.loc[mask, target].mean() if len(df.loc[mask]) > 0 else 0
            delta_pred = prediction - historique_retard
            st.metric("Historique moyen", f"{historique_retard:.1f} min",
                      delta=f"{delta_pred:+.1f} min vs prediction", delta_color="inverse")

    # Historical data
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    historique = df.loc[mask].copy()
    if len(historique) > 0:
        section_header("HISTORIQUE", f"{gare_dep} \\u2192 {gare_arr}",
                       f"Evolution des retards sur cette liaison depuis {historique['Annee'].min()}.")

        hist = historique.groupby(["Annee", "Mois"])[target].mean().reset_index()
        hist["Date"] = pd.to_datetime(
            hist["Annee"].astype(str) + "-" + hist["Mois"].astype(str).str.zfill(2) + "-01"
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist["Date"], y=hist[target],
            mode="lines+markers",
            line=dict(color="#3B82F6", width=2),
            marker=dict(size=4, color="#3B82F6"),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.05)",
            hovertemplate="<b>%{x|%b %Y}</b><br>Retard: %{y:.1f} min<extra></extra>",
        ))

        # Add average line
        avg = historique[target].mean()
        fig.add_hline(y=avg, line_dash="dash", line_color="#F59E0B",
                      annotation_text=f"Moyenne: {avg:.1f} min",
                      annotation_font_color="#F59E0B")

        fig.update_layout(**PLOTLY_LAYOUT, height=400,
                          xaxis=dict(showgrid=False),
                          yaxis=dict(title="Retard moyen (min)", gridcolor="rgba(255,255,255,0.05)"))
        st.plotly_chart(fig, use_container_width=True)

        # Stats grid
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("Retard moyen", f"{historique[target].mean():.1f} min")
        with s2:
            st.metric("Retard max", f"{historique[target].max():.1f} min")
        with s3:
            st.metric("Retard min", f"{historique[target].min():.1f} min")
        with s4:
            st.metric("Trains annules/mois", f"{historique[annul_col].mean():.0f}")

        # Monthly heatmap for this liaison
        if len(historique) > 12:
            section_header("HEATMAP", "Retards par mois et par annee")
            pivot = historique.pivot_table(values=target, index="Annee", columns="Mois", aggfunc="mean")
            mois_short = ["Jan","Fev","Mar","Avr","Mai","Juin","Juil","Aout","Sep","Oct","Nov","Dec"]
            pivot.columns = [mois_short[m-1] for m in pivot.columns]

            fig = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale=[[0, "#0A0E1A"], [0.3, "#3B82F6"], [0.7, "#F59E0B"], [1, "#EF4444"]],
                hovertemplate="<b>%{y} - %{x}</b><br>Retard: %{z:.1f} min<extra></extra>",
                colorbar=dict(title="min"),
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=350,
                              yaxis=dict(dtick=1, showgrid=False),
                              xaxis=dict(showgrid=False))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Aucune donnee historique pour cette liaison. Essayez une autre combinaison.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def build_app() -> None:
    """Main app entry point."""

    st.set_page_config(
        page_title="TGV Delay Predictor",
        page_icon="\U0001F684",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    df, target, col_dep, col_arr, cause_cols = load_data()

    with st.sidebar:
        st.markdown("""
        <div style="padding: 16px 0 24px 0;">
            <div style="font-size: 1.4rem; font-weight: 700; color: #F9FAFB; letter-spacing: -0.01em;">
                \U0001F684 TGV Predictor
            </div>
            <div style="font-size: 0.8rem; color: #6B7280; margin-top: 4px;">
                Machine Learning Dashboard
            </div>
        </div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            ["\U0001F4CA Vue d ensemble",
             "\U0001F50D Exploration",
             "\U000026A0 Causes de retard",
             "\U0001F916 Modeles",
             "\U0001F3AF Predicteur"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("""
        <div style="font-size: 0.78rem; color: #6B7280; line-height: 1.6;">
            <b style="color: #9CA3AF;">Source</b><br>
            SNCF Open Data<br>
            Regularite mensuelle TGV<br><br>
            <b style="color: #9CA3AF;">Projet</b><br>
            Albert School x Mines Paris PSL<br>
            Lenny Sebban
        </div>
        """, unsafe_allow_html=True)

    # Route to page
    if "Vue d ensemble" in page:
        page_overview(df, target, col_dep, col_arr)
    elif "Exploration" in page:
        page_exploration(df, target, col_dep, col_arr)
    elif "Causes" in page:
        page_causes(df, target, col_dep, col_arr, cause_cols)
    elif "Modeles" in page:
        page_models(df, target)
    elif "Predicteur" in page:
        page_predictor(df, target, col_dep, col_arr)

    # Footer
    st.markdown("""
    <div class="footer">
        Projet ML Proof of Concept &bull; Albert School x Mines Paris PSL &bull; Lenny Sebban &bull; 2025
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    build_app()