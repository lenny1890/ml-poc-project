"""Dataset loading for SNCF TGV delay prediction - V2 with enriched features."""

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from config import DATA_DIR, MODELS_DIR


def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    df = pd.read_csv(DATA_DIR / "sncf_retards.csv", sep=";", encoding="utf-8-sig")

    col_dep = [c for c in df.columns if c.startswith("Gare de d")][0]
    col_arr = [c for c in df.columns if "Gare" in c and "arriv" in c.lower()][0]
    target = [c for c in df.columns if "Retard moyen de tous les trains" in c and "arriv" in c][0]
    col_duree = [c for c in df.columns if "moyenne du trajet" in c][0]
    col_circ = [c for c in df.columns if "circulations" in c][0]
    col_annul = [c for c in df.columns if "annul" in c.lower() and "Nombre" in c][0]
    col_retard_dep = [c for c in df.columns if "Retard moyen de tous les trains" in c and "part" in c.lower()][0]
    col_nb_retard_dep = [c for c in df.columns if "Nombre de trains en retard au d" in c][0]
    cause_cols = [c for c in df.columns if c.startswith("Prct retard")]

    cols_drop = [c for c in df.columns if "Commentaire" in c] + ["Service"]
    df = df.drop(columns=cols_drop)

    df["Annee"] = df["Date"].str[:4].astype(int)
    df["Mois"] = df["Date"].str[5:7].astype(int)
    df = df.drop(columns=["Date"])

    df = df[df[target] >= 0]

    df["taux_annulation"] = df[col_annul] / df[col_circ] * 100
    df["mois_sin"] = np.sin(2 * np.pi * df["Mois"] / 12)
    df["mois_cos"] = np.cos(2 * np.pi * df["Mois"] / 12)
    df["pct_retard_depart"] = df[col_nb_retard_dep] / df[col_circ] * 100

    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)

    le_dep = LabelEncoder()
    le_arr = LabelEncoder()
    df["gare_dep_enc"] = le_dep.fit_transform(df[col_dep])
    df["gare_arr_enc"] = le_arr.fit_transform(df[col_arr])

    joblib.dump(le_dep, MODELS_DIR / "le_depart.joblib")
    joblib.dump(le_arr, MODELS_DIR / "le_arrivee.joblib")

    features = [
        "gare_dep_enc", "gare_arr_enc", col_duree,
        col_circ, col_annul, "Annee",
        "mois_sin", "mois_cos",
        "taux_annulation",
        col_retard_dep,
        "pct_retard_depart",
    ] + cause_cols

    X = df[features]
    y = df[target]

    return tuple(train_test_split(X, y, test_size=0.2, random_state=42))
