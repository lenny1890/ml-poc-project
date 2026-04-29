"""Dataset loading for SNCF TGV delay prediction."""

from __future__ import annotations
from typing import Any
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import DATA_DIR


def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    df = pd.read_csv(DATA_DIR / "sncf_retards.csv", sep=";")

    cols_to_drop = [c for c in df.columns if "Commentaire" in c]
    df = df.drop(columns=cols_to_drop)
    df = df.drop(columns=["Service"])

    df["Annee"] = df["Date"].str[:4].astype(int)
    df["Mois"] = df["Date"].str[5:7].astype(int)
    df = df.drop(columns=["Date"])

    target = [c for c in df.columns if "Retard moyen de tous les trains" in c and "arriv" in c][0]
    df = df[df[target] >= 0]

    col_dep = [c for c in df.columns if c.startswith("Gare de d")][0]
    col_arr = [c for c in df.columns if c.startswith("Gare d") and "arriv" in c.lower()][0]

    le_dep = LabelEncoder()
    le_arr = LabelEncoder()
    df["Gare_depart_enc"] = le_dep.fit_transform(df[col_dep])
    df["Gare_arrivee_enc"] = le_arr.fit_transform(df[col_arr])

    col_duree = [c for c in df.columns if "moyenne du trajet" in c][0]
    col_circ = [c for c in df.columns if "circulations" in c][0]
    col_annul = [c for c in df.columns if "annul" in c][0]

    features = ["Gare_depart_enc", "Gare_arrivee_enc", col_duree, col_circ, col_annul, "Annee", "Mois"]

    X = df[features]
    y = df[target]

    return tuple(train_test_split(X, y, test_size=0.2, random_state=42))
