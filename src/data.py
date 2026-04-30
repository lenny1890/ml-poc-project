"""Dataset loading for SNCF TGV delay prediction.

This module implements the load_dataset_split() contract required by
scripts/main.py. It loads the SNCF Open Data CSV, preprocesses it,
and returns a train/test split ready for model evaluation.
"""

from __future__ import annotations
from typing import Any
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from config import DATA_DIR, MODELS_DIR

# Column names (exact match with SNCF CSV)
COL_DEPART = "Gare de d\xe9part"
COL_ARRIVEE = "Gare d'arriv\xe9e"
TARGET = "Retard moyen de tous les trains \xe0 l'arriv\xe9e"
COL_DUREE = "Dur\xe9e moyenne du trajet"
COL_CIRCULATIONS = "Nombre de circulations pr\xe9vues"
COL_ANNULATIONS = "Nombre de trains annul\xe9s"


def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    """Load the SNCF dataset and return (X_train, X_test, y_train, y_test).

    Features used:
        - Gare de depart (label encoded)
        - Gare d'arrivee (label encoded)
        - Duree moyenne du trajet (minutes)
        - Nombre de circulations prevues
        - Nombre de trains annules
        - Annee
        - Mois

    Target:
        - Retard moyen de tous les trains a l'arrivee (minutes)

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) with 80/20 split.
    """
    df = pd.read_csv(DATA_DIR / "sncf_retards.csv", sep=";", encoding="utf-8-sig")

    # Validate expected columns exist
    for col in [COL_DEPART, COL_ARRIVEE, TARGET, COL_DUREE, COL_CIRCULATIONS, COL_ANNULATIONS]:
        if col not in df.columns:
            raise KeyError(f"Colonne manquante dans le CSV: '{col}'")

    # Drop comment columns (all NaN)
    cols_to_drop = [c for c in df.columns if "Commentaire" in c]
    df = df.drop(columns=cols_to_drop)
    df = df.drop(columns=["Service"])

    # Extract year and month
    df["Annee"] = df["Date"].str[:4].astype(int)
    df["Mois"] = df["Date"].str[5:7].astype(int)
    df = df.drop(columns=["Date"])

    # Remove aberrant values (negative delays)
    df = df[df[TARGET] >= 0]

    # Encode station names
    le_dep = LabelEncoder()
    le_arr = LabelEncoder()
    df["Gare_depart_enc"] = le_dep.fit_transform(df[COL_DEPART])
    df["Gare_arrivee_enc"] = le_arr.fit_transform(df[COL_ARRIVEE])

    # Save encoders for Streamlit predictor
    joblib.dump(le_dep, MODELS_DIR / "le_depart.joblib")
    joblib.dump(le_arr, MODELS_DIR / "le_arrivee.joblib")

    # Feature selection
    features = [
        "Gare_depart_enc",
        "Gare_arrivee_enc",
        COL_DUREE,
        COL_CIRCULATIONS,
        COL_ANNULATIONS,
        "Annee",
        "Mois",
    ]

    X = df[features]
    y = df[TARGET]

    return tuple(train_test_split(X, y, test_size=0.2, random_state=42))
