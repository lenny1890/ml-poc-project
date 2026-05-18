"""Dataset loading for SNCF TGV delay prediction - V2 with enriched features.

QA Notes (2026-05-18)
---------------------
- LabelEncoders are fitted exclusively on the training partition to prevent
  vocabulary leakage from the test set into saved encoder artifacts.
- The CSV is read from the project root (PROJECT_ROOT) — the actual file
  location. config.DATA_DIR points to a 'data/' subdirectory that is empty.
- col_retard_dep ("Retard moyen de tous les trains au depart") and the derived
  pct_retard_depart are contemporaneous with the target (same batch of trips,
  same month). They are excluded by default to avoid label leakage in any
  forward-looking prediction context.
  Set INCLUDE_DEPARTURE_DELAY_FEATURES = True only if the use-case is
  contemporaneous explanation (not forward-looking prediction), and document
  that choice explicitly in the model card.
"""

from __future__ import annotations

import warnings
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import MODELS_DIR, PROJECT_ROOT

# Set to True only for contemporaneous-explanation use cases (see docstring).
INCLUDE_DEPARTURE_DELAY_FEATURES: bool = False


def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    # Read from project root — sncf_retards.csv lives there, not in data/
    csv_path = PROJECT_ROOT / "sncf_retards.csv"
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")

    # --- Column discovery ---
    def _find_col(cols: list[str], predicate, label: str) -> str:
        """Return the first column matching *predicate* or raise a descriptive KeyError."""
        matches = [c for c in cols if predicate(c)]
        if not matches:
            raise KeyError(
                f"Could not locate column '{label}' in the CSV. "
                f"Available columns: {cols}"
            )
        return matches[0]

    all_cols = df.columns.tolist()
    col_dep = _find_col(all_cols, lambda c: c.startswith("Gare de d"), "Gare de départ")
    col_arr = _find_col(all_cols, lambda c: "Gare" in c and "arriv" in c.lower(), "Gare d'arrivée")
    target = _find_col(
        all_cols,
        lambda c: "Retard moyen de tous les trains" in c and "arriv" in c,
        "Retard moyen … à l'arrivée (target)",
    )
    col_duree = _find_col(all_cols, lambda c: "moyenne du trajet" in c, "Durée moyenne du trajet")
    col_circ = _find_col(all_cols, lambda c: "circulations" in c, "Nombre de circulations prévues")
    col_annul = _find_col(all_cols, lambda c: "annul" in c.lower() and "Nombre" in c, "Nombre de trains annulés")
    col_retard_dep = _find_col(
        all_cols,
        lambda c: "Retard moyen de tous les trains" in c and "part" in c.lower(),
        "Retard moyen … au départ",
    )
    col_nb_retard_dep = _find_col(
        all_cols,
        lambda c: "Nombre de trains en retard au d" in c,
        "Nombre de trains en retard au départ",
    )
    cause_cols = [c for c in all_cols if c.startswith("Prct retard")]

    # --- Cleaning ---
    cols_drop = [c for c in df.columns if "Commentaire" in c] + ["Service"]
    df = df.drop(columns=cols_drop)

    df["Annee"] = df["Date"].str[:4].astype(int)
    df["Mois"] = df["Date"].str[5:7].astype(int)
    df = df.drop(columns=["Date"])
    df = df[df[target] >= 0].copy()

    # --- Derived features (row-level, computed before split) ---
    df["taux_annulation"] = df[col_annul] / df[col_circ] * 100
    df["mois_sin"] = np.sin(2 * np.pi * df["Mois"] / 12)
    df["mois_cos"] = np.cos(2 * np.pi * df["Mois"] / 12)
    df["pct_retard_depart"] = df[col_nb_retard_dep] / df[col_circ] * 100
    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    # Split on indices BEFORE encoding — encoders must never see test rows
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)
    train_df = df.loc[train_idx].copy()
    test_df = df.loc[test_idx].copy()

    # --- LabelEncoders fitted on train partition only ---
    le_dep = LabelEncoder().fit(train_df[col_dep])
    le_arr = LabelEncoder().fit(train_df[col_arr])

    def _safe_transform(le: LabelEncoder, series: pd.Series) -> pd.Series:
        """Map unseen station names to -1 instead of raising ValueError."""
        known = set(le.classes_)
        return series.map(lambda v: le.transform([v])[0] if v in known else -1)

    train_df["gare_dep_enc"] = _safe_transform(le_dep, train_df[col_dep])
    train_df["gare_arr_enc"] = _safe_transform(le_arr, train_df[col_arr])
    test_df["gare_dep_enc"] = _safe_transform(le_dep, test_df[col_dep])
    test_df["gare_arr_enc"] = _safe_transform(le_arr, test_df[col_arr])

    joblib.dump(le_dep, MODELS_DIR / "le_depart.joblib")
    joblib.dump(le_arr, MODELS_DIR / "le_arrivee.joblib")

    # --- Feature set (leakage features excluded by default) ---
    features = [
        "gare_dep_enc", "gare_arr_enc", col_duree,
        col_circ, col_annul, "Annee",
        "mois_sin", "mois_cos",
        "taux_annulation",
    ] + cause_cols

    if INCLUDE_DEPARTURE_DELAY_FEATURES:
        warnings.warn(
            "INCLUDE_DEPARTURE_DELAY_FEATURES=True: col_retard_dep and "
            "pct_retard_depart are contemporaneous with the target and "
            "introduce label leakage in a forward-looking prediction context. "
            "Document this design choice in the model card.",
            UserWarning,
            stacklevel=2,
        )
        features += [col_retard_dep, "pct_retard_depart"]

    return (
        train_df[features],
        test_df[features],
        train_df[target],
        test_df[target],
    )
