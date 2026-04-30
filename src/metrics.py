"""Evaluation metrics for SNCF TGV delay prediction.

This module implements the compute_metrics() contract required by
scripts/main.py. It returns regression metrics for model comparison.
"""

from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return regression metrics comparing true and predicted delays.

    Args:
        y_true: Ground truth delay values (minutes).
        y_pred: Model predicted delay values (minutes).

    Returns:
        Dictionary with MAE, RMSE, and R2 scores.
    """
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }
