"""Evaluation metrics for SNCF TGV delay prediction.

This module implements the compute_metrics() contract required by
scripts/main.py. It returns regression metrics for model comparison.

Compatibility note
------------------
scikit-learn >= 1.4 removed the ``squared`` parameter from
``mean_squared_error``. RMSE is now a first-class function:
``sklearn.metrics.root_mean_squared_error``. This module uses
a version-safe import so it works on both old and new sklearn releases.
"""

from __future__ import annotations

from typing import Any

import sklearn
from sklearn.metrics import mean_absolute_error, r2_score

# sklearn 1.4+ ships root_mean_squared_error as a top-level function.
# Earlier versions require the squared=False workaround.
_sklearn_version = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
if _sklearn_version >= (1, 4):
    from sklearn.metrics import root_mean_squared_error as _rmse_fn

    def _rmse(y_true: Any, y_pred: Any) -> float:
        return float(_rmse_fn(y_true, y_pred))

else:
    from sklearn.metrics import mean_squared_error as _mse_fn  # type: ignore[assignment]

    def _rmse(y_true: Any, y_pred: Any) -> float:  # type: ignore[misc]
        return float(_mse_fn(y_true, y_pred, squared=False))


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": _rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }
