"""Helpers for saving evaluation results."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Optional

import pandas as pd


def write_metrics(
    rows: Iterable[dict[str, object]],
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Write model metrics to a CSV file and return the DataFrame.

    Parameters
    ----------
    rows:
        Iterable of metric dicts, one per model.
    output_path:
        Destination CSV path. When *None* the default from ``config`` is used.
        Pass an explicit path in tests to avoid touching the real filesystem.
    """
    if output_path is None:
        # Lazy import keeps this module decoupled from config at import time.
        from config import MODEL_METRICS_FILE  # noqa: PLC0415

        output_path = MODEL_METRICS_FILE

    metrics_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_path, index=False)
    return metrics_df
