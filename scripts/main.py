from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def _load_module(module_name: str, module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module `{module_name}` from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Bootstrap: resolve paths and register "config" in sys.modules so that
# src/ modules can do `from config import ...` regardless of cwd.
# All of this is intentionally inside a guard so that importing this file
# as a module (e.g. in tests) does NOT trigger filesystem side effects.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = SCRIPT_DIR.parent / "src"


def _bootstrap() -> tuple[Any, Any, Any, Any, Any]:
    """Load config + src modules and return (config, load_dataset_split,
    compute_metrics, load_model, write_metrics).

    Kept in a function so that nothing runs at import time.
    """
    config = _load_module("project_config", _SRC_DIR / "config.py")
    # Register under the bare name so `from config import X` works in src/.
    sys.modules["config"] = config
    load_dotenv(config.ENV_FILE)

    data_module = _load_module("project_data", _SRC_DIR / "data.py")
    metrics_module = _load_module("project_metrics", _SRC_DIR / "metrics.py")
    model_io_module = _load_module("project_model_io", _SRC_DIR / "model_io.py")
    results_module = _load_module("project_results", _SRC_DIR / "results.py")

    return (
        config,
        data_module.load_dataset_split,
        metrics_module.compute_metrics,
        model_io_module.load_model,
        results_module.write_metrics,
    )


def _validate_models_config(models: dict) -> None:
    if not models:
        raise ValueError("config.MODELS is empty. Add your trained models first.")

    for model_key, model_config in models.items():
        if "path" not in model_config:
            raise ValueError(
                f"Missing `path` for model `{model_key}` in config.MODELS."
            )


def _validate_app_entrypoint(app_entrypoint: Path) -> None:
    """Validate that app.py exposes a callable build_app.

    This is checked just before launching Streamlit — not before evaluation —
    so that a broken app entry point never blocks model metrics computation.
    """
    app_module = _load_module("project_app", app_entrypoint)
    if not hasattr(app_module, "build_app") or not callable(app_module.build_app):
        raise TypeError("app.build_app must be a callable Streamlit entry point.")


def _streamlit_env(src_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_entries = [str(src_dir)]
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)

    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return env


def _load_dataset(load_dataset_split) -> tuple[Any, Any, Any, Any]:
    dataset_split = load_dataset_split()
    if not isinstance(dataset_split, tuple) or len(dataset_split) != 4:
        raise ValueError(
            "data.load_dataset_split() must return exactly four values: "
            "(X_train, X_test, y_train, y_test)."
        )

    return dataset_split


def _evaluate_models(
    models: dict,
    load_model,
    compute_metrics,
    X_test: Any,
    y_test: Any,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for model_key, model_config in models.items():
        model = load_model(Path(model_config["path"]))

        if not hasattr(model, "predict"):
            raise TypeError(
                f"Loaded object for model `{model_key}` does not expose a `predict` method."
            )

        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)

        if not isinstance(metrics, dict) or not metrics:
            raise ValueError(
                "metrics.compute_metrics() must return a non-empty dictionary."
            )

        row: dict[str, object] = {
            "model_key": model_key,
            "model_name": model_config.get("name", model_key),
            "model_path": str(model_config["path"]),
        }

        for metric_name, metric_value in metrics.items():
            row[metric_name] = float(metric_value)

        rows.append(row)

    return rows


def _launch_streamlit(app_entrypoint: Path, project_root: Path, src_dir: Path,
                      host: str, port: int) -> None:
    if not app_entrypoint.exists():
        raise FileNotFoundError(f"Streamlit entry point not found: {app_entrypoint}")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_entrypoint),
            "--server.address",
            host,
            "--server.port",
            str(port),
        ],
        check=True,
        cwd=project_root,
        env=_streamlit_env(src_dir),
    )


def main() -> None:
    config, load_dataset_split, compute_metrics, load_model, write_metrics = _bootstrap()

    # Ensure all project directories exist before any I/O.
    config.ensure_dirs()

    _validate_models_config(config.MODELS)

    try:
        _, X_test, _, y_test = _load_dataset(load_dataset_split)
    except NotImplementedError as exc:
        raise NotImplementedError(
            "Dataset loading is still a template placeholder. "
            "Implement data.load_dataset_split()."
        ) from exc

    try:
        metrics_rows = _evaluate_models(
            config.MODELS, load_model, compute_metrics, X_test, y_test
        )
    except NotImplementedError as exc:
        raise NotImplementedError(
            "Metric computation is still a template placeholder. "
            "Implement metrics.compute_metrics()."
        ) from exc

    metrics_df = write_metrics(metrics_rows)

    print("Model evaluation completed. Metrics saved to results/model_metrics.csv")
    print(metrics_df.to_string(index=False))
    print(f"\nLaunching Streamlit on http://{config.STREAMLIT_HOST}:{config.STREAMLIT_PORT} ...")

    # Validate app entry point only just before launching — a broken Streamlit
    # file must not prevent model evaluation results from being written.
    _validate_app_entrypoint(config.APP_ENTRYPOINT)
    _launch_streamlit(
        config.APP_ENTRYPOINT,
        config.PROJECT_ROOT,
        config.SRC_DIR,
        config.STREAMLIT_HOST,
        config.STREAMLIT_PORT,
    )


if __name__ == "__main__":
    main()
