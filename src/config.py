from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PLOTS_DIR = PROJECT_ROOT / "plots"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"

_MANAGED_DIRS = [
    DATA_DIR,
    LOGS_DIR,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    PLOTS_DIR,
    RESULTS_DIR,
    SCRIPTS_DIR,
    TESTS_DIR,
]


def ensure_dirs() -> None:
    """Create all project directories that must exist at runtime.

    Called explicitly by entry points (scripts/main.py, app.py) rather than
    as a module-level side effect, so that importing ``config`` in tests or
    notebooks never touches the filesystem unexpectedly.
    """
    for d in _MANAGED_DIRS:
        d.mkdir(parents=True, exist_ok=True)

ENV_FILE = PROJECT_ROOT / ".env"
APP_ENTRYPOINT = PROJECT_ROOT / "src" / "app.py"
MODEL_METRICS_FILE = RESULTS_DIR / "model_metrics.csv"

STREAMLIT_HOST = "localhost"
STREAMLIT_PORT = 8501

# Trained models registry
MODELS = {
    "linear_regression": {
        "name": "Linear Regression",
        "description": "Baseline regression model with no regularization.",
        "path": MODELS_DIR / "linear_regression.joblib",
    },
    "random_forest": {
        "name": "Random Forest",
        "description": "Ensemble of 100 decision trees with bagging.",
        "path": MODELS_DIR / "random_forest.joblib",
    },
    "gradient_boosting": {
        "name": "Gradient Boosting",
        "description": "Sequential boosting with 200 estimators and max_depth=5.",
        "path": MODELS_DIR / "gradient_boosting.joblib",
    },
    "knn": {
        "name": "K-Nearest Neighbors",
        "description": "KNN regressor (k=10) with standard scaling pipeline.",
        "path": MODELS_DIR / "knn.joblib",
    },
}
