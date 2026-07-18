"""
config.py
---------
Centralized project configuration.
Edit this file to adjust paths, thresholds, and pipeline parameters.
"""

from pathlib import Path

# Project root, always relative to this file.
BASE_DIR = Path(__file__).parent

# Data paths.
DATA_RAW_DIR       = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATA_DICTIONARY_FILE = BASE_DIR / "data" / "clasificacion_variables_ml - clasificacion_variables_ml.csv"

# Input file. Place datos.tsv in data/raw/.
INPUT_FILE = DATA_RAW_DIR / "datos.tsv"

# Output file for the cleaned dataset.
CLEAN_FILE = DATA_PROCESSED_DIR / "datos_limpios.csv"

# Output paths.
OUTPUTS_DIR = BASE_DIR / "outputs"
INFO_DIR    = OUTPUTS_DIR / "info"


LOG_FILE    = INFO_DIR    / "variables_eliminadas.csv"

# Classification pipeline artifacts.
MODELS_DIR = OUTPUTS_DIR / "models"
MODEL_FILE = MODELS_DIR / "random_forest_status5.joblib"
METRICS_DIR = OUTPUTS_DIR / "metrics"
CONFUSION_MATRIX_FILE = METRICS_DIR / "confusion_matrix_status5.csv"
CLASSIFICATION_REPORT_FILE = METRICS_DIR / "classification_report_status5.csv"
BEST_PARAMS_FILE = METRICS_DIR / "best_params_status5.json"
NODE_SPLIT_LOGIC_FILE = METRICS_DIR / "node_split_logic_status5.txt"
MODEL_ARTIFACTS = {
    "random_forest": {
        "display_name": "Random Forest",
        "model": MODELS_DIR / "random_forest_status5.joblib",
        "confusion_matrix": METRICS_DIR / "random_forest_confusion_matrix_status5.csv",
        "classification_report": METRICS_DIR / "random_forest_classification_report_status5.csv",
        "best_params": METRICS_DIR / "random_forest_best_params_status5.json",
        "metrics": METRICS_DIR / "random_forest_metrics_status5.json",
        "feature_importance": METRICS_DIR / "random_forest_feature_importance_status5.csv",
        "node_split_logic": METRICS_DIR / "random_forest_node_split_logic_status5.txt",
    },
}

# Optuna optimization.
RANDOM_STATE: int = 42
N_SPLITS: int = 5
OPTUNA_N_TRIALS: int = 5

# Pipeline parameters.
# Step 2: columns with more than X% missing values are removed.
UMBRAL_NULOS: float = 0.50        # 50 %

# Step 3: columns whose most frequent value exceeds X% are removed.
UMBRAL_DOMINANTE: float = 0.95    # 95 %
