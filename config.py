"""
config.py
---------
Configuración centralizada del proyecto.
Modificar SOLO este archivo para ajustar rutas, umbrales y parámetros del pipeline.
"""

from pathlib import Path

# ── Raíz del proyecto (siempre relativa a este archivo) ──────────────────────
BASE_DIR = Path(__file__).parent

# ── Rutas de datos ────────────────────────────────────────────────────────────
DATA_RAW_DIR       = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATA_DICTIONARY_FILE = BASE_DIR / "data" / "clasificacion_variables_ml - clasificacion_variables_ml.csv"

# Archivo de entrada (coloca datos.tsv en data/raw/)
INPUT_FILE = DATA_RAW_DIR / "datos.tsv"

# Archivo de salida con el dataset limpio
CLEAN_FILE = DATA_PROCESSED_DIR / "datos_limpios.csv"

# ── Rutas de salidas ──────────────────────────────────────────────────────────
OUTPUTS_DIR = BASE_DIR / "outputs"
INFO_DIR    = OUTPUTS_DIR / "info"


LOG_FILE    = INFO_DIR    / "variables_eliminadas.csv"

# Artefactos del pipeline de clasificación
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

# Optimización Optuna
RANDOM_STATE: int = 42
N_SPLITS: int = 5
OPTUNA_N_TRIALS: int = 5

# ── Parámetros del pipeline ───────────────────────────────────────────────────
# Paso 2: columnas con más del X% de nulos serán eliminadas
UMBRAL_NULOS: float = 0.50        # 50 %

# Paso 3: columnas donde el valor más frecuente supera el X% serán eliminadas
UMBRAL_DOMINANTE: float = 0.95    # 95 %
