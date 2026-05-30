"""
train_model.py
--------------
Entrenamiento Random Forest para STATUS5.

Uso:
    python train_model.py
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

from config import (
    BEST_PARAMS_FILE,
    CLASSIFICATION_REPORT_FILE,
    CLEAN_FILE,
    CONFUSION_MATRIX_FILE,
    MODEL_ARTIFACTS,
    MODEL_FILE,
    NODE_SPLIT_LOGIC_FILE,
    N_SPLITS,
    OPTUNA_N_TRIALS,
    RANDOM_STATE,
)
from src.models.modeling import (
    TrainingArtifacts,
    TrainingSummary,
    load_preprocessed_dataset,
    make_cv,
)
from src.models.random_forest import train_random_forest


def build_artifacts() -> TrainingArtifacts:
    cfg = MODEL_ARTIFACTS["random_forest"]
    return TrainingArtifacts(
        model_path=cfg["model"],
        confusion_matrix_path=cfg["confusion_matrix"],
        classification_report_path=cfg["classification_report"],
        best_params_path=cfg["best_params"],
        metrics_path=cfg["metrics"],
        feature_importance_path=cfg["feature_importance"],
        node_split_logic_path=cfg["node_split_logic"],
    )


def sync_legacy_random_forest_outputs(summary: TrainingSummary) -> None:
    """Mantiene compatibilidad con rutas historicas del pipeline."""
    copies: list[tuple[Path | None, Path | None]] = [
        (summary.artifacts.model_path, MODEL_FILE),
        (summary.artifacts.confusion_matrix_path, CONFUSION_MATRIX_FILE),
        (summary.artifacts.classification_report_path, CLASSIFICATION_REPORT_FILE),
        (summary.artifacts.best_params_path, BEST_PARAMS_FILE),
        (summary.artifacts.node_split_logic_path, NODE_SPLIT_LOGIC_FILE),
    ]
    for source, target in copies:
        if source is not None and target is not None and source.exists() and source != target:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def print_summary(summary: TrainingSummary, elapsed: float) -> None:
    display_name = MODEL_ARTIFACTS["random_forest"]["display_name"]
    print("\n============================================================")
    print(f"  ENTRENAMIENTO STATUS5 COMPLETADO: {display_name}")
    print("============================================================")
    print(f"  Distribucion de clases     : {summary.class_counts}")
    print(f"  Mejor F1 macro Optuna      : {summary.best_cv_f1_macro:.4f}")
    print(f"  F1 macro final CV          : {summary.final_cv_f1_macro:.4f}")
    print(f"  Accuracy final CV          : {summary.metrics['accuracy']:.4f}")
    print(f"  F1 weighted final CV       : {summary.metrics['f1_weighted']:.4f}")
    print(f"  Mejores hiperparametros    : {summary.best_params}")
    print(f"  Modelo guardado            : {summary.artifacts.model_path}")
    print(f"  Metricas JSON              : {summary.artifacts.metrics_path}")
    print(f"  Matriz de confusion        : {summary.artifacts.confusion_matrix_path}")
    print(f"  Reporte por clase          : {summary.artifacts.classification_report_path}")
    if summary.artifacts.feature_importance_path is not None:
        print(f"  Feature importance         : {summary.artifacts.feature_importance_path}")
    if summary.artifacts.node_split_logic_path is not None:
        print(f"  Logica de nodos            : {summary.artifacts.node_split_logic_path}")
    print(f"  Tiempo modelo              : {elapsed:.1f}s")
    print("============================================================")


def main() -> None:
    start = time.time()

    x, y = load_preprocessed_dataset(CLEAN_FILE)
    cv = make_cv(n_splits=N_SPLITS, random_state=RANDOM_STATE)

    model_start = time.time()
    summary = train_random_forest(
        x=x,
        y=y,
        cv=cv,
        artifacts=build_artifacts(),
        n_trials=OPTUNA_N_TRIALS,
        random_state=RANDOM_STATE,
    )
    sync_legacy_random_forest_outputs(summary)
    print_summary(summary, elapsed=time.time() - model_start)

    print(f"\nTiempo total: {time.time() - start:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n  Error fatal durante entrenamiento: {exc}", file=sys.stderr)
        sys.exit(1)
