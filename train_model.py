"""
train_model.py
--------------
Random Forest training for STATUS5.

Usage:
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
    """Keep compatibility with historical pipeline paths."""
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
    print(f"  STATUS5 TRAINING COMPLETE: {display_name}")
    print("============================================================")
    print(f"  Class distribution         : {summary.class_counts}")
    print(f"  Best Optuna macro F1       : {summary.best_cv_f1_macro:.4f}")
    print(f"  Final CV macro F1          : {summary.final_cv_f1_macro:.4f}")
    print(f"  Final CV accuracy          : {summary.metrics['accuracy']:.4f}")
    print(f"  Final CV weighted F1       : {summary.metrics['f1_weighted']:.4f}")
    print(f"  Best hyperparameters       : {summary.best_params}")
    print(f"  Saved model                : {summary.artifacts.model_path}")
    print(f"  Metrics JSON               : {summary.artifacts.metrics_path}")
    print(f"  Confusion matrix           : {summary.artifacts.confusion_matrix_path}")
    print(f"  Per-class report           : {summary.artifacts.classification_report_path}")
    if summary.artifacts.feature_importance_path is not None:
        print(f"  Feature importance         : {summary.artifacts.feature_importance_path}")
    if summary.artifacts.node_split_logic_path is not None:
        print(f"  Node logic                 : {summary.artifacts.node_split_logic_path}")
    print(f"  Model time                 : {elapsed:.1f}s")
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

    print(f"\nTotal time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n  Fatal training error: {exc}", file=sys.stderr)
        sys.exit(1)
