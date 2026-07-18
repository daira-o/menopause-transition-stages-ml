"""
src/models/modeling.py
----------------------
Shared utilities for training and evaluating STATUS5 models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold


TARGET_COLUMN = "STATUS5"
CLASS_NAMES = {
    2: "Natural Post",
    3: "Late Peri",
    4: "Early Peri",
    5: "Pre",
}
@dataclass(frozen=True)
class TrainingArtifacts:
    """Paths for artifacts generated during training."""

    model_path: Path
    confusion_matrix_path: Path
    classification_report_path: Path
    best_params_path: Path
    metrics_path: Path
    feature_importance_path: Path | None = None
    node_split_logic_path: Path | None = None


@dataclass(frozen=True)
class TrainingSummary:
    """Minimal summary for console output or notebooks."""

    best_cv_f1_macro: float
    final_cv_f1_macro: float
    metrics: dict[str, float]
    best_params: dict[str, Any]
    class_counts: dict[int, int]
    artifacts: TrainingArtifacts


def load_preprocessed_dataset(
    path: Path,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load datos_limpios.csv and separate predictors from the target variable."""
    if not path.exists():
        raise FileNotFoundError(f"Preprocessed dataset not found: {path}")

    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise KeyError(f"Target variable '{target_column}' was not found.")

    y = pd.to_numeric(df[target_column], errors="raise").astype(int)
    x = df.drop(columns=[target_column])
    return x, y


def make_cv(n_splits: int, random_state: int) -> StratifiedKFold:
    """Create Stratified K-Fold to preserve the STATUS5 class proportion per fold."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def save_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def _serializable_params(params: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(params, default=str))


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute metrics that can be compared across models."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def save_evaluation_outputs(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: list[int],
    best_params: dict[str, Any],
    metrics: dict[str, float],
    artifacts: TrainingArtifacts,
) -> None:
    """Save confusion matrix, per-class report, metrics, and parameters."""
    artifacts.confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"real_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )
    cm_df.to_csv(artifacts.confusion_matrix_path, encoding="utf-8-sig")

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[CLASS_NAMES.get(label, str(label)) for label in labels],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        artifacts.classification_report_path,
        encoding="utf-8-sig",
    )

    artifacts.best_params_path.write_text(
        json.dumps(_serializable_params(best_params), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    artifacts.metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_feature_importance(
    feature_names: list[str],
    importances: list[float],
    path: Path | None,
) -> None:
    """Save feature importance if the model exposes it."""
    if path is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    (
        pd.DataFrame({"variable": feature_names, "importancia": importances})
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
        .to_csv(path, index=False, encoding="utf-8-sig")
    )
