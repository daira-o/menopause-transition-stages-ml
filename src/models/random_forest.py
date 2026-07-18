"""
src/models/random_forest.py
---------------------------
Random Forest baseline training for STATUS5.
"""

from __future__ import annotations

from typing import Any

import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline

from src.models.modeling import (
    TrainingArtifacts,
    TrainingSummary,
    compute_metrics,
    save_evaluation_outputs,
    save_feature_importance,
    save_model,
)


def protected_min_samples_leaf_upper_bound(y: pd.Series, n_splits: int) -> int:
    """
    Define a conservative upper bound for min_samples_leaf from the minority class.
    """
    min_class_count = int(y.value_counts().min())
    min_train_count = int(min_class_count * (n_splits - 1) / n_splits)
    return max(2, min(20, min_train_count // 8))


def build_random_forest_model(params: dict[str, Any], random_state: int) -> Pipeline:
    """Build the median-imputation + balanced Random Forest pipeline."""
    classifier = RandomForestClassifier(
        **params,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ]
    )


def optimize_random_forest(
    x: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    n_trials: int,
    random_state: int,
) -> optuna.study.Study:
    """Optimize hyperparameters with macro F1 as the objective metric."""
    max_leaf = protected_min_samples_leaf_upper_bound(y, cv.n_splits)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, max_leaf),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        }
        pipeline = build_random_forest_model(params=params, random_state=random_state)
        scores = cross_val_score(
            pipeline,
            x.apply(pd.to_numeric, errors="coerce"),
            y,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            error_score="raise",
        )
        return float(scores.mean())

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def node_split_logic_text(best_params: dict[str, Any]) -> str:
    """Summarize how the Random Forest decides node splits."""
    criterion = best_params["criterion"]
    max_depth = best_params["max_depth"]
    min_samples_leaf = best_params["min_samples_leaf"]

    return (
        "Random Forest node-splitting logic\n"
        "==================================\n\n"
        "Each tree evaluates binary splits of the form variable <= threshold. "
        "For each node, the algorithm selects the split that most reduces "
        f"impurity according to the optimized criterion: {criterion}.\n\n"
        "With criterion='gini', the model minimizes the probability of "
        "misclassifying a record if it is labeled according to the node "
        "distribution. With criterion='entropy', the model maximizes "
        "information gain.\n\n"
        f"Depth is limited to max_depth={max_depth} to reduce overfitting and "
        "favor biologically generalizable patterns. "
        f"In addition, min_samples_leaf={min_samples_leaf} prevents leaves from "
        "becoming too small, protecting stability for minority stages.\n\n"
        "class_weight='balanced' weights each class inversely to its frequency "
        "without generating synthetic observations."
    )


def train_random_forest(
    x: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    artifacts: TrainingArtifacts,
    n_trials: int,
    random_state: int,
) -> TrainingSummary:
    """Run tuning, final cross-validation, and baseline fitting."""
    x_numeric = x.apply(pd.to_numeric, errors="coerce")
    study = optimize_random_forest(
        x=x_numeric,
        y=y,
        cv=cv,
        n_trials=n_trials,
        random_state=random_state,
    )
    best_params = dict(study.best_params)
    final_pipeline = build_random_forest_model(params=best_params, random_state=random_state)

    y_pred = cross_val_predict(final_pipeline, x_numeric, y, cv=cv, n_jobs=-1)
    labels = sorted(y.unique().tolist())
    metrics = compute_metrics(y, pd.Series(y_pred, index=y.index))
    save_evaluation_outputs(
        y_true=y,
        y_pred=pd.Series(y_pred, index=y.index),
        labels=labels,
        best_params=best_params,
        metrics=metrics,
        artifacts=artifacts,
    )

    final_pipeline.fit(x_numeric, y)
    save_model(final_pipeline, artifacts.model_path)

    classifier = final_pipeline.named_steps["classifier"]
    save_feature_importance(
        feature_names=list(x_numeric.columns),
        importances=classifier.feature_importances_.tolist(),
        path=artifacts.feature_importance_path,
    )

    if artifacts.node_split_logic_path is not None:
        artifacts.node_split_logic_path.write_text(
            node_split_logic_text(best_params),
            encoding="utf-8",
        )

    return TrainingSummary(
        best_cv_f1_macro=float(study.best_value),
        final_cv_f1_macro=float(f1_score(y, y_pred, average="macro", zero_division=0)),
        metrics=metrics,
        best_params=best_params,
        class_counts={int(k): int(v) for k, v in y.value_counts().sort_index().items()},
        artifacts=artifacts,
    )
