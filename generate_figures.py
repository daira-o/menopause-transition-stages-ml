"""
generate_figures.py
-------------------
Genera figuras finales reproducibles para el paper/dashboard STATUS5.

Uso:
    python generate_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from config import CLEAN_FILE, METRICS_DIR, MODEL_ARTIFACTS, N_SPLITS, OUTPUTS_DIR, RANDOM_STATE
from src.models.modeling import CLASS_NAMES, TARGET_COLUMN, load_preprocessed_dataset, make_cv


FIGURES_DIR = OUTPUTS_DIR / "figures"
TOP_N = 10
RF_KEY = "random_forest"
SPANISH_CLASS_NAMES = {
    2: "Post natural",
    3: "Peri tardia",
    4: "Peri temprana",
    5: "Pre",
}


def setup_style() -> None:
    """Aplica una estetica sobria y consistente para figuras tipo paper IEEE."""
    sns.set_theme(
        context="paper",
        style="whitegrid",
        palette="deep",
        font="DejaVu Sans",
        rc={
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    )


def save_figure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


def label_status(values: Iterable[object]) -> list[str]:
    labels = []
    for value in values:
        try:
            key = int(value)
        except (TypeError, ValueError):
            labels.append(str(value))
        else:
            labels.append(f"{key} - {SPANISH_CLASS_NAMES.get(key, CLASS_NAMES.get(key, str(key)))}")
    return labels


def load_feature_importance(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontro feature importance: {path}")
    importances = pd.read_csv(path)
    required = {"variable", "importancia"}
    if not required.issubset(importances.columns):
        raise ValueError(f"El archivo {path} debe contener columnas {sorted(required)}")
    return importances.sort_values("importancia", ascending=False).reset_index(drop=True)


def expected_feature_names(model: object) -> list[str]:
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return list(names)

    classifier = getattr(model, "named_steps", {}).get("classifier")
    names = getattr(classifier, "feature_names_in_", None)
    if names is not None:
        return list(names)

    raise AttributeError(
        "No se pudieron recuperar las variables esperadas del modelo serializado."
    )


def validate_feature_frame(x: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    """Verifica que X coincida exactamente con las variables usadas por el modelo."""
    if TARGET_COLUMN in x.columns:
        raise ValueError(f"{TARGET_COLUMN} no debe estar dentro de X.")

    missing = [col for col in expected if col not in x.columns]
    extra = [col for col in x.columns if col not in expected]
    if missing or extra:
        raise ValueError(
            "Las columnas de X no coinciden con el modelo. "
            f"Faltantes={missing}; extra={extra}"
        )

    return x.loc[:, expected].apply(pd.to_numeric, errors="coerce")


def validation_subset(x: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Reconstruye una particion de validacion desde el mismo Stratified K-Fold."""
    cv = make_cv(n_splits=N_SPLITS, random_state=RANDOM_STATE)
    splits = list(cv.split(x, y))
    _, valid_idx = splits[-1]
    return x.iloc[valid_idx].copy(), y.iloc[valid_idx].copy()


def imputed_frame(model: object, x: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Replica el manejo de NaN del pipeline: imputacion mediana antes de SHAP."""
    imputer = model.named_steps["imputer"]
    x_imputed = imputer.transform(x)
    return pd.DataFrame(x_imputed, columns=feature_names, index=x.index)


def normalize_shap_values(shap_values: object, n_features: int) -> np.ndarray:
    """
    Devuelve SHAP con forma (n_samples, n_features, n_classes).

    TreeExplainer puede devolver una lista por clase o un arreglo 3D segun la
    version de SHAP/scikit-learn. Para importancia global multiclase se usa el
    promedio del valor absoluto entre muestras y clases.
    """
    if isinstance(shap_values, list):
        return np.stack(shap_values, axis=2)

    values = np.asarray(shap_values)
    if values.ndim == 2:
        return values[:, :, np.newaxis]
    if values.ndim != 3:
        raise ValueError(f"Forma SHAP no soportada: {values.shape}")

    if values.shape[1] == n_features:
        return values
    if values.shape[2] == n_features:
        if values.shape[0] < values.shape[1]:
            return np.moveaxis(values, 0, 2)
        return np.moveaxis(values, 2, 1)
    if values.shape[0] == n_features:
        return np.moveaxis(values, 0, 1)

    raise ValueError(f"No se pudo alinear SHAP con {n_features} variables: {values.shape}")


def compute_shap_importance(
    model: object,
    x_valid: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    classifier = model.named_steps["classifier"]
    x_imputed = imputed_frame(model, x_valid, feature_names)

    explainer = shap.TreeExplainer(classifier)
    shap_values = normalize_shap_values(
        explainer.shap_values(x_imputed),
        n_features=len(feature_names),
    )
    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            "La cantidad de variables SHAP no coincide con las columnas del modelo."
        )

    mean_abs = np.abs(shap_values).mean(axis=(0, 2))
    importance = (
        pd.DataFrame({"variable": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    return importance, shap_values, x_imputed


def plot_class_distribution(df: pd.DataFrame) -> Path:
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    plt.figure(figsize=(4.8, 3.2))
    ax = sns.barplot(x=label_status(counts.index), y=counts.values, color="#4C72B0")
    ax.set_title("Distribucion de clases por etapa de transicion menopausica")
    ax.set_xlabel("Clase STATUS5")
    ax.set_ylabel("Numero de registros")
    ax.bar_label(ax.containers[0], fontsize=8, padding=2)
    plt.xticks(rotation=20, ha="right")
    return save_figure(FIGURES_DIR / "class_distribution.png")


def plot_missing_values(df: pd.DataFrame) -> Path:
    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    missing = missing[missing > 0].head(20)
    if missing.empty:
        missing = pd.Series({"Sin valores faltantes": 0.0})

    plt.figure(figsize=(6.2, 3.8))
    ax = sns.barplot(x=missing.values, y=missing.index, color="#55A868")
    ax.set_title("Valores faltantes despues del preprocesamiento")
    ax.set_xlabel("Valores faltantes (%)")
    ax.set_ylabel("Variable")
    ax.set_xlim(0, max(1.0, float(missing.max()) * 1.15))
    return save_figure(FIGURES_DIR / "missing_values.png")


def plot_confusion_matrix(cm_path: Path) -> Path:
    cm = pd.read_csv(cm_path, index_col=0)
    cm.index = [label.replace("real_", "") for label in cm.index]
    cm.columns = [label.replace("pred_", "") for label in cm.columns]

    plt.figure(figsize=(4.6, 3.8))
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=0.5)
    ax.set_title("Matriz de confusion con validacion cruzada")
    ax.set_xlabel("STATUS5 predicho")
    ax.set_ylabel("STATUS5 real")
    ax.set_xticklabels(label_status(cm.columns), rotation=25, ha="right")
    ax.set_yticklabels(label_status(cm.index), rotation=0)
    return save_figure(FIGURES_DIR / "confusion_matrix.png")


def plot_feature_importance(importances: pd.DataFrame) -> Path:
    top = importances.head(TOP_N).iloc[::-1]
    plt.figure(figsize=(5.8, 3.8))
    ax = sns.barplot(data=top, x="importancia", y="variable", color="#C44E52")
    ax.set_title("Importancia de variables Random Forest - Top 10")
    ax.set_xlabel("Disminucion media de impureza")
    ax.set_ylabel("Variable clinica")
    return save_figure(FIGURES_DIR / "feature_importance_top10.png")


def plot_shap_global_bar(shap_top: pd.DataFrame) -> Path:
    top = shap_top.iloc[::-1]
    plt.figure(figsize=(5.8, 3.8))
    ax = sns.barplot(data=top, x="mean_abs_shap", y="variable", color="#8172B3")
    ax.set_title("Importancia global SHAP - Top 10")
    ax.set_xlabel("Valor SHAP absoluto medio entre clases")
    ax.set_ylabel("Variable clinica")
    return save_figure(FIGURES_DIR / "shap_global_bar_top10.png")


def plot_shap_bar_by_class(
    shap_values: np.ndarray,
    x_imputed: pd.DataFrame,
    class_labels: Iterable[int],
) -> list[Path]:
    """Genera un SHAP bar plot limpio para cada clase STATUS5."""
    generated: list[Path] = []
    for class_index, class_label in enumerate(class_labels):
        explanation = shap.Explanation(
            values=shap_values[:, :, class_index],
            data=x_imputed.values,
            feature_names=list(x_imputed.columns),
        )
        plt.figure(figsize=(6.0, 4.2))
        shap.plots.bar(explanation, max_display=TOP_N, show=False)
        ax = plt.gca()
        class_name = SPANISH_CLASS_NAMES.get(int(class_label), CLASS_NAMES.get(int(class_label), str(class_label)))
        ax.set_title(f"SHAP por clase {class_label} - {class_name}")
        ax.set_xlabel("Valor SHAP absoluto medio")
        generated.append(save_figure(FIGURES_DIR / f"shap_bar_class_{class_label}_top10.png"))
    return generated


def plot_shap_beeswarm_by_class(
    shap_values: np.ndarray,
    x_imputed: pd.DataFrame,
    class_labels: Iterable[int],
) -> list[Path]:
    """Genera un SHAP beeswarm independiente para cada clase STATUS5."""
    generated: list[Path] = []
    for class_index, class_label in enumerate(class_labels):
        explanation = shap.Explanation(
            values=shap_values[:, :, class_index],
            data=x_imputed.values,
            feature_names=list(x_imputed.columns),
        )
        plt.figure(figsize=(6.4, 4.8))
        shap.plots.beeswarm(explanation, max_display=TOP_N, show=False)
        ax = plt.gca()
        class_name = SPANISH_CLASS_NAMES.get(
            int(class_label),
            CLASS_NAMES.get(int(class_label), str(class_label)),
        )
        ax.set_title(f"SHAP beeswarm clase {class_label} - {class_name}")
        ax.set_xlabel("Valor SHAP")
        generated.append(save_figure(FIGURES_DIR / f"shap_beeswarm_class_{class_label}_top10.png"))
    return generated


def predicted_class_shap_values(
    model: object,
    shap_values: np.ndarray,
    x_imputed: pd.DataFrame,
) -> np.ndarray:
    """
    Convierte SHAP multiclase a una matriz 2D usando la clase predicha.

    SHAP entrega un tensor muestra x variable x clase. Para obtener el summary
    plot clasico de puntos coloreados, se usa para cada registro la contribucion
    SHAP correspondiente a la clase predicha por el Random Forest.
    """
    classifier = model.named_steps["classifier"]
    predicted_labels = classifier.predict(x_imputed)
    class_to_index = {label: idx for idx, label in enumerate(classifier.classes_)}
    class_indices = np.array([class_to_index[label] for label in predicted_labels])
    row_indices = np.arange(shap_values.shape[0])
    return shap_values[row_indices, :, class_indices]


def plot_shap_summary(
    model: object,
    shap_values: np.ndarray,
    x_imputed: pd.DataFrame,
    shap_top: pd.DataFrame,
) -> Path:
    top_vars = shap_top["variable"].tolist()
    top_indices = [x_imputed.columns.get_loc(variable) for variable in top_vars]
    shap_predicted = predicted_class_shap_values(model, shap_values, x_imputed)

    plt.figure(figsize=(6.4, 4.8))
    shap.summary_plot(
        shap_predicted[:, top_indices],
        x_imputed.loc[:, top_vars],
        feature_names=top_vars,
        max_display=TOP_N,
        show=False,
        plot_size=None,
    )
    ax = plt.gca()
    ax.set_title("SHAP summary plot - Top 10")
    ax.set_xlabel("Valor SHAP para la clase predicha")
    return save_figure(FIGURES_DIR / "shap_summary_top10.png")


def plot_correlation_heatmap(df: pd.DataFrame, top_features: list[str]) -> Path:
    features = [feature for feature in top_features if feature in df.columns]
    corr = df[features].apply(pd.to_numeric, errors="coerce").corr(method="spearman")

    plt.figure(figsize=(5.8, 4.8))
    ax = sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        square=True,
        linewidths=0.3,
        cbar_kws={"label": "Rho de Spearman"},
    )
    ax.set_title("Mapa de calor de correlacion de variables principales")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=45, ha="right")
    return save_figure(FIGURES_DIR / "correlation_heatmap_top_features.png")


def plot_symptom_boxplots(df: pd.DataFrame, variables: list[str]) -> Path:
    """Compara sintomas clinicamente relevantes contra STATUS5 sin solapar grupos."""
    plot_df = df[[TARGET_COLUMN, *variables]].copy()
    plot_df[TARGET_COLUMN] = plot_df[TARGET_COLUMN].map(lambda value: label_status([value])[0])
    status_order = [
        "5 - Pre",
        "4 - Peri temprana",
        "3 - Peri tardia",
        "2 - Post natural",
    ]
    status_tick_labels = ["Pre", "Peri\ntemprana", "Peri\ntardia", "Post\nnatural"]

    fig, axes = plt.subplots(1, len(variables), figsize=(11.5, 3.4), sharey=True)
    if len(variables) == 1:
        axes = [axes]

    for ax, variable in zip(axes, variables):
        panel = plot_df[[TARGET_COLUMN, variable]].copy()
        panel[variable] = pd.to_numeric(panel[variable], errors="coerce")
        sns.boxplot(
            data=panel,
            x=TARGET_COLUMN,
            y=variable,
            order=status_order,
            color="#4C72B0",
            fliersize=1.5,
            linewidth=0.8,
            ax=ax,
        )
        ax.set_title(f"{variable} vs STATUS5")
        ax.set_xlabel("Clase STATUS5")
        ax.set_ylabel("Puntaje de sintoma" if ax is axes[0] else "")
        ax.set_xticks(range(len(status_tick_labels)), labels=status_tick_labels, rotation=0)

    fig.suptitle("Puntajes de sintomas seleccionados por clase STATUS5", y=1.03, fontsize=11)
    return save_figure(FIGURES_DIR / "symptom_boxplots_by_status.png")


def main() -> None:
    setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    cfg = MODEL_ARTIFACTS[RF_KEY]
    df = pd.read_csv(CLEAN_FILE)
    x, y = load_preprocessed_dataset(CLEAN_FILE)
    model = joblib.load(cfg["model"])
    feature_names = expected_feature_names(model)
    x = validate_feature_frame(x, feature_names)
    x_valid, _ = validation_subset(x, y)

    importances = load_feature_importance(cfg["feature_importance"])
    shap_importance, shap_values, x_imputed = compute_shap_importance(
        model,
        x_valid,
        feature_names,
    )
    shap_top = shap_importance.head(TOP_N).copy()
    shap_csv = METRICS_DIR / "shap_importance_top10.csv"
    shap_top.to_csv(shap_csv, index=False, encoding="utf-8-sig")

    shap_class_bar_paths = plot_shap_bar_by_class(
        shap_values,
        x_imputed,
        class_labels=model.named_steps["classifier"].classes_,
    )
    shap_class_beeswarm_paths = plot_shap_beeswarm_by_class(
        shap_values,
        x_imputed,
        class_labels=model.named_steps["classifier"].classes_,
    )

    generated = [
        plot_class_distribution(df),
        plot_missing_values(df),
        plot_confusion_matrix(cfg["confusion_matrix"]),
        plot_feature_importance(importances),
        plot_shap_global_bar(shap_top),
        *shap_class_bar_paths,
        *shap_class_beeswarm_paths,
        plot_shap_summary(model, shap_values, x_imputed, shap_top),
        plot_correlation_heatmap(df, importances["variable"].head(TOP_N).tolist()),
        plot_symptom_boxplots(df, ["HOTFLAS5", "NITESWE5", "DESIRSE5", "VAGINDR5"]),
        shap_csv,
    ]

    print("\nArchivos generados:")
    for path in generated:
        print(f"- {path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nError generando figuras finales: {exc}", file=sys.stderr)
        sys.exit(1)
