"""
src/evaluation/visualizer.py
----------------------------
Componentes Streamlit para analizar resultados STATUS5.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
SHAP_IMPORTANCE_FILE = PROJECT_ROOT / "outputs" / "metrics" / "shap_importance_top10.csv"

CLASS_LABELS = {
    "2": "Natural Post",
    "3": "Late Peri",
    "4": "Early Peri",
    "5": "Pre",
    "Natural Post": "Natural Post",
    "Late Peri": "Late Peri",
    "Early Peri": "Early Peri",
    "Pre": "Pre",
}


def _assert_exists(path: Path, label: str) -> bool:
    if not path.exists():
        st.warning(f"No se encontro {label}: {path}")
        return False
    return True


@st.cache_data(show_spinner=False)
def load_csv(path: Path, index_col: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, index_col=index_col)


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_optional_text(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return path.read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def load_image_bytes(path: Path) -> bytes | None:
    if not path.exists():
        return None
    return path.read_bytes()


@st.cache_resource(show_spinner=False)
def load_model(path: Path) -> Any:
    return joblib.load(path)


def class_rows(report: pd.DataFrame) -> pd.DataFrame:
    excluded = {"accuracy", "macro avg", "weighted avg"}
    rows = report.loc[[idx for idx in report.index if str(idx) not in excluded]].copy()
    rows.insert(0, "clase", [CLASS_LABELS.get(str(idx), str(idx)) for idx in rows.index])
    return rows


def report_metrics(report: pd.DataFrame) -> dict[str, float]:
    return {
        "accuracy": float(report.loc["accuracy", "f1-score"]),
        "f1_macro": float(report.loc["macro avg", "f1-score"]),
        "f1_weighted": float(report.loc["weighted avg", "f1-score"]),
    }


def existing_model_keys(model_artifacts: dict[str, dict[str, Any]]) -> list[str]:
    keys = []
    for key, cfg in model_artifacts.items():
        required = [
            cfg["confusion_matrix"],
            cfg["classification_report"],
            cfg["best_params"],
            cfg["metrics"],
            cfg["model"],
        ]
        if all(Path(path).exists() for path in required):
            keys.append(key)
    return keys


def load_model_result(model_key: str, cfg: dict[str, Any]) -> dict[str, Any]:
    result = {
        "key": model_key,
        "display_name": cfg["display_name"],
        "cm": load_csv(cfg["confusion_matrix"], index_col=0),
        "report": load_csv(cfg["classification_report"], index_col=0),
        "params": load_json(cfg["best_params"]),
        "metrics": load_json(cfg["metrics"]),
        "model": load_model(cfg["model"]),
        "paths": cfg,
        "node_logic": load_optional_text(cfg.get("node_split_logic")),
        "feature_importance": None,
    }
    feature_path = cfg.get("feature_importance")
    if feature_path is not None and Path(feature_path).exists():
        result["feature_importance"] = load_csv(feature_path)
    return result


def render_sidebar(model_artifacts: dict[str, dict[str, Any]]) -> None:
    st.sidebar.header("Artefactos")
    st.sidebar.caption("Resultados usados por esta app")
    lines = []
    for cfg in model_artifacts.values():
        lines.extend(
            [
                str(cfg["model"]),
                str(cfg["classification_report"]),
                str(cfg["confusion_matrix"]),
                str(cfg["metrics"]),
            ]
        )
    st.sidebar.code("\n".join(lines), language="text")


def render_metric_cards(result: dict[str, Any]) -> None:
    metrics = result["metrics"] or report_metrics(result["report"])
    total_support = int(result["report"].loc["weighted avg", "support"])
    cols = st.columns(4)
    cols[0].metric("F1 macro", f"{metrics['f1_macro']:.3f}")
    cols[1].metric("F1 weighted", f"{metrics['f1_weighted']:.3f}")
    cols[2].metric("Accuracy CV", f"{metrics['accuracy']:.3f}")
    cols[3].metric("Pacientes evaluados", f"{total_support}")


def render_final_figure(filename: str, caption: str) -> None:
    path = FIGURES_DIR / filename
    image = load_image_bytes(path)
    if image is None:
        st.info(f"Figura pendiente. Ejecuta python generate_figures.py para crear {filename}.")
        return
    st.image(image, caption=caption, use_column_width=True)


def render_shap_interpretability() -> None:
    st.subheader("SHAP Interpretability")
    st.write(
        "SHAP resume cuanto contribuye cada variable clinica al cambio de la "
        "prediccion del Random Forest. En este problema multiclase se promedia "
        "el valor absoluto de SHAP entre clases, por lo que valores mayores "
        "indican mayor influencia global sobre la clasificacion STATUS5."
    )
    st.caption(
        "Las figuras se limitan a las 10 variables principales para mantener "
        "interpretacion global legible y apta para paper, sin analisis individual por paciente."
    )

    left, right = st.columns(2)
    with left:
        render_final_figure(
            "shap_global_bar_top10.png",
            "Importancia SHAP global promedio entre clases - top 10.",
        )
    with right:
        render_final_figure(
            "shap_summary_top10.png",
            "Distribucion por registro del |SHAP| promedio multiclase - top 10.",
        )

    if SHAP_IMPORTANCE_FILE.exists():
        st.subheader("Ranking SHAP top 10")
        shap_importance = load_csv(SHAP_IMPORTANCE_FILE)
        st.dataframe(
            shap_importance.style.format({"mean_abs_shap": "{:.5f}"}),
            use_container_width=True,
        )


def render_standard_figures() -> None:
    st.subheader("Figuras finales")
    st.caption("Figuras PNG estandarizadas en outputs/figures para dashboard y manuscrito.")

    first, second = st.columns(2)
    with first:
        render_final_figure("class_distribution.png", "Distribucion de clases STATUS5.")
        render_final_figure("feature_importance_top10.png", "Feature importance Random Forest - top 10.")
    with second:
        render_final_figure("confusion_matrix.png", "Matriz de confusion validada cruzadamente.")
        render_final_figure("missing_values.png", "Porcentaje de valores faltantes tras preprocessing.")

    render_final_figure(
        "correlation_heatmap_top_features.png",
        "Correlacion Spearman entre variables principales.",
    )
    render_final_figure(
        "symptom_boxplots_by_status.png",
        "Distribucion de sintomas seleccionados por clase STATUS5.",
    )


def render_single_model(result: dict[str, Any]) -> None:
    st.subheader(result["display_name"])
    render_metric_cards(result)

    tab_metrics, tab_model, tab_shap, tab_figures, tab_files = st.tabs(
        ["Metricas", "Modelo", "SHAP Interpretability", "Figuras finales", "Artefactos"]
    )

    with tab_metrics:
        left, right = st.columns([1.05, 0.95])

        with left:
            st.subheader("Matriz de confusion")
            st.dataframe(
                result["cm"].style.background_gradient(cmap="Blues", axis=None),
                use_container_width=True,
            )

        with right:
            st.subheader("F1-score por clase")
            per_class = class_rows(result["report"])
            st.bar_chart(per_class.set_index("clase")[["f1-score"]], height=320)

        st.subheader("Reporte completo")
        visible_cols = ["precision", "recall", "f1-score", "support"]
        st.dataframe(
            result["report"][visible_cols].style.format(
                {
                    "precision": "{:.3f}",
                    "recall": "{:.3f}",
                    "f1-score": "{:.3f}",
                    "support": "{:.0f}",
                }
            ),
            use_container_width=True,
        )

    with tab_model:
        left, right = st.columns([0.8, 1.2])

        with left:
            st.subheader("Mejores hiperparametros")
            st.json(result["params"])

            st.subheader("Pipeline")
            st.code(str(result["model"]), language="text")

        with right:
            st.subheader("Importancia de variables")
            importances = result["feature_importance"]
            if importances is None:
                st.info("No hay feature_importance guardado para este modelo.")
            else:
                top_n = st.slider(
                    "Cantidad de variables",
                    5,
                    min(30, len(importances)),
                    min(15, len(importances)),
                    key=f"top_n_{result['key']}",
                )
                top_importances = importances.head(top_n).set_index("variable")
                st.bar_chart(top_importances, y="importancia", height=420)
                st.dataframe(importances, use_container_width=True)

    with tab_shap:
        render_shap_interpretability()

    with tab_figures:
        render_standard_figures()

    with tab_files:
        node_logic = result["node_logic"]
        if node_logic:
            st.subheader("Logica de division de nodos")
            st.code(node_logic, language="text")

        st.subheader("Archivos")
        paths = result["paths"]
        st.download_button(
            "Descargar classification_report",
            Path(paths["classification_report"]).read_bytes(),
            file_name=Path(paths["classification_report"]).name,
            mime="text/csv",
            key=f"report_{result['key']}",
        )
        st.download_button(
            "Descargar confusion_matrix",
            Path(paths["confusion_matrix"]).read_bytes(),
            file_name=Path(paths["confusion_matrix"]).name,
            mime="text/csv",
            key=f"cm_{result['key']}",
        )


def render_streamlit_app(model_artifacts: dict[str, dict[str, Any]]) -> None:
    """Renderiza la app Streamlit de analisis del modelo."""
    st.set_page_config(page_title="STATUS5 - Random Forest", layout="wide")
    st.title("Analisis del modelo STATUS5")
    st.caption("Validacion Stratified K-Fold, Optuna y F1 macro como metrica principal.")

    render_sidebar(model_artifacts)

    available_keys = existing_model_keys(model_artifacts)
    if not available_keys:
        st.error("No se encontraron artefactos entrenados. Ejecuta python train_model.py.")
        for model_key, cfg in model_artifacts.items():
            _assert_exists(cfg["classification_report"], f"reporte de {model_key}")
            _assert_exists(cfg["confusion_matrix"], f"matriz de {model_key}")
        st.stop()

    results = {
        key: load_model_result(key, model_artifacts[key])
        for key in available_keys
    }

    render_single_model(results[available_keys[0]])
