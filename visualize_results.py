"""
visualize_results.py
--------------------
App Streamlit para inspeccionar y comparar modelos STATUS5.

Uso:
    streamlit run visualize_results.py
"""

from config import MODEL_ARTIFACTS
from src.evaluation.visualizer import render_streamlit_app


render_streamlit_app(model_artifacts=MODEL_ARTIFACTS)
