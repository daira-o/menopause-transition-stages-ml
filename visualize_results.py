"""
visualize_results.py
--------------------
Streamlit app for inspecting and comparing STATUS5 models.

Usage:
    streamlit run visualize_results.py
"""

from config import MODEL_ARTIFACTS
from src.evaluation.visualizer import render_streamlit_app


render_streamlit_app(model_artifacts=MODEL_ARTIFACTS)
