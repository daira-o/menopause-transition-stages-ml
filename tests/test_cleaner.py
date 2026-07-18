"""
tests/test_cleaner.py
---------------------
Unit tests for the cleaning filters.
Run with:  pytest tests/
"""

import pandas as pd
import pytest

from src.data.cleaner import (
    PRELIMINARY_SELECTED_COLUMNS,
    filter_high_nullity,
    filter_low_variance,
    replace_missing_codes_with_nan,
    run_cleaning,
)


# Fixtures.

@pytest.fixture
def df_con_nulos():
    """DataFrame with one column above 50% missing values."""
    return pd.DataFrame({
        "id":       [1, 2, 3, 4, 5, 6],
        "nombre":   ["a", "b", "c", "d", "e", "f"],
        "col_mala": [None, None, None, None, "x", None],  # 83% missing -> should be removed
    })


@pytest.fixture
def df_con_varianza_baja():
    """DataFrame with one near-constant column (>95% same value)."""
    return pd.DataFrame({
        "id":        range(100),
        "constante": ["SI"] * 97 + ["NO"] * 3,  # 97% -> should be removed
        "variable":  range(100),
    })


# Tests for filter_high_nullity.

def test_elimina_columna_con_muchos_nulos(df_con_nulos):
    df_out, log = filter_high_nullity(df_con_nulos, umbral=0.50)
    assert "col_mala" not in df_out.columns
    assert len(log) == 1
    assert log[0].columna == "col_mala"


def test_conserva_columna_con_pocos_nulos():
    df = pd.DataFrame({"a": [1, None, 3, 4], "b": [None, None, None, 4]})
    df_out, log = filter_high_nullity(df, umbral=0.50)
    assert "a" in df_out.columns    # 25% missing -> kept
    assert "b" not in df_out.columns  # 75% missing -> removed


def test_sin_nulos_no_elimina_nada():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df_out, log = filter_high_nullity(df, umbral=0.50)
    assert df_out.shape == df.shape
    assert len(log) == 0


def test_reemplaza_missing_codes_pero_conserva_menos_uno():
    df = pd.DataFrame({"a": [-9, -8, -7, -1, 1], "b": ["-9", "-8", "-7", "-1", "1"]})
    df_out = replace_missing_codes_with_nan(df)

    assert df_out["a"].isna().tolist() == [True, True, True, False, False]
    assert df_out["b"].isna().tolist() == [True, True, True, False, False]
    assert df_out.loc[3, "a"] == -1
    assert df_out.loc[3, "b"] == "-1"


# Tests for filter_low_variance.

def test_elimina_columna_cuasi_constante(df_con_varianza_baja):
    df_out, log = filter_low_variance(df_con_varianza_baja, umbral=0.95)
    assert "constante" not in df_out.columns
    assert any(e.columna == "constante" for e in log)


def test_conserva_columna_variable(df_con_varianza_baja):
    df_out, log = filter_low_variance(df_con_varianza_baja, umbral=0.95)
    assert "variable" in df_out.columns


def test_columna_vacia_se_elimina():
    df = pd.DataFrame({"a": [1, 2, 3], "vacia": [None, None, None]})
    df_out, log = filter_low_variance(df, umbral=0.95)
    assert "vacia" not in df_out.columns


# Integration tests for run_cleaning.

def _df_preliminar_valido(n=20):
    data = {col: [1, 2] * (n // 2) for col in PRELIMINARY_SELECTED_COLUMNS}
    data["STATUS5"] = [2, 3, 4, 5] * (n // 4)
    data["RACE"] = [1, 2, 3, 4] * (n // 4)
    return pd.DataFrame(data)


def test_run_cleaning_log_acumula_pasos():
    df = _df_preliminar_valido()
    df["HBCHOLE5"] = [None] * 15 + [1] * 5
    df["FIBRUTR5"] = [1] * 20

    resultado = run_cleaning(df, umbral_nulos=0.50, umbral_dominante=0.95)
    pasos = {e.paso for e in resultado.log}
    assert 3 in pasos
    assert 4 in pasos
    assert "HBCHOLE5" not in resultado.df.columns
    assert "FIBRUTR5" not in resultado.df.columns
    assert "HIGHBP5" in resultado.df.columns


def test_run_cleaning_dataframe_limpio_shape():
    df = _df_preliminar_valido()
    df["HBCHOLE5"] = [None] * 15 + [1] * 5
    df["FIBRUTR5"] = [1] * 20

    resultado = run_cleaning(df, umbral_nulos=0.50, umbral_dominante=0.95)
    assert resultado.df.shape[1] == len(PRELIMINARY_SELECTED_COLUMNS) - 2
