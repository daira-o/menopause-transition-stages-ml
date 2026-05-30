"""
src/data/loader.py
------------------
Módulo de carga de datos.
Responsabilidad única: leer el archivo de entrada y devolver un DataFrame crudo.
"""

import pandas as pd
from pathlib import Path

from src.data.constants import MISSING_VALUE_CODES


def load_tsv(path: Path) -> pd.DataFrame:
    """
    Lee un archivo TSV y retorna un DataFrame.

    Parameters
    ----------
    path : Path
        Ruta al archivo .tsv

    Returns
    -------
    pd.DataFrame
        DataFrame crudo (df_raw).

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe en la ruta indicada.
    ValueError
        Si el archivo está vacío o no puede parsearse.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de entrada: '{path}'\n"
            f"Coloca 'datos.tsv' en la carpeta: {path.parent}"
        )

    df = pd.read_csv(
        path,
        sep="\t",
        low_memory=False,
        na_values=MISSING_VALUE_CODES,
        keep_default_na=False,
    )

    if df.empty:
        raise ValueError(f"El archivo '{path.name}' está vacío o no pudo parsearse correctamente.")

    print(f"  ✔ Archivo cargado: '{path.name}'")
    print(f"  → Dimensiones: {df.shape[0]:,} filas × {df.shape[1]:,} columnas")
    return df
