"""
src/reporting/reporter.py
-------------------------
Módulo de generación de reportes y persistencia de artefactos.
Responsabilidades:
  - Guardar el DataFrame limpio en disco.
  - Exportar el log de eliminaciones como CSV.
"""

import pandas as pd
from pathlib import Path
from src.data.cleaner import CleaningResult


def save_clean_dataset(df: pd.DataFrame, path: Path) -> None:
    """
    Persiste el DataFrame limpio en CSV para uso posterior.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"   ✔ Dataset limpio guardado: '{path}'")


def save_elimination_log(result: CleaningResult, path: Path) -> None:
    """
    Exporta el log de columnas eliminadas a un CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df_log = result.log_as_dataframe()
    df_log.to_csv(path, index=False, encoding="utf-8-sig")

    if df_log.empty:
        print(f"   ✔ No se eliminó ninguna variable. Log vacío guardado: '{path}'")
    else:
        print(f"   ✔ Log de eliminaciones guardado: '{path}' ({len(df_log)} variables)")
