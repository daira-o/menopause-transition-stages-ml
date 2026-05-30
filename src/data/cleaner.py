"""
src/data/cleaner.py
-------------------
Módulo de limpieza de datos.
Aplica filtros configurables sobre un DataFrame y registra cada decisión
en un log estructurado para trazabilidad total del pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.constants import MISSING_VALUE_CODES

STATUS5_VALID_CATEGORIES = [2, 3, 4, 5]
PRELIMINARY_SELECTED_COLUMNS = [
    "HIGHBP5",
    "HBCHOLE5",
    "FIBRUTR5",
    "NERVES5",
    "ENERGY5",
    "WORNOUT5",
    "TIRED5",
    "IRRITAB5",
    "FEELBLU5",
    "NRVOUS5",
    "MOODCHG5",
    "HOTFLAS5",
    "NITESWE5",
    "COLDSWE5",
    "BRSTPAI5",
    "VAGINDR5",
    "DESIRSE5",
    "ENGAGSE5",
    "LUBRICN5",
    "PELVIC5",
    "TRBLSLE5",
    "WAKEUP5",
    "WAKEARL5",
    "SLEEPQL5",
    "GETUPUR5",
    "FORGET5",
    "BMI5",
    "RACE",
    "STATUS5",
    "AGE5"
]

@dataclass
class EliminationLog:
    """
    Registro inmutable de una columna eliminada durante la limpieza.

    Attributes
    ----------
    columna : str
        Nombre de la columna eliminada.
    motivo : str
        Descripción legible del criterio que causó la eliminación.
    paso : int
        Número del paso del pipeline donde ocurrió la eliminación.
    """
    columna: str
    motivo: str
    paso: int


@dataclass
class CleaningResult:
    """
    Resultado completo del proceso de limpieza.

    Attributes
    ----------
    df : pd.DataFrame
        DataFrame resultante tras aplicar todos los filtros.
    log : list[EliminationLog]
        Lista de todas las columnas eliminadas con su trazabilidad.
    """
    df: pd.DataFrame
    log: list[EliminationLog] = field(default_factory=list)

    def log_as_dataframe(self) -> pd.DataFrame:
        """Exporta el log de eliminaciones como DataFrame."""
        if not self.log:
            return pd.DataFrame(columns=["Paso", "Variable", "Motivo_de_eliminacion"])
        return pd.DataFrame(
            [
                {
                    "Paso": e.paso,
                    "Variable": e.columna,
                    "Motivo_de_eliminacion": e.motivo,
                }
                for e in self.log
            ]
        )


def filter_status5_clear_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina filas con STATUS5 en categorías hormonal o quirúrgicamente enmascaradas.

    Se conservan solo las categorías claras:
      2, 3, 4 y 5

    Se eliminan:
      1 Histerectomía
      6 Embarazo
      7 Desconocido por HT
      8 Desconocido por Histerectomía
      valores faltantes o no interpretables
    """
    if "STATUS5" not in df.columns:
        raise KeyError("No se encontró la columna requerida 'STATUS5'.")

    status = pd.to_numeric(df["STATUS5"], errors="coerce")
    mask = status.isin(STATUS5_VALID_CATEGORIES)
    filas_eliminadas = len(df) - int(mask.sum())

    print(
        f"  -> Filas eliminadas por STATUS5 enmascarado/no claro: "
        f"{filas_eliminadas:,}"
    )

    return df.loc[mask].copy()


def select_preliminary_columns(
    df: pd.DataFrame,
    paso: int = 0,
) -> tuple[pd.DataFrame, list[EliminationLog]]:
    """
    Selecciona las columnas definidas a priori para el análisis.

    Este paso ocurre después del filtrado de filas por STATUS5 y antes de
    calcular nulidad o baja varianza.
    """
    missing_cols = [col for col in PRELIMINARY_SELECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        missing = ", ".join(missing_cols)
        raise KeyError(f"Faltan columnas preliminares requeridas: {missing}")

    cols_eliminar = [col for col in df.columns if col not in PRELIMINARY_SELECTED_COLUMNS]
    log = [
        EliminationLog(
            columna=col,
            motivo="Eliminada por no pertenecer al set preliminar seleccionado",
            paso=paso,
        )
        for col in cols_eliminar
    ]

    print(f"  -> Columnas preliminares seleccionadas: {len(PRELIMINARY_SELECTED_COLUMNS):,}")
    print(f"  -> Columnas descartadas fuera del set seleccionado: {len(cols_eliminar):,}")

    return df.loc[:, PRELIMINARY_SELECTED_COLUMNS].copy(), log


def clean_empty_spaces(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reemplaza strings vacíos, espacios y puntos usados como missing por NaN.

    Ejemplos convertidos a NaN:
      ""
      " "
      "   "
      "."
      " ."
      ". "
      " . "
    """
    df = df.replace(r"^\s*$", np.nan, regex=True)
    df = df.replace(r"^\s*\.\s*$", np.nan, regex=True)
    return df


def replace_missing_codes_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reemplaza códigos especiales de valores faltantes por NaN.

    Códigos considerados missing:
      -9 Missing
      -8 Do not know
      -7 Refused
      -1 N/A se conserva como categoria valida.

    Importante:
      1 = No y 2 = Yes se conservan como valores válidos.
    """
    return df.replace(MISSING_VALUE_CODES, np.nan)


def filter_high_nullity(
    df: pd.DataFrame,
    umbral: float = 0.50,
    paso: int = 2,
) -> tuple[pd.DataFrame, list[EliminationLog]]:
    """
    Elimina columnas con proporción de nulos superior al umbral.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    umbral : float
        Fracción máxima permitida de nulos (default: 0.50 -> 50%).
    paso : int
        Identificador del paso del pipeline para el log.

    Returns
    -------
    tuple[pd.DataFrame, list[EliminationLog]]
        DataFrame filtrado y lista de columnas eliminadas con sus motivos.
    """
    pct_nulos = df.isnull().mean()
    cols_eliminar = pct_nulos[pct_nulos > umbral].index.tolist()
    log: list[EliminationLog] = []

    for col in cols_eliminar:
        pct = pct_nulos[col] * 100
        motivo = (
            f"Alta nulidad: {pct:.1f}% de valores nulos "
            f"(umbral configurado: {umbral * 100:.0f}%)"
        )
        log.append(EliminationLog(columna=col, motivo=motivo, paso=paso))
        print(f"  X '{col}' - {motivo}")

    if not cols_eliminar:
        print(f"  OK Ninguna columna superó el umbral de nulidad ({umbral * 100:.0f}%).")

    return df.drop(columns=cols_eliminar), log


def filter_low_variance(
    df: pd.DataFrame,
    umbral: float = 0.95,
    paso: int = 3,
) -> tuple[pd.DataFrame, list[EliminationLog]]:
    """
    Elimina columnas cuasi-constantes donde el valor más frecuente
    supera el umbral indicado.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    umbral : float
        Fracción máxima permitida para el valor dominante (default: 0.95 -> 95%).
    paso : int
        Identificador del paso del pipeline para el log.

    Returns
    -------
    tuple[pd.DataFrame, list[EliminationLog]]
        DataFrame filtrado y lista de columnas eliminadas con sus motivos.
    """
    cols_eliminar: list[str] = []
    log: list[EliminationLog] = []

    for col in df.columns:
        freq = df[col].value_counts(normalize=True, dropna=True)

        if freq.empty:
            motivo = "Columna completamente vacía (sin valores no nulos)"
        elif freq.iloc[0] > umbral:
            pct = freq.iloc[0] * 100
            valor_dom = freq.index[0]
            motivo = (
                f"Baja varianza: '{valor_dom}' representa {pct:.1f}% "
                f"de los registros (umbral: {umbral * 100:.0f}%)"
            )
        else:
            continue

        cols_eliminar.append(col)
        log.append(EliminationLog(columna=col, motivo=motivo, paso=paso))
        print(f"  X '{col}' - {motivo}")

    if not cols_eliminar:
        print(
            f"  OK Ninguna columna superó el umbral de valor dominante "
            f"({umbral * 100:.0f}%)."
        )

    return df.drop(columns=cols_eliminar), log


def run_cleaning(
    df: pd.DataFrame,
    umbral_nulos: float = 0.50,
    umbral_dominante: float = 0.95,
) -> CleaningResult:
    """
    Orquesta todos los filtros de limpieza en secuencia y devuelve
    un CleaningResult con el DataFrame limpio y el log completo.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame crudo de entrada.
    umbral_nulos : float
        Umbral para el filtro de nulidad.
    umbral_dominante : float
        Umbral para el filtro de baja varianza.

    Returns
    -------
    CleaningResult
    """
    log_total: list[EliminationLog] = []

    print("\n  [Paso 0] Seleccionando columnas preliminares...")
    df, log0 = select_preliminary_columns(df, paso=0)
    log_total.extend(log0)
    print(f"  -> Tras selección preliminar: {df.shape[1]:,} columnas restantes")

    print("\n  [Paso 1] Filtrando STATUS5 biológico/quirúrgico claro...")
    df = filter_status5_clear_categories(df)
    print(f"  -> Tras filtro STATUS5: {df.shape[0]:,} filas restantes")

    print("\n  [Paso 2] Limpiando espacios vacíos...")
    df = clean_empty_spaces(df)

    print("\n  [Paso 2] Reemplazando códigos missing por NaN...")
    df = replace_missing_codes_with_nan(df)

    print("\n  [Paso 3] Filtro de nulidad...")
    df, log2 = filter_high_nullity(df, umbral=umbral_nulos, paso=3)
    log_total.extend(log2)
    print(f"  -> Tras filtro: {df.shape[1]:,} columnas restantes")

    print("\n  [Paso 4] Filtro de baja varianza...")
    df, log3 = filter_low_variance(df, umbral=umbral_dominante, paso=4)
    log_total.extend(log3)
    print(f"  -> Tras filtro: {df.shape[1]:,} columnas restantes")

    return CleaningResult(df=df, log=log_total)
