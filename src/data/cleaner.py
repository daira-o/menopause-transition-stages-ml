"""
src/data/cleaner.py
-------------------
Data-cleaning module.
Applies configurable filters to a DataFrame and records each decision in a
structured log for full pipeline traceability.
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
    Immutable record for a column removed during cleaning.

    Attributes
    ----------
    columna : str
        Name of the removed column.
    motivo : str
        Human-readable description of the removal criterion.
    paso : int
        Pipeline step where the removal occurred.
    """
    columna: str
    motivo: str
    paso: int


@dataclass
class CleaningResult:
    """
    Complete result of the cleaning process.

    Attributes
    ----------
    df : pd.DataFrame
        DataFrame produced after applying all filters.
    log : list[EliminationLog]
        List of all removed columns with traceability details.
    """
    df: pd.DataFrame
    log: list[EliminationLog] = field(default_factory=list)

    def log_as_dataframe(self) -> pd.DataFrame:
        """Export the elimination log as a DataFrame."""
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
    Remove rows with hormonally or surgically masked STATUS5 categories.

    Only clear categories are kept:
      2, 3, 4 y 5

    Removed categories:
      1 Hysterectomy
      6 Pregnancy
      7 Unknown due to hormone therapy
      8 Unknown due to hysterectomy
      missing or uninterpretable values
    """
    if "STATUS5" not in df.columns:
        raise KeyError("Required column 'STATUS5' was not found.")

    status = pd.to_numeric(df["STATUS5"], errors="coerce")
    mask = status.isin(STATUS5_VALID_CATEGORIES)
    removed_rows = len(df) - int(mask.sum())

    print(
        f"  -> Rows removed due to masked or unclear STATUS5: "
        f"{removed_rows:,}"
    )

    return df.loc[mask].copy()


def select_preliminary_columns(
    df: pd.DataFrame,
    paso: int = 0,
) -> tuple[pd.DataFrame, list[EliminationLog]]:
    """
    Select the columns defined a priori for the analysis.

    This step occurs after filtering STATUS5 rows and before calculating
    missingness or low variance.
    """
    missing_cols = [col for col in PRELIMINARY_SELECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        missing = ", ".join(missing_cols)
        raise KeyError(f"Missing required preliminary columns: {missing}")

    cols_eliminar = [col for col in df.columns if col not in PRELIMINARY_SELECTED_COLUMNS]
    log = [
        EliminationLog(
            columna=col,
            motivo="Removed because it is outside the selected preliminary feature set",
            paso=paso,
        )
        for col in cols_eliminar
    ]

    print(f"  -> Preliminary columns selected: {len(PRELIMINARY_SELECTED_COLUMNS):,}")
    print(f"  -> Columns discarded outside the selected set: {len(cols_eliminar):,}")

    return df.loc[:, PRELIMINARY_SELECTED_COLUMNS].copy(), log


def clean_empty_spaces(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace empty strings, whitespace, and dot placeholders used as missing values with NaN.

    Examples converted to NaN:
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
    Replace special missing-value codes with NaN.

    Codes treated as missing:
      -9 Missing
      -8 Do not know
      -7 Refused
      -1 N/A is preserved as a valid category.

    Important:
      1 = No and 2 = Yes are preserved as valid values.
    """
    return df.replace(MISSING_VALUE_CODES, np.nan)


def filter_high_nullity(
    df: pd.DataFrame,
    umbral: float = 0.50,
    paso: int = 2,
) -> tuple[pd.DataFrame, list[EliminationLog]]:
    """
    Remove columns whose missing-value proportion is above the threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    umbral : float
        Maximum allowed missing-value fraction (default: 0.50 -> 50%).
    paso : int
        Pipeline step identifier for the log.

    Returns
    -------
    tuple[pd.DataFrame, list[EliminationLog]]
        Filtered DataFrame and list of removed columns with reasons.
    """
    pct_nulos = df.isnull().mean()
    cols_eliminar = pct_nulos[pct_nulos > umbral].index.tolist()
    log: list[EliminationLog] = []

    for col in cols_eliminar:
        pct = pct_nulos[col] * 100
        motivo = (
            f"High missingness: {pct:.1f}% missing values "
            f"(configured threshold: {umbral * 100:.0f}%)"
        )
        log.append(EliminationLog(columna=col, motivo=motivo, paso=paso))
        print(f"  X '{col}' - {motivo}")

    if not cols_eliminar:
        print(f"  OK No column exceeded the missingness threshold ({umbral * 100:.0f}%).")

    return df.drop(columns=cols_eliminar), log


def filter_low_variance(
    df: pd.DataFrame,
    umbral: float = 0.95,
    paso: int = 3,
) -> tuple[pd.DataFrame, list[EliminationLog]]:
    """
    Remove near-constant columns whose most frequent value exceeds the threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    umbral : float
        Maximum allowed fraction for the dominant value (default: 0.95 -> 95%).
    paso : int
        Pipeline step identifier for the log.

    Returns
    -------
    tuple[pd.DataFrame, list[EliminationLog]]
        Filtered DataFrame and list of removed columns with reasons.
    """
    cols_eliminar: list[str] = []
    log: list[EliminationLog] = []

    for col in df.columns:
        freq = df[col].value_counts(normalize=True, dropna=True)

        if freq.empty:
            motivo = "Completely empty column (no non-null values)"
        elif freq.iloc[0] > umbral:
            pct = freq.iloc[0] * 100
            valor_dom = freq.index[0]
            motivo = (
                f"Low variance: '{valor_dom}' represents {pct:.1f}% "
                f"of records (threshold: {umbral * 100:.0f}%)"
            )
        else:
            continue

        cols_eliminar.append(col)
        log.append(EliminationLog(columna=col, motivo=motivo, paso=paso))
        print(f"  X '{col}' - {motivo}")

    if not cols_eliminar:
        print(
            f"  OK No column exceeded the dominant-value threshold "
            f"({umbral * 100:.0f}%)."
        )

    return df.drop(columns=cols_eliminar), log


def run_cleaning(
    df: pd.DataFrame,
    umbral_nulos: float = 0.50,
    umbral_dominante: float = 0.95,
) -> CleaningResult:
    """
    Run all cleaning filters in sequence and return a CleaningResult with the
    cleaned DataFrame and complete log.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame.
    umbral_nulos : float
        Threshold for the missingness filter.
    umbral_dominante : float
        Threshold for the low-variance filter.

    Returns
    -------
    CleaningResult
    """
    log_total: list[EliminationLog] = []

    print("\n  [Step 0] Selecting preliminary columns...")
    df, log0 = select_preliminary_columns(df, paso=0)
    log_total.extend(log0)
    print(f"  -> After preliminary selection: {df.shape[1]:,} columns remaining")

    print("\n  [Step 1] Filtering for biologically/surgically clear STATUS5...")
    df = filter_status5_clear_categories(df)
    print(f"  -> After STATUS5 filter: {df.shape[0]:,} rows remaining")

    print("\n  [Step 2] Cleaning empty whitespace...")
    df = clean_empty_spaces(df)

    print("\n  [Step 2] Replacing missing-value codes with NaN...")
    df = replace_missing_codes_with_nan(df)

    print("\n  [Step 3] Missingness filter...")
    df, log2 = filter_high_nullity(df, umbral=umbral_nulos, paso=3)
    log_total.extend(log2)
    print(f"  -> After filter: {df.shape[1]:,} columns remaining")

    print("\n  [Step 4] Low-variance filter...")
    df, log3 = filter_low_variance(df, umbral=umbral_dominante, paso=4)
    log_total.extend(log3)
    print(f"  -> After filter: {df.shape[1]:,} columns remaining")

    return CleaningResult(df=df, log=log_total)
