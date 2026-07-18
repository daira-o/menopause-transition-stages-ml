"""
src/data/loader.py
------------------
Data-loading module.
Single responsibility: read the input file and return a raw DataFrame.
"""

import pandas as pd
from pathlib import Path

from src.data.constants import MISSING_VALUE_CODES


def load_tsv(path: Path) -> pd.DataFrame:
    """
    Read a TSV file and return a DataFrame.

    Parameters
    ----------
    path : Path
        Path to the .tsv file.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame (df_raw).

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    ValueError
        If the file is empty or cannot be parsed.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: '{path}'\n"
            f"Place 'datos.tsv' in this folder: {path.parent}"
        )

    df = pd.read_csv(
        path,
        sep="\t",
        low_memory=False,
        na_values=MISSING_VALUE_CODES,
        keep_default_na=False,
    )

    if df.empty:
        raise ValueError(f"The file '{path.name}' is empty or could not be parsed correctly.")

    print(f"  Loaded file: '{path.name}'")
    print(f"  Dimensions: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
    return df
