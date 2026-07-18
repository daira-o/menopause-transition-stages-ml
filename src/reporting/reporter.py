"""
src/reporting/reporter.py
-------------------------
Reporting and artifact-persistence module.
Responsibilities:
  - Save the cleaned DataFrame to disk.
  - Export the elimination log as CSV.
"""

import pandas as pd
from pathlib import Path
from src.data.cleaner import CleaningResult


def save_clean_dataset(df: pd.DataFrame, path: Path) -> None:
    """
    Save the cleaned DataFrame as CSV for later use.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"   Cleaned dataset saved: '{path}'")


def save_elimination_log(result: CleaningResult, path: Path) -> None:
    """
    Export the removed-column log to CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df_log = result.log_as_dataframe()
    df_log.to_csv(path, index=False, encoding="utf-8-sig")

    if df_log.empty:
        print(f"   No variables were removed. Empty log saved: '{path}'")
    else:
        print(f"   Elimination log saved: '{path}' ({len(df_log)} variables)")
