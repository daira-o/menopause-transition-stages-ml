"""
main.py
-------
Entry point for the data-cleaning pipeline.
Runs the modules in order and prints a final summary.

Usage:
    python main.py
"""

import sys
import time
from config import (
    INPUT_FILE, CLEAN_FILE,
    LOG_FILE,
    UMBRAL_NULOS, UMBRAL_DOMINANTE,
)
from src.data.loader import load_tsv
from src.data.cleaner import run_cleaning
from src.reporting.reporter import save_clean_dataset, save_elimination_log


def _header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main() -> None:
    start = time.time()

    # Step 1: load the raw input file.
    _header("STEP 1 - Load input file")
    df_raw = load_tsv(INPUT_FILE)

    # Steps 2 and 3: apply cleaning and information-quality filters.
    _header("STEPS 2-3 - Quality and information filters")
    result = run_cleaning(
        df_raw,
        umbral_nulos=UMBRAL_NULOS,
        umbral_dominante=UMBRAL_DOMINANTE,
    )

    # Step 4: persist the generated artifacts.
    _header("STEP 4 - Save artifacts")
    save_clean_dataset(result.df, CLEAN_FILE)
    save_elimination_log(result, LOG_FILE)

    elapsed = time.time() - start
    # Count log entries according to the step numbers defined in cleaner.py.
    nullity_removed = sum(1 for e in result.log if e.paso == 3)
    low_variance_removed = sum(1 for e in result.log if e.paso == 4)

    _header("FINAL SUMMARY")
    print(f"  Original dataset        : {df_raw.shape[0]:,} rows x {df_raw.shape[1]:,} columns")
    print(f"  Cleaned dataset         : {result.df.shape[0]:,} rows x {result.df.shape[1]:,} columns")
    print(f"  Removed for missingness : {nullity_removed}")
    print(f"  Removed for low variance: {low_variance_removed}")
    print(f"  Elimination log         : {LOG_FILE}")
    print(f"  Cleaned dataset         : {CLEAN_FILE}")
    print(f"  Total time              : {elapsed:.1f}s")
    print("=" * 60)
    print("  Pipeline completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n  Fatal error: {exc}", file=sys.stderr)
        sys.exit(1)
