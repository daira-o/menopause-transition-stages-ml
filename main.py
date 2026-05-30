"""
main.py
-------
Punto de entrada del pipeline de limpieza.
Orquesta los módulos en orden y muestra un resumen al finalizar.

Uso:
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


def _header(titulo: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {titulo}")
    print(f"{'=' * 60}")


def main() -> None:
    inicio = time.time()

    # ── PASO 1: Carga ─────────────────────────────────────────────────────────
    _header("PASO 1 · Carga del archivo")
    df_raw = load_tsv(INPUT_FILE)

    # ── PASOS 2 y 3: Limpieza ─────────────────────────────────────────────────
    _header("PASOS 2-3 · Filtros de calidad e información")
    resultado = run_cleaning(
        df_raw,
        umbral_nulos=UMBRAL_NULOS,
        umbral_dominante=UMBRAL_DOMINANTE,
    )

    # ── PASO 4: Persistencia de artefactos ────────────────────────────────────
    _header("PASO 4 · Guardando artefactos")
    save_clean_dataset(resultado.df, CLEAN_FILE)
    save_elimination_log(resultado, LOG_FILE)

    # ── RESUMEN ───────────────────────────────────────────────────────────────
    elapsed = time.time() - inicio
    # Ajustamos el conteo de logs según los pasos definidos en cleaner.py
    cols_nulidad  = sum(1 for e in resultado.log if e.paso == 3)
    cols_varianza = sum(1 for e in resultado.log if e.paso == 4)

    _header("RESUMEN FINAL")
    print(f"  Dataset original      : {df_raw.shape[0]:,} filas × {df_raw.shape[1]:,} columnas")
    print(f"  Dataset limpio        : {resultado.df.shape[0]:,} filas × {resultado.df.shape[1]:,} columnas")
    print(f"  Eliminadas (nulidad)  : {cols_nulidad}")
    print(f"  Eliminadas (varianza) : {cols_varianza}")
    print(f"  Log de eliminaciones  : {LOG_FILE}")
    print(f"  Dataset limpio        : {CLEAN_FILE}")
    print(f"  Tiempo total          : {elapsed:.1f}s")
    print("=" * 60)
    print("  ✅  Pipeline completado sin errores.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n  ❌ Error fatal: {exc}", file=sys.stderr)
        sys.exit(1)
