# Pipeline de Limpieza, Modelado y Evaluacion STATUS5

Proyecto de Data Science / Machine Learning para predecir `STATUS5` a partir de
datos clinicos/tabulares. La version actual mantiene un pipeline de limpieza,
entrenamiento con Random Forest, metricas y visualizacion de resultados.

## Estructura del proyecto

```text
menopause-transition-stages-ml/
|-- config.py
|-- main.py
|-- train_model.py
|-- visualize_results.py
|-- requirements.txt
|-- requirements-dev.txt
|-- pyproject.toml
|-- README.md
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- clasificacion_variables_ml - clasificacion_variables_ml.csv
|-- notebooks/
|-- src/
|   |-- __init__.py
|   |-- data/
|   |   |-- __init__.py
|   |   |-- loader.py
|   |   `-- cleaner.py
|   |-- models/
|   |   |-- __init__.py
|   |   |-- modeling.py
|   |   `-- random_forest.py
|   |-- evaluation/
|   |   |-- __init__.py
|   |   `-- visualizer.py
|   |-- reporting/
|   |   |-- __init__.py
|   |   `-- reporter.py
|-- outputs/
|   |-- models/
|   |-- metrics/
|   |-- figures/
|   `-- info/
`-- tests/
    `-- test_cleaner.py
```

## Uso rapido

```bash
pip install -r requirements.txt
python main.py
python train_model.py
streamlit run visualize_results.py
pip install -r requirements-dev.txt
pytest tests/
```

## Entradas y salidas

- Entrada cruda: `data/raw/datos.tsv`
- Diccionario/metadata auxiliar de variables: `data/clasificacion_variables_ml - clasificacion_variables_ml.csv`
- Dataset limpio: `data/processed/datos_limpios.csv`
- Modelo entrenado: `outputs/models/random_forest_status5.joblib`
- Metricas: `outputs/metrics/`
- Informacion auxiliar: `outputs/info/`
- Figuras futuras o exportadas: `outputs/figures/`

## Modelado STATUS5

El entrenamiento usa `data/processed/datos_limpios.csv` como entrada y separa
`STATUS5` como variable objetivo. El pipeline aplica imputacion por mediana y un
`RandomForestClassifier` con `class_weight="balanced"`, sin sobremuestreo
sintetico.

La optimizacion se realiza con Optuna usando validacion `StratifiedKFold(k=5)` y
F1-score macro como metrica objetivo. Los rangos configurados son:

- `n_estimators`: 100 a 1000
- `max_depth`: 5 a 15
- `criterion`: `gini` o `entropy`
- `min_samples_leaf`: rango conservador derivado de la clase minoritaria

Artefactos generados:

- `outputs/models/random_forest_status5.joblib`
- `outputs/metrics/confusion_matrix_status5.csv`
- `outputs/metrics/classification_report_status5.csv`
- `outputs/metrics/best_params_status5.json`
- `outputs/metrics/node_split_logic_status5.txt`

`visualize_results.py` abre una app Streamlit para analizar esos artefactos.

## Configuracion

Editar `config.py` para cambiar:

- Rutas de entrada/salida
- Umbral de nulidad (`UMBRAL_NULOS`, default 50%)
- Umbral de varianza (`UMBRAL_DOMINANTE`, default 95%)
- Parametros de entrenamiento (`N_SPLITS`, `OPTUNA_N_TRIALS`, `RANDOM_STATE`)

## Agregar un nuevo filtro

1. Agregar la funcion en `src/data/cleaner.py` con la misma firma general:
   `df, umbral, paso -> tuple[df, log]`.
2. Llamarla dentro de `run_cleaning()` en el mismo archivo.
3. Agregar el parametro correspondiente en `config.py`.
4. Escribir o actualizar tests en `tests/test_cleaner.py`.

## Alcance actual

La investigacion queda enfocada exclusivamente en Random Forest.
