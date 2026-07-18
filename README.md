# Menopause Transition Stage Classification with Random Forest

This repository contains a machine learning pipeline for classifying menopause
transition stages from clinical, demographic, and symptom data collected in the
Study of Women’s Health Across the Nation (SWAN).

The project evaluates how well a Random Forest model can distinguish four
menopause transition stages without relying on hormonal biomarkers. The workflow
includes data cleaning, feature selection, class-balanced training,
cross-validation, model evaluation, SHAP-based interpretation, and generation of
figures for academic reporting.

This project was developed for academic research and portfolio purposes. It is
not intended for clinical diagnosis or individual patient decision-making.

## Research Question

Can clinical, demographic, menstrual, and symptom variables help distinguish
menopause transition stages when hormonal biomarkers are not available?

The target variable is `STATUS5`, restricted to the following four categories:

| Code | Stage |
|---:|---|
| 2 | Natural postmenopause |
| 3 | Late perimenopause |
| 4 | Early perimenopause |
| 5 | Premenopause |

The task is treated as a multiclass classification problem.

## Dataset

The analysis uses data from Visit 05 of the **Study of Women’s Health Across the
Nation (SWAN)**.

SWAN is a longitudinal study designed to investigate biological, psychological,
and social changes during the menopausal transition.

The working dataset includes variables related to:

- Age and demographic characteristics.
- Menstrual history.
- Vasomotor symptoms.
- Sleep-related symptoms.
- Self-reported health information.
- Other clinical and behavioral variables available at the selected visit.

The original data files are not redistributed through this repository. Users
must obtain the dataset from its official source and comply with the applicable
access, citation, and usage conditions.

Participant identifiers and raw participant-level records should not be included
in the public repository.

## Methodology

The project follows the workflow below:

```text
SWAN Visit 05 data
        │
        ▼
Data cleaning and eligibility filtering
        │
        ▼
Variable typing and feature selection
        │
        ▼
Median imputation
        │
        ▼
Class-balanced Random Forest
        │
        ▼
Stratified 5-fold cross-validation
        │
        ▼
Evaluation and SHAP interpretation
```

### Data cleaning

The cleaning pipeline:

- Selects the variables defined for the analysis.
- Restricts the target to `STATUS5` classes 2, 3, 4, and 5.
- Converts SWAN missing-value codes to missing values.
- Removes variables above the configured missingness threshold.
- Removes near-constant variables.
- Converts predictors to the expected numerical or categorical representation.
- Excludes records that do not meet the analysis criteria.

Run the cleaning pipeline with:

```bash
python main.py
```

Generated files include:

```text
data/processed/datos_limpios.csv
outputs/info/variables_eliminadas.csv
```

> The handling of the `-1` code should match the implementation in the cleaning
> script. Verify whether it is treated as missing or retained as a valid value
> before publishing this documentation.

### Feature processing

Variables are organized by data type, including:

- Continuous variables.
- Binary variables.
- Ordinal variables.
- Nominal variables.

Feature selection considers:

- Missingness.
- Low variance.
- Redundancy.
- Mutual information.
- Clinical relevance.

Median imputation is applied within the modeling pipeline to prevent information
leakage across validation folds.

## Model Training

Train the model with:

```bash
python train_model.py
```

The current implementation uses a `RandomForestClassifier` with:

- `class_weight="balanced"` to reduce the impact of class imbalance.
- Stratified 5-fold cross-validation.
- Macro F1-score as the main optimization objective.
- Hyperparameter tuning with Optuna.
- A fixed random seed for reproducibility.

The search space includes:

- Number of trees.
- Maximum tree depth.
- Splitting criterion.
- Minimum number of samples per leaf.

The final model and evaluation artifacts are stored under:

```text
outputs/models/
outputs/metrics/
```

## Interpretation

Model interpretation is performed using:

- Random Forest feature importance.
- SHAP values.
- Class-specific confusion matrices.
- Symptom comparisons across transition stages.
- Association analyses between selected variables and `STATUS5`.

The interpretation workflow examines how age, vasomotor symptoms, sleep-related
variables, and other selected features contribute to the model predictions.

SHAP values and symptom comparisons are used to explore overlap between adjacent
transition stages and identify variables that may influence classification.

Generate the final figures with:

```bash
python generate_figures.py
```

The script produces:

- Target class distribution.
- Missing-value summaries.
- Confusion matrix.
- Feature-importance plots.
- SHAP summary plots.
- Selected clinical and symptom comparisons.

## Results Dashboard

Launch the Streamlit dashboard with:

```bash
streamlit run visualize_results.py
```

The dashboard presents:

- Global model metrics.
- Cross-validated confusion matrix.
- Per-class precision, recall, and F1-score.
- Selected hyperparameters.
- Feature importance.
- SHAP visualizations.
- Paths to generated artifacts.

## Repository Structure

```text
menopause-transition-stages-ml/
├── config.py
├── main.py
├── train_model.py
├── generate_figures.py
├── visualize_results.py
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── data/
│   ├── raw/
│   ├── processed/
│   └── clasificacion_variables_ml - clasificacion_variables_ml.csv
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── data/
│   ├── models/
│   ├── evaluation/
│   └── reporting/
├── outputs/
│   ├── models/
│   ├── metrics/
│   ├── figures/
│   └── info/
└── tests/
    └── test_cleaner.py
```

## Installation

Install the runtime dependencies:

```bash
pip install -r requirements.txt
```

For development and testing:

```bash
pip install -r requirements-dev.txt
pytest tests/
```

## Configuration

Project paths, preprocessing thresholds, cross-validation parameters, and
Optuna settings are defined in `config.py`.

The main configurable values include:

- Missingness threshold.
- Dominant-value threshold.
- Number of cross-validation folds.
- Number of Optuna trials.
- Random seed.
- Input and output paths.

Some internal variable and file names remain in Spanish to preserve compatibility
with the original analysis scripts and generated artifacts.

## Exploratory Analysis

The exploratory notebook is available at:

```text
notebooks/eda.ipynb
```

It includes:

- Dataset quality checks.
- Target distribution.
- Missing-value analysis.
- Feature associations.
- Symptom comparisons.
- Random Forest feature importance.
- SHAP-based interpretation.

## Limitations

- The analysis uses data from a single SWAN visit.
- The model does not predict future transition between stages.
- Hormonal biomarkers are not included.
- The dataset is imbalanced, especially for the premenopause class.
- Adjacent menopause stages show overlapping clinical and symptom profiles.
- Hyperparameter optimization and evaluation are based on the same
  cross-validation framework rather than nested cross-validation.
- The model has not been validated on an independent external cohort.
- Age has a strong influence on the predictions and may limit generalization.
- The results should not be interpreted as clinical diagnostic performance.

## Future Work

Potential next steps include:

- Nested cross-validation.
- External validation.
- Comparison with additional classifiers.
- Probability calibration.
- Evaluation against an expert-defined reference standard.
- Sensitivity analyses with and without age.
- Statistical comparison between models.
- Improved evaluation of minority classes.

## Intended Use

This repository is intended for:

- Academic research.
- Reproducible machine learning experiments.
- Multiclass classification studies using clinical tabular data.
- Educational and portfolio use.

It is not intended for:

- Clinical diagnosis.
- Patient screening.
- Treatment recommendations.
- Autonomous medical decision-making.

## Author

**Daira Orlandini**

Computer Engineering student  
Universidad de Palermo