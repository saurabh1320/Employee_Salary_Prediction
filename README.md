# Employee Salary Prediction
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Model%20Training-F7931E?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success)

A reproducible machine-learning project that explores employee attributes and trains a regression model to estimate salary.

> **Important:** This is an educational demonstration, not a compensation benchmark or a tool for making employment decisions. Predictions can reflect biases and limitations in the dataset.

## Project goals

- Inspect and clean tabular employee data.
- Explore salary relationships with age, education, experience, and job title.
- Train a regression model with preprocessing fitted only on training data.
- Evaluate with MAE, RMSE, and R².
- Provide a reusable prediction function and automated tests.

## Dataset

The repository includes `Dataset09-Employee-salary-prediction.csv`.

Features in the provided file:
- `Age`
- `Gender`
- `Education Level`
- `Job Title`
- `Years of Experience`

Target:
- `Salary`

The CSV has 375 rows and 6 columns before cleaning. The supplied data contains missing values and exact duplicate rows. The training script drops rows with missing target values, removes exact duplicates, and imputes missing feature values within the model pipeline. Review the dataset provenance and licensing before redistribution or production use.

## Quick start

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Activate the environment:

- Windows: `.venv\\Scripts\\activate`
- macOS/Linux: `source .venv/bin/activate`

Install dependencies:

```bash
pip install -r requirements.txt
```

Train, compare, and evaluate:

```bash
python -m src.train
```

The script compares a median baseline, Linear Regression, Random Forest, and Extra Trees. It selects a model using mean 5-fold cross-validation MAE on the training split, then reports held-out test MAE, RMSE, and R². It saves the selected pipeline to `models/salary_model.joblib`, final metrics to `reports/metrics.json`, and comparison details to `reports/model_comparison.json`.

## Make a prediction

After training:

```bash
python -m src.predict
```

Edit the example raw input in `src/predict.py` to try a different employee profile. Inputs use the original feature names and raw category values; the saved pipeline handles preprocessing.

## Project structure

```text
.
├── Dataset09-Employee-salary-prediction.csv
├── notebooks/
│   └── EmployeeSalaryPrediction.ipynb
├── src/
│   ├── __init__.py
│   ├── train.py
│   └── predict.py
├── tests/
│   └── test_pipeline.py
├── reports/              # Generated metrics and model comparison
├── models/               # Generated trained model
├── requirements.txt
├── .gitignore
└── README.md
```

## Methodology

- A fixed random seed makes the holdout split reproducible.
- Candidate models are compared using 5-fold cross-validation on the training split; the test split is reserved for final evaluation.
- Numeric and categorical preprocessing is performed inside a scikit-learn `Pipeline` / `ColumnTransformer`.
- Imputation and one-hot encoding are learned from training data only.
- Unknown categories at inference time are ignored by the encoder rather than causing prediction to fail.
- Regression metrics are calculated on a held-out test set.

## Model Comparison

Four regression models were evaluated to compare their performance on the employee salary prediction task. Models were compared using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the coefficient of determination (R²).

### Test Set Results

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|
| Extra Trees Regressor | 11,759.50 | 16,942.39 | 0.8481 |
| Random Forest Regressor | 11,612.08 | 16,873.61 | 0.8493 |
| Linear Regression | 12,066.48 | 15,585.17 | 0.8715 |
| Dummy Regressor (Baseline) | 36,000.00 | 43,474.13 | -0.0001 |

*Lower MAE and RMSE indicate smaller prediction errors; higher R² indicates a better fit to the test data.*

### Evaluation Setup

- Dataset after duplicate removal and handling missing target values: 324 rows.
- Train/test split: 80% training and 20% testing.
- Random state: 42.
- Model selection: 5-fold cross-validation using MAE on the training data.
- Final evaluation: MAE, RMSE, and R² on the held-out test set.

### Results Interpretation

- **Random Forest** achieved the lowest test MAE among the three trained regression models.
- **Linear Regression** achieved the lowest test RMSE and highest test R² in this experiment.
- **Extra Trees** was selected using cross-validation MAE on the training set; it did not achieve the best score on every test metric.
- The Dummy Regressor provides a baseline to assess whether the trained models improve on a simple prediction strategy.

These results are specific to the current dataset and train/test split. They should not be interpreted as a guarantee of performance on unseen real-world salary data.

## Limitations

- Performance depends on the dataset's representativeness, quality, and provenance.
- Salary values may contain anomalies or reflect historical inequities.
- A random holdout score does not establish real-world or future-market accuracy.
- Gender is included in the source data, but the default model excludes it to reduce direct use of a sensitive attribute. This does not guarantee fairness: other features can act as proxies.
- Validate the dataset, features, and use case before interpreting predictions.

## Future improvements

- Expand cross-validation and tune promising models only after the baseline comparison.
- Add residual plots and error analysis by relevant groups.
- Record dataset provenance, license, and limitations.
- Add a small web interface or API after the baseline pipeline is validated.

## License

No license has been specified yet. Add a license only after confirming you have the right to distribute the code and dataset.
