# Employee Salary Prediction

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
