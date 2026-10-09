"""Train, compare, and evaluate employee salary regression models."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Dataset09-Employee-salary-prediction.csv"
MODEL_PATH = ROOT / "models" / "salary_model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
COMPARISON_PATH = ROOT / "reports" / "model_comparison.json"

NUMERIC_FEATURES = ["Age", "Years of Experience"]
CATEGORICAL_FEATURES = ["Education Level", "Job Title"]
TARGET = "Salary"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load data and apply documented row-level cleaning."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the CSV at the repository root."
        )
    data = pd.read_csv(path)
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing_columns = sorted(set(required) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    # Exact duplicates are removed. Review this policy if identical rows could
    # represent separate people; row identity is unavailable in this dataset.
    data = data.drop_duplicates().copy()
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    # Rows without a usable target cannot be used in supervised learning.
    data = data.dropna(subset=[TARGET]).copy()
    return data


def build_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps=numeric_steps), NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline() -> Pipeline:
    """Build the baseline linear regression pipeline."""
    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessor(scale_numeric=True)),
            ("regressor", LinearRegression()),
        ]
    )


def build_candidates() -> dict[str, Pipeline]:
    """Return comparable models using the same leakage-safe preprocessing."""
    return {
        "DummyRegressor (median)": Pipeline(
            [
                ("preprocessing", build_preprocessor()),
                ("regressor", DummyRegressor(strategy="median")),
            ]
        ),
        "Linear Regression": build_pipeline(),
        "Random Forest": Pipeline(
            [
                ("preprocessing", build_preprocessor()),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Extra Trees": Pipeline(
            [
                ("preprocessing", build_preprocessor()),
                (
                    "regressor",
                    ExtraTreesRegressor(
                        n_estimators=300,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def main() -> None:
    data = load_data()
    features = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    target = data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    results = []
    candidates = build_candidates()
    for name, model in candidates.items():
        cv_scores = cross_val_score(
            model,
            x_train,
            y_train,
            cv=5,
            scoring="neg_mean_absolute_error",
        )
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        results.append(
            {
                "model": name,
                "cv_mae_mean": float(-cv_scores.mean()),
                "cv_mae_std": float(cv_scores.std()),
                "test_mae": float(mean_absolute_error(y_test, predictions)),
                "test_rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
                "test_r2": float(r2_score(y_test, predictions)),
            }
        )

    # Choose the model using training-only cross-validation, not test scores.
    results.sort(key=lambda row: row["cv_mae_mean"])
    selected_name = results[0]["model"]
    selected_model = candidates[selected_name]
    selected_model.fit(x_train, y_train)
    selected_predictions = selected_model.predict(x_test)

    metrics = {
        "selected_model": selected_name,
        "test_rows": int(len(y_test)),
        "mae": float(mean_absolute_error(y_test, selected_predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, selected_predictions))),
        "r2": float(r2_score(y_test, selected_predictions)),
        "random_state": 42,
        "test_size": 0.2,
        "selection_method": "Lowest mean 5-fold CV MAE on training split",
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "excluded_feature": "Gender",
    }
    comparison = {
        "selection_method": "Lowest mean 5-fold cross-validation MAE on training split",
        "split": {"test_size": 0.2, "random_state": 42},
        "feature_columns": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "target": TARGET,
        "results": results,
        "selected_model": selected_name,
        "note": "Test metrics are reported for comparison; model selection uses training-only cross-validation.",
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    COMPARISON_PATH.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    print(f"Rows used after cleaning: {len(data)}")
    print(f"Training rows: {len(x_train)} | Test rows: {len(x_test)}")
    print("\nModel comparison (sorted by training CV MAE)")
    for row in results:
        print(
            f"{row['model']}: CV MAE={row['cv_mae_mean']:,.2f} "
            f"+/- {row['cv_mae_std']:,.2f}; test MAE={row['test_mae']:,.2f}; "
            f"RMSE={row['test_rmse']:,.2f}; R2={row['test_r2']:.4f}"
        )
    print(f"\nSelected model: {selected_name}")
    print(f"Saved model: {MODEL_PATH.relative_to(ROOT)}")
    print(f"Saved metrics: {METRICS_PATH.relative_to(ROOT)}")
    print(f"Saved comparison: {COMPARISON_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
