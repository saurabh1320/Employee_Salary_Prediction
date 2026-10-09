"""Basic checks for data loading and model pipelines."""
import pandas as pd

from src.train import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_candidates,
    build_pipeline,
    load_data,
)


def test_dataset_loads_and_has_required_columns():
    data = load_data()
    expected = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["Salary"])
    assert expected.issubset(data.columns)
    assert len(data) > 0
    assert data["Salary"].notna().all()


def test_pipeline_fits_and_predicts_with_unseen_category():
    data = load_data()
    features = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    target = data["Salary"]
    model = build_pipeline()
    model.fit(features.iloc[:100], target.iloc[:100])
    sample = pd.DataFrame([{
        "Age": 31,
        "Years of Experience": 6,
        "Education Level": "Bachelor's",
        "Job Title": "Previously Unseen Job Title",
    }])
    prediction = model.predict(sample)
    assert len(prediction) == 1
    assert pd.notna(prediction[0])


def test_candidate_models_are_available():
    candidates = build_candidates()
    assert "DummyRegressor (median)" in candidates
    assert "Linear Regression" in candidates
    assert "Random Forest" in candidates
    assert "Extra Trees" in candidates
