"""Make a prediction from raw employee feature values."""
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "salary_model.joblib"


def predict_salary(employee: dict, model_path: Path = MODEL_PATH) -> float:
    """Predict salary from raw, named input features."""
    if not model_path.exists():
        raise FileNotFoundError(
            "Model not found. Train it first with: python -m src.train"
        )
    model = joblib.load(model_path)
    frame = pd.DataFrame([employee])
    prediction = model.predict(frame)
    return float(prediction[0])


if __name__ == "__main__":
    example_employee = {
        "Age": 30,
        "Years of Experience": 5,
        "Education Level": "Bachelor's",
        "Job Title": "Software Engineer",
    }
    result = predict_salary(example_employee)
    print(f"Predicted salary: {result:,.2f} salary units")
    print("Note: This is an educational model estimate, not compensation advice.")
