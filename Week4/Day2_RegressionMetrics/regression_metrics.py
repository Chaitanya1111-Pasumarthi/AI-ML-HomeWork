"""Evaluate regression predictions with MAE and MSE."""

import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "student_scores.csv"
OUTPUT_FILE = BASE_DIR / "metrics.txt"


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    hours, scores = [], []
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            hours.append(float(row["hours_studied"]))
            scores.append(float(row["exam_score"]))
    return np.array(hours).reshape(-1, 1), np.array(scores)


def main() -> None:
    features, target = load_data(DATA_FILE)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, random_state=42
    )
    model = LinearRegression().fit(x_train, y_train)
    predictions = model.predict(x_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    report = (
        "Regression Evaluation\n"
        "=====================\n"
        f"Test samples: {len(y_test)}\n"
        f"Mean Absolute Error (MAE): {mae:.4f}\n"
        f"Mean Squared Error (MSE): {mse:.4f}\n\n"
        "MAE is the average absolute difference between actual and predicted "
        "scores.\nMSE is the average squared difference, so it penalizes larger "
        "errors more strongly.\n"
    )
    OUTPUT_FILE.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved metrics to {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()

