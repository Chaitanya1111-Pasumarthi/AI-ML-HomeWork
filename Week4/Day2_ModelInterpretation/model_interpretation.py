"""Train and explain a simple linear regression model."""

import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "student_scores.csv"
OUTPUT_FILE = BASE_DIR / "interpretation.txt"


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

    coefficient = model.coef_[0]
    intercept = model.intercept_
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r_squared = r2_score(y_test, predictions)
    five_hour_prediction = model.predict(np.array([[5.0]]))[0]

    report = f"""Linear Regression Interpretation
================================

Model equation
---------------
Predicted exam score = {intercept:.2f} + ({coefficient:.2f} x hours studied)

Coefficient
-----------
The coefficient is {coefficient:.2f}. Within the range represented by this
dataset, one additional hour of study is associated with an increase of about
{coefficient:.2f} points in the predicted exam score.

Intercept
---------
The intercept is {intercept:.2f}. It is the model's predicted score when study
hours equal zero. Because zero hours is outside the observed data range, this
value should be interpreted cautiously.

Example prediction
------------------
For 5.0 hours of study, the model predicts a score of {five_hour_prediction:.2f}.

Test-set behavior
-----------------
MAE: {mae:.4f}
MSE: {mse:.4f}
R-squared: {r_squared:.4f}

The small MAE and MSE indicate that predictions are close to the actual test
scores. The R-squared value shows how much of the test-score variation is
explained by study hours in this test split. The positive coefficient confirms
a positive linear relationship. This is an educational example and shows
association, not proof that additional study time alone causes a particular
score.
"""
    OUTPUT_FILE.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved interpretation to {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()

