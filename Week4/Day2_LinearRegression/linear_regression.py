"""Train a simple linear regression model and generate predictions."""

import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "student_scores.csv"
OUTPUT_FILE = BASE_DIR / "predictions.csv"


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load study hours as the feature and exam score as the target."""
    hours = []
    scores = []
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

    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["hours_studied", "actual_score", "predicted_score"])
        for hours, actual, predicted in sorted(
            zip(x_test.ravel(), y_test, predictions)
        ):
            writer.writerow([f"{hours:.1f}", f"{actual:.1f}", f"{predicted:.2f}"])

    print(f"Training samples: {len(x_train)}")
    print(f"Testing samples: {len(x_test)}")
    print(f"Model equation: score = {model.intercept_:.2f} + "
          f"({model.coef_[0]:.2f} x hours)")
    print("\nTest predictions:")
    for hours, actual, predicted in sorted(zip(x_test.ravel(), y_test, predictions)):
        print(
            f"  Hours: {hours:>3.1f} | Actual: {actual:>5.1f} | "
            f"Predicted: {predicted:>5.2f}"
        )
    print(f"\nSaved predictions to {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()

