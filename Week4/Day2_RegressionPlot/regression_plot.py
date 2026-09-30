"""Plot actual versus predicted values for a linear regression model."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "student_scores.csv"
OUTPUT_FILE = BASE_DIR / "actual_vs_predicted.png"


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

    lower = min(y_test.min(), predictions.min()) - 2
    upper = max(y_test.max(), predictions.max()) + 2

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, color="#2563eb", s=75, label="Test samples")
    plt.plot([lower, upper], [lower, upper], "--", color="#dc2626", label="Perfect prediction")
    plt.xlabel("Actual exam score")
    plt.ylabel("Predicted exam score")
    plt.title("Actual vs. Predicted Exam Scores")
    plt.xlim(lower, upper)
    plt.ylim(lower, upper)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=160)
    plt.close()

    print(f"Created plot: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

