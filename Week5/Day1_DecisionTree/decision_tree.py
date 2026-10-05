"""Train a Decision Tree classifier and generate predictions."""

import csv
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
PREDICTIONS_FILE = BASE_DIR / "predictions.csv"
METRICS_FILE = BASE_DIR / "model_metrics.txt"
RANDOM_STATE = 42


def main() -> None:
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=iris.target,
    )

    model = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    with PREDICTIONS_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["test_row", "actual_species", "predicted_species"])
        for row_number, (actual, predicted) in enumerate(
            zip(y_test, predictions), start=1
        ):
            writer.writerow(
                [row_number, iris.target_names[actual], iris.target_names[predicted]]
            )

    metrics = (
        "Decision Tree Classifier\n"
        "========================\n"
        "Dataset: Iris\n"
        f"Training samples: {len(y_train)}\n"
        f"Testing samples: {len(y_test)}\n"
        "Maximum tree depth: 3\n"
        f"Test accuracy: {accuracy:.4f} ({accuracy:.2%})\n"
    )
    METRICS_FILE.write_text(metrics, encoding="utf-8")
    print(metrics)
    print(f"Saved predictions to {PREDICTIONS_FILE.name}")


if __name__ == "__main__":
    main()

