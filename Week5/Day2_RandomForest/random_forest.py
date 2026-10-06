"""Train and evaluate a Random Forest classifier."""

import csv
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
PREDICTIONS_FILE = BASE_DIR / "predictions.csv"
METRICS_FILE = BASE_DIR / "model_metrics.txt"
RANDOM_STATE = 42


def main() -> None:
    dataset = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=dataset.target,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    accuracy = accuracy_score(y_test, predictions)

    with PREDICTIONS_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["test_row", "actual_class", "predicted_class", "confidence"]
        )
        for row_number, (actual, predicted, probability) in enumerate(
            zip(y_test, predictions, probabilities), start=1
        ):
            writer.writerow(
                [
                    row_number,
                    dataset.target_names[actual],
                    dataset.target_names[predicted],
                    f"{probability[predicted]:.4f}",
                ]
            )

    metrics = (
        "Random Forest Classifier\n"
        "========================\n"
        "Dataset: Breast Cancer Wisconsin\n"
        f"Training samples: {len(y_train)}\n"
        f"Testing samples: {len(y_test)}\n"
        f"Number of trees: {model.n_estimators}\n"
        f"Test accuracy: {accuracy:.4f} ({accuracy:.2%})\n"
    )
    METRICS_FILE.write_text(metrics, encoding="utf-8")
    print(metrics)
    print(f"Saved predictions to {PREDICTIONS_FILE.name}")


if __name__ == "__main__":
    main()

