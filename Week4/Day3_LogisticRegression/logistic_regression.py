"""Train a logistic regression model on a binary classification dataset."""

import csv
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
PREDICTIONS_FILE = BASE_DIR / "predictions.csv"
SUMMARY_FILE = BASE_DIR / "model_summary.txt"
RANDOM_STATE = 42


def main() -> None:
    dataset = load_breast_cancer()
    features_train, features_test, target_train, target_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=dataset.target,
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    )
    model.fit(features_train, target_train)
    predictions = model.predict(features_test)
    probabilities = model.predict_proba(features_test)

    with PREDICTIONS_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["test_row", "actual_class", "predicted_class", "prediction_confidence"]
        )
        for row_number, (actual, predicted, probability) in enumerate(
            zip(target_test, predictions, probabilities), start=1
        ):
            writer.writerow(
                [
                    row_number,
                    dataset.target_names[actual],
                    dataset.target_names[predicted],
                    f"{probability[predicted]:.4f}",
                ]
            )

    summary = (
        "Logistic Regression Model Summary\n"
        "=================================\n"
        f"Dataset: {dataset.DESCR.splitlines()[0]}\n"
        f"Target classes: {', '.join(dataset.target_names)}\n"
        f"Number of features: {dataset.data.shape[1]}\n"
        f"Training samples: {len(target_train)}\n"
        f"Testing samples: {len(target_test)}\n"
        "Preprocessing: StandardScaler\n"
        "Model: LogisticRegression(max_iter=1000, random_state=42)\n"
    )
    SUMMARY_FILE.write_text(summary, encoding="utf-8")

    print(summary)
    print(f"Saved {len(predictions)} predictions to {PREDICTIONS_FILE.name}")


if __name__ == "__main__":
    main()

