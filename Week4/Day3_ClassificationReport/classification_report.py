"""Evaluate logistic regression with precision, recall, and F1-score."""

from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "classification_report.txt"
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

    metrics = classification_report(
        target_test,
        predictions,
        target_names=dataset.target_names,
        digits=4,
        zero_division=0,
    )
    explanation = (
        "\nMetric meanings\n"
        "---------------\n"
        "Precision: Of the samples predicted as a class, how many were correct.\n"
        "Recall: Of the actual samples in a class, how many the model found.\n"
        "F1-score: The harmonic mean of precision and recall.\n"
        "Support: The number of actual test samples belonging to each class.\n"
    )
    report = "Classification Report\n=====================\n" + metrics + explanation
    OUTPUT_FILE.write_text(report, encoding="utf-8")

    print(report)
    print(f"Saved classification report to {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()

