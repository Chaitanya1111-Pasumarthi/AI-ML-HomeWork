"""Calculate classification accuracy for logistic regression predictions."""

from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "accuracy.txt"
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

    accuracy = accuracy_score(target_test, predictions)
    correct = int((target_test == predictions).sum())
    total = len(target_test)
    report = (
        "Classification Accuracy\n"
        "=======================\n"
        f"Correct predictions: {correct}\n"
        f"Total predictions: {total}\n"
        f"Accuracy: {accuracy:.4f} ({accuracy:.2%})\n\n"
        "Accuracy is the proportion of all test samples that the model "
        "classified correctly.\n"
    )
    OUTPUT_FILE.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved accuracy results to {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()

