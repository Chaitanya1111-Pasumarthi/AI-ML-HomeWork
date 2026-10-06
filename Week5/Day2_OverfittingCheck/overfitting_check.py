"""Compare training and testing accuracy to identify overfitting."""

import csv
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BASE_DIR / "accuracy_comparison.csv"
NOTES_FILE = BASE_DIR / "overfitting_observations.txt"
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

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=1
        ),
    }
    results = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        train_accuracy = accuracy_score(y_train, model.predict(x_train))
        test_accuracy = accuracy_score(y_test, model.predict(x_test))
        gap = train_accuracy - test_accuracy
        results.append((name, train_accuracy, test_accuracy, gap))

    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["model", "training_accuracy", "testing_accuracy", "accuracy_gap"]
        )
        for name, train_accuracy, test_accuracy, gap in results:
            writer.writerow(
                [name, f"{train_accuracy:.4f}", f"{test_accuracy:.4f}", f"{gap:.4f}"]
            )

    result_lines = [
        f"{name}: training={train:.2%}, testing={test:.2%}, gap={gap:.2%}"
        for name, train, test, gap in results
    ]
    smaller_gap_model = min(results, key=lambda result: result[3])[0]
    notes = (
        "Overfitting Check\n"
        "=================\n"
        + "\n".join(result_lines)
        + "\n\n"
        + "A large positive gap means the model performs much better on training "
        "data than unseen test data, which is a sign of overfitting. "
        f"In this experiment, {smaller_gap_model} has the smaller accuracy gap.\n"
    )
    NOTES_FILE.write_text(notes, encoding="utf-8")

    print(notes)
    print(f"Saved accuracy comparison to {RESULTS_FILE.name}")


if __name__ == "__main__":
    main()

