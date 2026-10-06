"""Compare Decision Tree and Random Forest classification results."""

import csv
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BASE_DIR / "model_comparison.csv"
NOTES_FILE = BASE_DIR / "comparison_notes.txt"
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
        predictions = model.predict(x_test)
        results.append(
            (
                name,
                accuracy_score(y_test, predictions),
                f1_score(y_test, predictions, average="weighted"),
            )
        )

    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "test_accuracy", "weighted_f1_score"])
        for name, accuracy, f1 in results:
            writer.writerow([name, f"{accuracy:.4f}", f"{f1:.4f}"])

    best_name, best_accuracy, best_f1 = max(results, key=lambda result: result[1])
    notes = (
        "Decision Tree vs. Random Forest\n"
        "===============================\n"
        f"Better test result: {best_name}\n"
        f"Best test accuracy: {best_accuracy:.2%}\n"
        f"Best weighted F1-score: {best_f1:.4f}\n\n"
        "A Decision Tree is simple and easy to interpret, but it can fit noise in "
        "the training data. A Random Forest averages predictions from many trees, "
        "which usually improves stability and reduces overfitting.\n"
    )
    NOTES_FILE.write_text(notes, encoding="utf-8")

    print("Model comparison:")
    for name, accuracy, f1 in results:
        print(f"  {name}: accuracy={accuracy:.2%}, weighted F1={f1:.4f}")
    print(f"\nSaved comparison to {RESULTS_FILE.name}")
    print(f"Saved explanation to {NOTES_FILE.name}")


if __name__ == "__main__":
    main()

