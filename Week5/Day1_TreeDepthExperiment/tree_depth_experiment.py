"""Compare Decision Tree performance at different maximum depths."""

import csv
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BASE_DIR / "depth_results.csv"
NOTES_FILE = BASE_DIR / "depth_observations.txt"
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

    results = []
    for depth in range(1, 7):
        model = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
        model.fit(x_train, y_train)
        train_accuracy = accuracy_score(y_train, model.predict(x_train))
        test_accuracy = accuracy_score(y_test, model.predict(x_test))
        results.append((depth, train_accuracy, test_accuracy))

    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["max_depth", "training_accuracy", "testing_accuracy"])
        for depth, train_accuracy, test_accuracy in results:
            writer.writerow([depth, f"{train_accuracy:.4f}", f"{test_accuracy:.4f}"])

    best_depth, best_train, best_test = max(results, key=lambda row: (row[2], -row[0]))
    notes = (
        "Tree Depth Experiment\n"
        "=====================\n"
        f"Best tested depth: {best_depth}\n"
        f"Training accuracy at best depth: {best_train:.2%}\n"
        f"Testing accuracy at best depth: {best_test:.2%}\n\n"
        "Observation: A shallow tree may underfit because it cannot learn enough "
        "rules. A deeper tree can fit the training data more closely, but extra "
        "depth does not always improve unseen test data and may cause overfitting.\n"
    )
    NOTES_FILE.write_text(notes, encoding="utf-8")

    print("Depth | Training accuracy | Testing accuracy")
    for depth, train_accuracy, test_accuracy in results:
        print(f"{depth:>5} | {train_accuracy:>17.2%} | {test_accuracy:>16.2%}")
    print(f"\nSaved results to {RESULTS_FILE.name}")
    print(f"Saved observations to {NOTES_FILE.name}")


if __name__ == "__main__":
    main()

