"""Test preprocessing with Decision Tree and Random Forest classifiers."""

import csv
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BASE_DIR / "test_results.csv"
SUMMARY_FILE = BASE_DIR / "test_summary.txt"
RANDOM_STATE = 42


def main() -> None:
    features, target = load_breast_cancer(return_X_y=True)
    features = features.copy()
    rng = np.random.default_rng(RANDOM_STATE)
    missing_mask = rng.random(features.shape) < 0.02
    features[missing_mask] = np.nan

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, random_state=RANDOM_STATE, n_jobs=1
        ),
    }
    results = []
    for name, classifier in models.items():
        model = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("classifier", classifier)]
        )
        model.fit(x_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(x_test))
        results.append((name, accuracy))

    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "test_accuracy"])
        for name, accuracy in results:
            writer.writerow([name, f"{accuracy:.4f}"])

    best_name, best_accuracy = max(results, key=lambda result: result[1])
    summary = (
        "Week 6 Day 1 Test\n"
        "=================\n"
        f"Missing feature values inserted: {int(missing_mask.sum())}\n"
        "Preprocessing: median imputation fitted on training data.\n"
        f"Decision Tree accuracy: {results[0][1]:.2%}\n"
        f"Random Forest accuracy: {results[1][1]:.2%}\n"
        f"Better test result: {best_name} ({best_accuracy:.2%})\n"
    )
    SUMMARY_FILE.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Saved results to {RESULTS_FILE.name}")


if __name__ == "__main__":
    main()

