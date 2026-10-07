"""Create a comparison sheet for several classification models."""

import csv
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
SHEET_FILE = BASE_DIR / "accuracy_comparison.csv"
SUMMARY_FILE = BASE_DIR / "accuracy_comparison_summary.txt"
RANDOM_STATE = 42


def main() -> None:
    features, target = load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    tuning = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {
            "criterion": ["gini", "entropy"],
            "max_depth": [2, 3, 4, 5, None],
            "min_samples_split": [2, 5, 10],
        },
        cv=folds,
        scoring="accuracy",
        n_jobs=1,
    )
    tuning.fit(x_train, y_train)

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Tuned Decision Tree": tuning.best_estimator_,
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }

    results = []
    for name, model in models.items():
        validation_scores = cross_val_score(
            model, x_train, y_train, cv=folds, scoring="accuracy"
        )
        model.fit(x_train, y_train)
        test_accuracy = accuracy_score(y_test, model.predict(x_test))
        results.append(
            {
                "model": name,
                "validation_mean": float(np.mean(validation_scores)),
                "validation_std": float(np.std(validation_scores)),
                "test_accuracy": test_accuracy,
            }
        )

    results.sort(key=lambda result: result["test_accuracy"], reverse=True)
    with SHEET_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "rank",
                "model",
                "mean_cv_accuracy",
                "cv_standard_deviation",
                "test_accuracy",
            ]
        )
        for rank, result in enumerate(results, start=1):
            writer.writerow(
                [
                    rank,
                    result["model"],
                    f"{result['validation_mean']:.4f}",
                    f"{result['validation_std']:.4f}",
                    f"{result['test_accuracy']:.4f}",
                ]
            )

    best = results[0]
    summary = (
        "Model Accuracy Comparison\n"
        "=========================\n"
        f"Best held-out test accuracy: {best['model']} "
        f"({best['test_accuracy']:.2%})\n"
        f"Its mean cross-validation accuracy was {best['validation_mean']:.2%} "
        f"with a standard deviation of {best['validation_std']:.4f}.\n\n"
        "The comparison uses identical training data, test data, and validation "
        "folds for every model, making the scores directly comparable.\n"
    )
    SUMMARY_FILE.write_text(summary, encoding="utf-8")

    print("Model accuracy comparison:")
    for rank, result in enumerate(results, start=1):
        print(
            f"  {rank}. {result['model']}: CV={result['validation_mean']:.2%}, "
            f"test={result['test_accuracy']:.2%}"
        )
    print(f"\nSaved comparison sheet to {SHEET_FILE.name}")
    print(f"Saved summary to {SUMMARY_FILE.name}")


if __name__ == "__main__":
    main()

