"""Tune a Decision Tree classifier with GridSearchCV."""

import csv
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BASE_DIR / "grid_search_results.csv"
SUMMARY_FILE = BASE_DIR / "tuning_summary.txt"
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

    parameter_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [2, 3, 4, 5, None],
        "min_samples_split": [2, 5, 10],
    }
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        parameter_grid,
        cv=folds,
        scoring="accuracy",
        n_jobs=1,
        return_train_score=True,
    )
    search.fit(x_train, y_train)

    rows = sorted(
        zip(
            search.cv_results_["rank_test_score"],
            search.cv_results_["params"],
            search.cv_results_["mean_train_score"],
            search.cv_results_["mean_test_score"],
            search.cv_results_["std_test_score"],
        ),
        key=lambda row: row[0],
    )
    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "rank",
                "criterion",
                "max_depth",
                "min_samples_split",
                "mean_training_accuracy",
                "mean_validation_accuracy",
                "validation_std_dev",
            ]
        )
        for rank, parameters, train_score, validation_score, std_dev in rows:
            writer.writerow(
                [
                    rank,
                    parameters["criterion"],
                    parameters["max_depth"],
                    parameters["min_samples_split"],
                    f"{train_score:.4f}",
                    f"{validation_score:.4f}",
                    f"{std_dev:.4f}",
                ]
            )

    test_accuracy = search.score(x_test, y_test)
    summary = (
        "GridSearchCV Summary\n"
        "====================\n"
        f"Parameter combinations tested: {len(rows)}\n"
        "Cross-validation folds: 5\n"
        f"Best parameters: {search.best_params_}\n"
        f"Best validation accuracy: {search.best_score_:.4f} "
        f"({search.best_score_:.2%})\n"
        f"Held-out test accuracy: {test_accuracy:.4f} ({test_accuracy:.2%})\n"
    )
    SUMMARY_FILE.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Saved all parameter results to {RESULTS_FILE.name}")


if __name__ == "__main__":
    main()

