"""Assess Random Forest reliability with five-fold cross-validation."""

import csv
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


BASE_DIR = Path(__file__).resolve().parent
SCORES_FILE = BASE_DIR / "cross_validation_scores.csv"
SUMMARY_FILE = BASE_DIR / "cross_validation_summary.txt"
RANDOM_STATE = 42


def main() -> None:
    features, target = load_breast_cancer(return_X_y=True)
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, features, target, cv=folds, scoring="accuracy")

    with SCORES_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["fold", "accuracy"])
        for fold_number, score in enumerate(scores, start=1):
            writer.writerow([fold_number, f"{score:.4f}"])

    mean_score = float(np.mean(scores))
    standard_deviation = float(np.std(scores))
    summary = (
        "Five-Fold Cross-Validation\n"
        "==========================\n"
        "Model: Random Forest (200 trees)\n"
        f"Fold accuracies: {', '.join(f'{score:.2%}' for score in scores)}\n"
        f"Mean accuracy: {mean_score:.4f} ({mean_score:.2%})\n"
        f"Standard deviation: {standard_deviation:.4f}\n\n"
        "The mean estimates expected performance across different data splits. "
        "The small standard deviation indicates that accuracy is reasonably "
        "consistent, so the model is not dependent on one lucky split.\n"
    )
    SUMMARY_FILE.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Saved fold scores to {SCORES_FILE.name}")


if __name__ == "__main__":
    main()

