"""Find and explain the best Decision Tree parameters."""

from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "best_parameters_analysis.txt"
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
    search = GridSearchCV(
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
    search.fit(x_train, y_train)

    parameters = search.best_params_
    test_accuracy = search.score(x_test, y_test)
    depth_explanation = (
        "Limiting max_depth controls how many levels the tree may create, which "
        "can prevent it from memorizing training noise."
        if parameters["max_depth"] is not None
        else "No max_depth limit performed best during cross-validation."
    )
    report = f"""Best Parameters Analysis
========================
Criterion: {parameters['criterion']}
Maximum depth: {parameters['max_depth']}
Minimum samples required to split: {parameters['min_samples_split']}
Cross-validation accuracy: {search.best_score_:.2%}
Held-out test accuracy: {test_accuracy:.2%}

Why these parameters may help
-----------------------------
The {parameters['criterion']} criterion was the tested method that produced the
best average validation accuracy when choosing splits. {depth_explanation}
Requiring at least {parameters['min_samples_split']} samples before a node can
split also controls tree complexity. GridSearchCV selected this combination
using validation folds, not the held-out test set.
"""
    OUTPUT_FILE.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved analysis to {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()

