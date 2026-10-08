"""Run a complete workflow from raw CSV data through model evaluation."""

import csv
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "Day4_PreprocessingRevision" / "student_success.csv"
PREDICTIONS_FILE = BASE_DIR / "predictions.csv"
REPORT_FILE = BASE_DIR / "evaluation_report.txt"
RANDOM_STATE = 42


def load_data() -> tuple[np.ndarray, np.ndarray]:
    features, target = [], []
    with DATA_FILE.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            features.append(
                [
                    float(row["study_hours"]) if row["study_hours"] else np.nan,
                    float(row["attendance"]) if row["attendance"] else np.nan,
                    float(row["previous_score"]) if row["previous_score"] else np.nan,
                    row["learning_mode"] or None,
                    row["tutoring"] or None,
                ]
            )
            target.append(int(row["passed"]))
    return np.array(features, dtype=object), np.array(target)


def build_pipeline() -> Pipeline:
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                [0, 1, 2],
            ),
            (
                "category",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(missing_values=None, strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                [3, 4],
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )


def main() -> None:
    features, target = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, random_state=RANDOM_STATE, stratify=target
    )
    model = build_pipeline()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)

    with PREDICTIONS_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["test_row", "actual", "predicted", "confidence"])
        for row_number, (actual, predicted, probability) in enumerate(
            zip(y_test, predictions, probabilities), start=1
        ):
            writer.writerow([row_number, actual, predicted, f"{probability[predicted]:.4f}"])

    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)
    metrics = classification_report(
        y_test,
        predictions,
        target_names=["not passed", "passed"],
        digits=4,
        zero_division=0,
    )
    report = (
        "End-to-End Student Success Classification\n"
        "=========================================\n"
        f"Raw rows loaded: {len(target)}\n"
        f"Training rows: {len(y_train)}\n"
        f"Testing rows: {len(y_test)}\n"
        f"Accuracy: {accuracy:.4f} ({accuracy:.2%})\n"
        f"Confusion matrix: {matrix.tolist()}\n\n"
        "Classification report\n"
        "---------------------\n"
        f"{metrics}\n"
        "Workflow: load raw CSV -> split data -> impute -> encode -> scale -> train -> predict -> evaluate.\n"
    )
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved predictions to {PREDICTIONS_FILE.name}")


if __name__ == "__main__":
    main()
