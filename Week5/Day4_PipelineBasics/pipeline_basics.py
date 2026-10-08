"""Combine preprocessing and logistic regression in one pipeline."""

import csv
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "Day4_PreprocessingRevision" / "student_success.csv"
OUTPUT_FILE = BASE_DIR / "pipeline_result.txt"
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


def main() -> None:
    features, target = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, random_state=RANDOM_STATE, stratify=target
    )
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(missing_values=None, strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        [("numeric", numeric_pipeline, [0, 1, 2]), ("category", categorical_pipeline, [3, 4])]
    )
    model = Pipeline(
        [
            ("preprocessing", preprocessing),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    model.fit(x_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(x_test))

    result = (
        "Pipeline Basics\n"
        "===============\n"
        "Step 1: Fill missing numerical and categorical values.\n"
        "Step 2: Scale numerical features and one-hot encode categories.\n"
        "Step 3: Train Logistic Regression on the processed features.\n"
        f"Test accuracy: {accuracy:.4f} ({accuracy:.2%})\n\n"
        "The pipeline applies the same learned preprocessing to training data, test data, and future predictions.\n"
    )
    OUTPUT_FILE.write_text(result, encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
