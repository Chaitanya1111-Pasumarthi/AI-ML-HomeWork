"""Review missing values, encoding, scaling, and train/test splitting."""

import csv
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "student_success.csv"
PREVIEW_FILE = BASE_DIR / "processed_preview.csv"
SUMMARY_FILE = BASE_DIR / "preprocessing_summary.txt"
RANDOM_STATE = 42


def load_data() -> tuple[np.ndarray, np.ndarray]:
    rows, targets = [], []
    with DATA_FILE.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            rows.append(
                [
                    float(row["study_hours"]) if row["study_hours"] else np.nan,
                    float(row["attendance"]) if row["attendance"] else np.nan,
                    float(row["previous_score"]) if row["previous_score"] else np.nan,
                    row["learning_mode"] or None,
                    row["tutoring"] or None,
                ]
            )
            targets.append(int(row["passed"]))
    return np.array(rows, dtype=object), np.array(targets)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(missing_values=None, strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipeline, [0, 1, 2]), ("category", categorical_pipeline, [3, 4])]
    )


def main() -> None:
    features, target = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    preprocessor = build_preprocessor()
    transformed_train = preprocessor.fit_transform(x_train)
    transformed_test = preprocessor.transform(x_test)
    feature_names = preprocessor.get_feature_names_out(
        ["study_hours", "attendance", "previous_score", "learning_mode", "tutoring"]
    )

    with PREVIEW_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([*feature_names, "passed"])
        for row, label in zip(transformed_train[:5], y_train[:5]):
            writer.writerow([*(f"{value:.4f}" for value in row), label])

    missing_values = int(sum(value is None or (isinstance(value, float) and np.isnan(value)) for row in features for value in row))
    summary = (
        "Preprocessing Revision\n"
        "======================\n"
        f"Raw rows: {len(target)}\n"
        f"Missing feature values found: {missing_values}\n"
        f"Training rows: {len(y_train)}\n"
        f"Testing rows: {len(y_test)}\n"
        f"Features after encoding: {transformed_train.shape[1]}\n\n"
        "Numerical values were filled with the training median and standardized.\n"
        "Categorical values were filled with the most frequent training value and one-hot encoded.\n"
        "The preprocessor was fitted only on training data to prevent data leakage.\n"
    )
    SUMMARY_FILE.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Transformed test shape: {transformed_test.shape}")
    print(f"Saved preview to {PREVIEW_FILE.name}")


if __name__ == "__main__":
    main()
