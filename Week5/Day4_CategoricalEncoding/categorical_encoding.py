"""Practice label encoding and one-hot encoding."""

import csv
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "Day4_PreprocessingRevision" / "student_success.csv"
OUTPUT_FILE = BASE_DIR / "encoded_categories.csv"
NOTES_FILE = BASE_DIR / "encoding_notes.txt"


def most_common(values: list[str]) -> str:
    return Counter(value for value in values if value).most_common(1)[0][0]


def main() -> None:
    with DATA_FILE.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    tutoring = [row["tutoring"] for row in rows]
    learning_modes = [row["learning_mode"] for row in rows]
    tutoring_fill = most_common(tutoring)
    learning_mode_fill = most_common(learning_modes)
    tutoring = [value or tutoring_fill for value in tutoring]
    learning_modes = [value or learning_mode_fill for value in learning_modes]

    label_encoder = LabelEncoder()
    tutoring_labels = label_encoder.fit_transform(tutoring)
    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    learning_mode_values = one_hot_encoder.fit_transform(
        np.array(learning_modes).reshape(-1, 1)
    )
    encoded_columns = [
        name.replace("x0_", "learning_mode_")
        for name in one_hot_encoder.get_feature_names_out()
    ]

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["original_tutoring", "tutoring_label", "original_learning_mode", *encoded_columns]
        )
        for tutoring_value, tutoring_label, mode, encoded_mode in zip(
            tutoring, tutoring_labels, learning_modes, learning_mode_values
        ):
            writer.writerow([tutoring_value, tutoring_label, mode, *(int(value) for value in encoded_mode)])

    label_mapping = {
        str(category): int(encoded_value)
        for category, encoded_value in zip(
            label_encoder.classes_, label_encoder.transform(label_encoder.classes_)
        )
    }
    notes = (
        "Categorical Encoding\n"
        "====================\n"
        f"Label encoding for tutoring: {label_mapping}\n"
        f"One-hot columns for learning mode: {', '.join(encoded_columns)}\n\n"
        "Label encoding is suitable here because tutoring has two values. One-hot "
        "encoding is used for learning mode because its categories have no natural order.\n"
    )
    NOTES_FILE.write_text(notes, encoding="utf-8")
    print(notes)
    print(f"Saved encoded values to {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
