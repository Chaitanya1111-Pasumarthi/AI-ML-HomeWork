"""Generate and explain a confusion matrix for binary classification."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
MATRIX_FILE = BASE_DIR / "confusion_matrix.csv"
PLOT_FILE = BASE_DIR / "confusion_matrix.png"
NOTES_FILE = BASE_DIR / "confusion_matrix_notes.txt"
RANDOM_STATE = 42


def main() -> None:
    dataset = load_breast_cancer()
    features_train, features_test, target_train, target_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=dataset.target,
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    )
    model.fit(features_train, target_train)
    predictions = model.predict(features_test)
    matrix = confusion_matrix(target_test, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()

    with MATRIX_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual / predicted", "malignant", "benign"])
        writer.writerow(["malignant", matrix[0, 0], matrix[0, 1]])
        writer.writerow(["benign", matrix[1, 0], matrix[1, 1]])

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=dataset.target_names,
    )
    display.plot(cmap="Blues", values_format="d", colorbar=False)
    display.ax_.set_title("Logistic Regression Confusion Matrix")
    display.figure_.tight_layout()
    display.figure_.savefig(PLOT_FILE, dpi=160)
    plt.close(display.figure_)

    notes = f"""Confusion Matrix Explanation
============================
Positive class used here: benign (class 1)

True Negatives (TN): {true_negative}
Malignant cases correctly predicted as malignant.

False Positives (FP): {false_positive}
Malignant cases incorrectly predicted as benign.

False Negatives (FN): {false_negative}
Benign cases incorrectly predicted as malignant.

True Positives (TP): {true_positive}
Benign cases correctly predicted as benign.
"""
    NOTES_FILE.write_text(notes, encoding="utf-8")

    print(notes)
    print(f"Saved matrix data to {MATRIX_FILE.name}")
    print(f"Saved matrix plot to {PLOT_FILE.name}")


if __name__ == "__main__":
    main()

