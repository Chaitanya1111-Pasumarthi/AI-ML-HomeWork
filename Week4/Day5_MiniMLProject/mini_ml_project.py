"""End-to-end Iris flower classification mini project."""

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "iris_dataset.csv"
PREDICTIONS_FILE = BASE_DIR / "predictions.csv"
REPORT_FILE = BASE_DIR / "evaluation_report.txt"
PLOT_FILE = BASE_DIR / "confusion_matrix.png"
RANDOM_STATE = 42


def save_confusion_matrix_plot(matrix: np.ndarray, class_names: np.ndarray) -> None:
    """Create a portable confusion-matrix image without a GUI backend."""
    image = Image.new("RGB", (1000, 760), "white")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default(size=34)
    label_font = ImageFont.load_default(size=25)
    value_font = ImageFont.load_default(size=38)
    small_font = ImageFont.load_default(size=21)

    draw.text(
        (500, 45),
        "Iris Logistic Regression Confusion Matrix",
        fill="#111827",
        font=title_font,
        anchor="mm",
    )
    draw.text((570, 710), "Predicted species", fill="#111827", font=label_font, anchor="mm")
    draw.multiline_text(
        (70, 400),
        "Actual\nspecies",
        fill="#111827",
        font=label_font,
        anchor="mm",
        align="center",
        spacing=6,
    )

    left, top, cell_size = 270, 100, 200
    maximum = int(matrix.max())
    for index, name in enumerate(class_names):
        center = left + index * cell_size + cell_size // 2
        draw.text((center, 75), str(name), fill="#111827", font=small_font, anchor="mm")
        row_center = top + index * cell_size + cell_size // 2
        draw.text((245, row_center), str(name), fill="#111827", font=small_font, anchor="rm")

    for row in range(3):
        for column in range(3):
            value = int(matrix[row, column])
            intensity = value / maximum if maximum else 0
            shade = int(245 - 155 * intensity)
            fill = (shade, min(250, shade + 25), 255)
            x0 = left + column * cell_size
            y0 = top + row * cell_size
            draw.rectangle(
                (x0, y0, x0 + cell_size, y0 + cell_size),
                fill=fill,
                outline="#334155",
                width=2,
            )
            text_color = "white" if intensity > 0.55 else "#0f172a"
            draw.text(
                (x0 + cell_size // 2, y0 + cell_size // 2),
                str(value),
                fill=text_color,
                font=value_font,
                anchor="mm",
            )

    image.save(PLOT_FILE)


def export_dataset(path: Path) -> None:
    """Export scikit-learn's Iris dataset so the project includes its data."""
    iris = load_iris()
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([*iris.feature_names, "target", "species"])
        for features, target in zip(iris.data, iris.target):
            writer.writerow([*features, int(target), iris.target_names[target]])


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load numeric features, target labels, and feature names from CSV."""
    features = []
    targets = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            features.append([float(value) for value in row[:4]])
            targets.append(int(row[4]))
    return np.array(features), np.array(targets), header[:4]


def run_project() -> dict[str, float]:
    """Run data preparation, training, prediction, and evaluation."""
    export_dataset(DATA_FILE)
    features, target, feature_names = load_dataset(DATA_FILE)
    class_names = load_iris().target_names

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)

    with PREDICTIONS_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["test_row", "actual_species", "predicted_species", "confidence"]
        )
        for row_number, (actual, predicted, probability) in enumerate(
            zip(y_test, predictions, probabilities), start=1
        ):
            writer.writerow(
                [
                    row_number,
                    class_names[actual],
                    class_names[predicted],
                    f"{probability[predicted]:.4f}",
                ]
            )

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    report_text = (
        "Iris Classification Mini Project\n"
        "================================\n"
        "Problem: Predict Iris species from four flower measurements.\n"
        f"Dataset rows: {len(target)}\n"
        f"Features: {', '.join(feature_names)}\n"
        f"Training rows: {len(y_train)}\n"
        f"Testing rows: {len(y_test)}\n"
        "Preprocessing: StandardScaler fitted on training data only.\n"
        "Model: Multiclass Logistic Regression.\n"
        f"Test accuracy: {accuracy:.4f} ({accuracy:.2%})\n\n"
        "Classification report\n"
        "---------------------\n"
        f"{report}\n"
        "Conclusion\n"
        "----------\n"
        "The model separates the three Iris species well on the held-out test set. "
        "Setosa is easiest to distinguish, while any mistakes are most likely "
        "between versicolor and virginica because their measurements overlap.\n"
    )
    REPORT_FILE.write_text(report_text, encoding="utf-8")

    matrix = confusion_matrix(y_test, predictions)
    save_confusion_matrix_plot(matrix, class_names)

    return {"accuracy": accuracy, "training_rows": len(y_train), "testing_rows": len(y_test)}


def main() -> None:
    results = run_project()
    print("Iris classification project completed successfully.")
    print(f"Training rows: {results['training_rows']}")
    print(f"Testing rows: {results['testing_rows']}")
    print(f"Test accuracy: {results['accuracy']:.2%}")
    print(f"Saved report: {REPORT_FILE.name}")
    print(f"Saved predictions: {PREDICTIONS_FILE.name}")
    print(f"Saved plot: {PLOT_FILE.name}")


if __name__ == "__main__":
    main()
