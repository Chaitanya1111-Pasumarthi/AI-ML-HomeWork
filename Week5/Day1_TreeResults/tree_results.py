"""Extract and visualize Decision Tree feature importance."""

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
VALUES_FILE = BASE_DIR / "feature_importance.csv"
PLOT_FILE = BASE_DIR / "feature_importance.png"
RANDOM_STATE = 42


def create_plot(feature_names: list[str], importances: list[float]) -> None:
    """Save a horizontal feature-importance bar chart as a PNG."""
    image = Image.new("RGB", (1100, 650), "white")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default(size=34)
    label_font = ImageFont.load_default(size=23)
    value_font = ImageFont.load_default(size=21)

    draw.text(
        (550, 50),
        "Decision Tree Feature Importance",
        fill="#111827",
        font=title_font,
        anchor="mm",
    )

    bar_left = 365
    bar_width = 600
    maximum = max(importances) if importances else 1
    for index, (name, importance) in enumerate(zip(feature_names, importances)):
        y = 135 + index * 115
        draw.text((335, y + 28), name, fill="#111827", font=label_font, anchor="rm")
        draw.rectangle(
            (bar_left, y, bar_left + bar_width, y + 55),
            fill="#e2e8f0",
            outline="#64748b",
            width=2,
        )
        filled_width = int(bar_width * importance / maximum) if maximum else 0
        if filled_width:
            draw.rectangle(
                (bar_left, y, bar_left + filled_width, y + 55),
                fill="#2563eb",
            )
        draw.text(
            (bar_left + filled_width + 12, y + 28),
            f"{importance:.3f}",
            fill="#111827",
            font=value_font,
            anchor="lm",
        )

    draw.text(
        (665, 610),
        "Higher values indicate greater influence on the model's decisions",
        fill="#475569",
        font=value_font,
        anchor="mm",
    )
    image.save(PLOT_FILE)


def main() -> None:
    iris = load_iris()
    x_train, _, y_train, _ = train_test_split(
        iris.data,
        iris.target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=iris.target,
    )

    model = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
    model.fit(x_train, y_train)

    importance_pairs = sorted(
        zip(iris.feature_names, model.feature_importances_),
        key=lambda pair: pair[1],
        reverse=True,
    )
    feature_names = [name for name, _ in importance_pairs]
    importances = [float(value) for _, value in importance_pairs]

    with VALUES_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["feature", "importance"])
        for name, importance in importance_pairs:
            writer.writerow([name, f"{importance:.6f}"])

    create_plot(feature_names, importances)

    print("Feature importance:")
    for name, importance in importance_pairs:
        print(f"  {name}: {importance:.4f}")
    print(f"\nSaved values to {VALUES_FILE.name}")
    print(f"Saved visualization to {PLOT_FILE.name}")


if __name__ == "__main__":
    main()

