"""Extract and visualize important Random Forest features."""

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
VALUES_FILE = BASE_DIR / "feature_importance.csv"
PLOT_FILE = BASE_DIR / "top_features.png"
NOTES_FILE = BASE_DIR / "feature_importance_notes.txt"
RANDOM_STATE = 42


def create_plot(feature_names: list[str], importances: list[float]) -> None:
    """Create a horizontal bar chart for the ten most important features."""
    image = Image.new("RGB", (1200, 950), "white")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default(size=34)
    label_font = ImageFont.load_default(size=21)
    value_font = ImageFont.load_default(size=19)

    draw.text(
        (600, 45),
        "Random Forest: Top 10 Feature Importances",
        fill="#111827",
        font=title_font,
        anchor="mm",
    )

    bar_left, bar_width = 360, 680
    maximum = max(importances) if importances else 1
    for index, (name, importance) in enumerate(zip(feature_names, importances)):
        y = 100 + index * 78
        draw.text((335, y + 24), name, fill="#111827", font=label_font, anchor="rm")
        draw.rectangle(
            (bar_left, y, bar_left + bar_width, y + 48),
            fill="#e2e8f0",
            outline="#64748b",
            width=2,
        )
        filled_width = int(bar_width * importance / maximum)
        draw.rectangle(
            (bar_left, y, bar_left + filled_width, y + 48),
            fill="#0f766e",
        )
        draw.text(
            (bar_left + filled_width + 10, y + 24),
            f"{importance:.4f}",
            fill="#111827",
            font=value_font,
            anchor="lm",
        )

    draw.text(
        (600, 915),
        "Importance values sum to 1 across all 30 input features",
        fill="#475569",
        font=value_font,
        anchor="mm",
    )
    image.save(PLOT_FILE)


def main() -> None:
    dataset = load_breast_cancer()
    x_train, _, y_train, _ = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=dataset.target,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    model.fit(x_train, y_train)
    importance_pairs = sorted(
        zip(dataset.feature_names, model.feature_importances_),
        key=lambda pair: pair[1],
        reverse=True,
    )

    with VALUES_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["rank", "feature", "importance"])
        for rank, (name, importance) in enumerate(importance_pairs, start=1):
            writer.writerow([rank, name, f"{importance:.6f}"])

    top_features = importance_pairs[:10]
    create_plot(
        [str(name) for name, _ in top_features],
        [float(importance) for _, importance in top_features],
    )

    top_name, top_value = top_features[0]
    notes = (
        "Random Forest Feature Importance\n"
        "================================\n"
        f"Most important feature: {top_name}\n"
        f"Importance value: {top_value:.4f}\n\n"
        "Feature importance measures how much each input contributes to reducing "
        "uncertainty across the forest's decision trees. It shows model influence, "
        "not proof that a feature causes the outcome.\n"
    )
    NOTES_FILE.write_text(notes, encoding="utf-8")

    print("Top 10 features:")
    for name, importance in top_features:
        print(f"  {name}: {importance:.4f}")
    print(f"\nSaved all feature values to {VALUES_FILE.name}")
    print(f"Saved visualization to {PLOT_FILE.name}")


if __name__ == "__main__":
    main()

