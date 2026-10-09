"""End-to-end student final-score prediction project."""

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR.parent / "Day5_ProjectDocs"
DATA_FILE = BASE_DIR / "student_performance.csv"
COMPARISON_FILE = BASE_DIR / "model_comparison.csv"
PREDICTIONS_FILE = BASE_DIR / "best_model_predictions.csv"
SUMMARY_FILE = BASE_DIR / "best_model_summary.txt"
PLOT_FILE = BASE_DIR / "model_comparison.png"
DOCUMENTATION_FILE = DOCS_DIR / "README.md"
RANDOM_STATE = 42


def generate_dataset(rows: int = 120) -> None:
    """Create a reproducible mixed-type dataset with some missing values."""
    rng = np.random.default_rng(RANDOM_STATE)
    study_hours = rng.uniform(1.0, 10.0, rows)
    attendance = rng.integers(50, 101, rows)
    previous_score = rng.integers(40, 96, rows)
    sleep_hours = rng.uniform(4.5, 9.0, rows)
    tutoring = rng.choice(["yes", "no"], rows, p=[0.45, 0.55])
    learning_mode = rng.choice(["classroom", "online", "group"], rows)
    noise = rng.normal(0, 3.0, rows)

    final_scores = (
        -5
        + 2.5 * study_hours
        + 0.25 * attendance
        + 0.35 * previous_score
        + 0.6 * sleep_hours
        + np.where(tutoring == "yes", 2.0, 0.0)
        + np.where(learning_mode == "group", 1.0, 0.0)
        + noise
    )
    final_scores = np.clip(final_scores, 0, 100)

    with DATA_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "study_hours",
                "attendance",
                "previous_score",
                "sleep_hours",
                "tutoring",
                "learning_mode",
                "final_score",
            ]
        )
        for index in range(rows):
            writer.writerow(
                [
                    "" if index % 17 == 0 else f"{study_hours[index]:.2f}",
                    "" if index % 19 == 3 else int(attendance[index]),
                    "" if index % 31 == 7 else int(previous_score[index]),
                    "" if index % 37 == 11 else f"{sleep_hours[index]:.2f}",
                    "" if index % 23 == 13 else tutoring[index],
                    "" if index % 29 == 17 else learning_mode[index],
                    f"{final_scores[index]:.2f}",
                ]
            )


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Load feature values and the final-score target from the CSV file."""
    features, target = [], []
    with DATA_FILE.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            features.append(
                [
                    float(row["study_hours"]) if row["study_hours"] else np.nan,
                    float(row["attendance"]) if row["attendance"] else np.nan,
                    float(row["previous_score"]) if row["previous_score"] else np.nan,
                    float(row["sleep_hours"]) if row["sleep_hours"] else np.nan,
                    row["tutoring"] or None,
                    row["learning_mode"] or None,
                ]
            )
            target.append(float(row["final_score"]))
    return np.array(features, dtype=object), np.array(target)


def build_pipeline(regressor: object) -> Pipeline:
    """Combine leakage-safe preprocessing with a regression model."""
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
                [0, 1, 2, 3],
            ),
            (
                "category",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(missing_values=None, strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                [4, 5],
            ),
        ]
    )
    return Pipeline([("preprocessing", preprocessing), ("regressor", regressor)])


def create_comparison_plot(results: list[dict[str, float]]) -> None:
    """Create a horizontal MAE comparison chart; lower is better."""
    image = Image.new("RGB", (1100, 600), "white")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default(size=34)
    label_font = ImageFont.load_default(size=24)
    value_font = ImageFont.load_default(size=22)
    draw.text(
        (550, 52),
        "Student Score Models: Mean Absolute Error",
        fill="#111827",
        font=title_font,
        anchor="mm",
    )

    maximum = max(result["mae"] for result in results)
    colors = ["#15803d", "#2563eb", "#d97706"]
    for index, result in enumerate(results):
        y = 135 + index * 125
        draw.text((290, y + 30), result["model"], fill="#111827", font=label_font, anchor="rm")
        draw.rectangle((320, y, 950, y + 60), fill="#e2e8f0", outline="#64748b", width=2)
        width = int(630 * result["mae"] / maximum)
        draw.rectangle((320, y, 320 + width, y + 60), fill=colors[index])
        draw.text(
            (330 + width, y + 30),
            f"MAE {result['mae']:.2f}",
            fill="#111827",
            font=value_font,
            anchor="lm",
        )
    draw.text(
        (550, 550),
        "Lower MAE indicates predictions are closer to actual final scores",
        fill="#475569",
        font=value_font,
        anchor="mm",
    )
    image.save(PLOT_FILE)


def write_documentation(results: list[dict[str, float]], best: dict[str, float]) -> None:
    """Write the complete project documentation required by the homework."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    result_rows = "\n".join(
        f"| {result['model']} | {result['mae']:.4f} | {result['mse']:.4f} | {result['r2']:.4f} |"
        for result in results
    )
    documentation = f"""# Student Performance Prediction Project

## Problem statement

Predict a student's numerical final score using study habits, attendance,
previous performance, sleep, tutoring, and learning mode.

## Dataset summary

- 120 reproducibly generated student records
- Four numerical features and two categorical features
- Final score is the regression target
- Selected values are missing intentionally to practice realistic preprocessing

## Preprocessing

1. Split data into 80% training and 20% testing sets.
2. Fill missing numerical values with training-set medians.
3. Standardize numerical features.
4. Fill missing categories with the most frequent training value.
5. One-hot encode categorical features.
6. Keep preprocessing and modeling together in scikit-learn pipelines.

## Models and results

| Model | MAE | MSE | R-squared |
|---|---:|---:|---:|
{result_rows}

## Conclusion

**{best['model']}** was selected because it produced the lowest test MAE of
**{best['mae']:.4f}**. Its MSE was **{best['mse']:.4f}** and R-squared was
**{best['r2']:.4f}**. MAE was the selection metric because it directly describes
the average prediction error in score points. These results apply to this
educational synthetic dataset and should not be treated as a real academic
decision system.
"""
    DOCUMENTATION_FILE.write_text(documentation, encoding="utf-8")


def run_project() -> list[dict[str, float]]:
    """Generate data, train all models, evaluate them, and save outputs."""
    generate_dataset()
    features, target = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }

    results = []
    fitted_models = {}
    for name, regressor in models.items():
        model = build_pipeline(regressor)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        results.append(
            {
                "model": name,
                "mae": float(mean_absolute_error(y_test, predictions)),
                "mse": float(mean_squared_error(y_test, predictions)),
                "r2": float(r2_score(y_test, predictions)),
            }
        )
        fitted_models[name] = model

    results.sort(key=lambda result: result["mae"])
    best = results[0]
    best_predictions = fitted_models[best["model"]].predict(x_test)

    with COMPARISON_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["rank", "model", "mae", "mse", "r_squared"])
        for rank, result in enumerate(results, start=1):
            writer.writerow(
                [
                    rank,
                    result["model"],
                    f"{result['mae']:.4f}",
                    f"{result['mse']:.4f}",
                    f"{result['r2']:.4f}",
                ]
            )

    with PREDICTIONS_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["test_row", "actual_score", "predicted_score", "absolute_error"])
        for row_number, (actual, predicted) in enumerate(
            zip(y_test, best_predictions), start=1
        ):
            writer.writerow(
                [row_number, f"{actual:.2f}", f"{predicted:.2f}", f"{abs(actual - predicted):.2f}"]
            )

    summary = (
        "Best Model Summary\n"
        "==================\n"
        f"Selected model: {best['model']}\n"
        f"Mean Absolute Error: {best['mae']:.4f}\n"
        f"Mean Squared Error: {best['mse']:.4f}\n"
        f"R-squared: {best['r2']:.4f}\n"
        f"Training rows: {len(y_train)}\n"
        f"Testing rows: {len(y_test)}\n"
    )
    SUMMARY_FILE.write_text(summary, encoding="utf-8")
    create_comparison_plot(results)
    write_documentation(results, best)
    return results


def main() -> None:
    results = run_project()
    print("Model comparison (ranked by lowest MAE):")
    for rank, result in enumerate(results, start=1):
        print(
            f"  {rank}. {result['model']}: MAE={result['mae']:.4f}, "
            f"MSE={result['mse']:.4f}, R2={result['r2']:.4f}"
        )
    print(f"\nSelected model: {results[0]['model']}")
    print(f"Saved documentation to {DOCUMENTATION_FILE}")


if __name__ == "__main__":
    main()
