import csv
from pathlib import Path

import numpy as np


OUTPUT_DIR = Path(__file__).parent
SOURCE_FILE = OUTPUT_DIR.parent / "Day2_RealDatasetClustering" / "customer_data.csv"

with SOURCE_FILE.open() as file:
    customers = list(csv.DictReader(file))

# Add three intentionally unusual records to demonstrate real-data cleaning.
customers.extend([
    {"customer_id": "2001", "age": "92", "annual_income_k": "250", "spending_score": "99"},
    {"customer_id": "2002", "age": "16", "annual_income_k": "4", "spending_score": "2"},
    {"customer_id": "2003", "age": "85", "annual_income_k": "210", "spending_score": "5"},
])
features = ["age", "annual_income_k", "spending_score"]
values = np.array([[float(row[name]) for name in features] for row in customers])
original_means = values.mean(axis=0)

cleaned = values.copy()
bounds = []
outlier_rows = set()
for column in range(values.shape[1]):
    q1, q3 = np.percentile(values[:, column], [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    bounds.append((lower, upper))
    matches = np.where((values[:, column] < lower) | (values[:, column] > upper))[0]
    outlier_rows.update(matches.tolist())
    cleaned[:, column] = np.clip(values[:, column], lower, upper)

with (OUTPUT_DIR / "cleaned_customer_data.csv").open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["customer_id"] + features + ["was_capped"])
    for index, (customer, row) in enumerate(zip(customers, cleaned)):
        writer.writerow([customer["customer_id"]] + [round(value, 2) for value in row]
                        + [index in outlier_rows])

cleaned_means = cleaned.mean(axis=0)
lines = [
    "REAL DATA CLEANING WITH OUTLIERS",
    f"Rows analyzed: {len(customers)}",
    f"Rows containing outliers: {len(outlier_rows)}",
    "Method: cap values at the IQR lower and upper bounds",
    "",
    "MEAN BEFORE -> AFTER CLEANING",
]
for name, before, after, (lower, upper) in zip(
        features, original_means, cleaned_means, bounds):
    lines.append(
        f"{name}: {before:.2f} -> {after:.2f} (bounds {lower:.2f} to {upper:.2f})"
    )
(OUTPUT_DIR / "outlier_cleaning_summary.txt").write_text("\n".join(lines) + "\n")
print(f"Capped outliers in {len(outlier_rows)} of {len(customers)} customer rows.")
