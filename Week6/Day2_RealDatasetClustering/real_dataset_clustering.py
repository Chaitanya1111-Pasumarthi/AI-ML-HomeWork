import csv
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path(__file__).parent
FEATURES = ["age", "annual_income_k", "spending_score"]


def create_customer_data():
    """Create a small, reproducible sample customer dataset."""
    rng = np.random.default_rng(42)
    profiles = [
        (24, 32, 78),
        (44, 82, 76),
        (48, 35, 28),
        (32, 78, 27),
    ]
    rows = []
    customer_id = 1001
    for age, income, spending in profiles:
        for _ in range(30):
            rows.append([
                customer_id,
                int(np.clip(rng.normal(age, 4), 18, 70)),
                round(float(np.clip(rng.normal(income, 8), 15, 120)), 1),
                int(np.clip(rng.normal(spending, 9), 1, 100)),
            ])
            customer_id += 1
    rng.shuffle(rows)
    return rows


def write_csv(path, header, rows):
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


customers = create_customer_data()
write_csv(OUTPUT_DIR / "customer_data.csv", ["customer_id"] + FEATURES, customers)

values = np.array([row[1:] for row in customers], dtype=float)
scaler = StandardScaler()
scaled_values = scaler.fit_transform(values)
model = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = model.fit_predict(scaled_values)

clustered_rows = [row + [int(label)] for row, label in zip(customers, labels)]
write_csv(
    OUTPUT_DIR / "clustered_customers.csv",
    ["customer_id"] + FEATURES + ["cluster"],
    clustered_rows,
)

centers = scaler.inverse_transform(model.cluster_centers_)
center_rows = [[number] + [round(value, 2) for value in center]
               for number, center in enumerate(centers)]
write_csv(OUTPUT_DIR / "cluster_centers.csv", ["cluster"] + FEATURES, center_rows)

counts = np.bincount(labels, minlength=4)
summary = [
    "REAL DATASET CLUSTERING",
    "Dataset: 120 sample customer records",
    "Features: age, annual income, and spending score",
    "Method: StandardScaler followed by KMeans with K=4",
    "",
]
summary.extend(f"Cluster {number}: {count} customers" for number, count in enumerate(counts))
(OUTPUT_DIR / "clustering_summary.txt").write_text("\n".join(summary) + "\n")

print("Created customer dataset and grouped 120 customers into 4 clusters.")
