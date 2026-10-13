import csv
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path(__file__).parent
DATA_FILE = OUTPUT_DIR.parent / "Day2_RealDatasetClustering" / "customer_data.csv"


with DATA_FILE.open() as file:
    customers = list(csv.DictReader(file))

values = np.array([[row["age"], row["annual_income_k"], row["spending_score"]]
                   for row in customers], dtype=float)
raw_model = KMeans(n_clusters=4, random_state=42, n_init=10)
raw_labels = raw_model.fit_predict(values)

scaled_values = StandardScaler().fit_transform(values)
scaled_model = KMeans(n_clusters=4, random_state=42, n_init=10)
scaled_labels = scaled_model.fit_predict(scaled_values)

raw_score = silhouette_score(values, raw_labels)
scaled_score = silhouette_score(scaled_values, scaled_labels)
agreement = adjusted_rand_score(raw_labels, scaled_labels)

with (OUTPUT_DIR / "scaling_comparison.csv").open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["customer_id", "cluster_without_scaling", "cluster_with_scaling"])
    for customer, raw, scaled in zip(customers, raw_labels, scaled_labels):
        writer.writerow([customer["customer_id"], int(raw), int(scaled)])

(OUTPUT_DIR / "scaling_summary.txt").write_text(
    "STANDARDIZATION BEFORE CLUSTERING\n"
    f"Silhouette score without scaling: {raw_score:.3f}\n"
    f"Silhouette score with scaling: {scaled_score:.3f}\n"
    f"Assignment agreement (Adjusted Rand Index): {agreement:.3f}\n\n"
    "Standardization gives age, income, and spending equal influence by putting them "
    "on comparable scales. The scaled result is used for the final customer groups.\n"
)
print("Compared KMeans before and after standardization.")
