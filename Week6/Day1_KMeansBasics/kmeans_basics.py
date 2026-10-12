"""Apply KMeans and inspect cluster assignments and centers."""

import csv
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


BASE_DIR = Path(__file__).resolve().parent
ASSIGNMENTS_FILE = BASE_DIR / "cluster_assignments.csv"
CENTERS_FILE = BASE_DIR / "cluster_centers.csv"
SUMMARY_FILE = BASE_DIR / "kmeans_summary.txt"
RANDOM_STATE = 42


def main() -> None:
    features, _ = make_blobs(
        n_samples=90,
        centers=[(-5, -2), (0, 5), (5, 0)],
        cluster_std=[0.8, 0.9, 0.7],
        random_state=RANDOM_STATE,
    )
    model = KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE)
    assignments = model.fit_predict(features)

    with ASSIGNMENTS_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["point_id", "feature_1", "feature_2", "cluster"])
        for point_id, (point, cluster) in enumerate(zip(features, assignments), start=1):
            writer.writerow(
                [point_id, f"{point[0]:.4f}", f"{point[1]:.4f}", int(cluster)]
            )

    with CENTERS_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["cluster", "center_feature_1", "center_feature_2"])
        for cluster, center in enumerate(model.cluster_centers_):
            writer.writerow([cluster, f"{center[0]:.4f}", f"{center[1]:.4f}"])

    sizes = [int((assignments == cluster).sum()) for cluster in range(3)]
    summary = (
        "KMeans Basics\n"
        "=============\n"
        f"Data points: {len(features)}\n"
        f"Clusters requested: {model.n_clusters}\n"
        f"Cluster sizes: {sizes}\n"
        f"Inertia: {model.inertia_:.4f}\n\n"
        "Inertia is the sum of squared distances from points to their assigned centers.\n"
    )
    SUMMARY_FILE.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Saved assignments to {ASSIGNMENTS_FILE.name}")
    print(f"Saved centers to {CENTERS_FILE.name}")


if __name__ == "__main__":
    main()

