"""Introduce unsupervised learning by grouping unlabeled sample data."""

import csv
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "grouped_points.csv"
NOTES_FILE = BASE_DIR / "unsupervised_learning_notes.txt"
RANDOM_STATE = 42


def main() -> None:
    features, _ = make_blobs(
        n_samples=60,
        centers=3,
        n_features=2,
        cluster_std=0.75,
        random_state=RANDOM_STATE,
    )
    model = KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE)
    cluster_labels = model.fit_predict(features)

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["feature_1", "feature_2", "cluster"])
        for point, cluster in zip(features, cluster_labels):
            writer.writerow([f"{point[0]:.4f}", f"{point[1]:.4f}", int(cluster)])

    cluster_sizes = [int((cluster_labels == cluster).sum()) for cluster in range(3)]
    notes = (
        "Unsupervised Learning Introduction\n"
        "==================================\n"
        "Unsupervised learning finds patterns without known target labels.\n"
        "KMeans assigns each point to the nearest learned cluster center.\n"
        f"Rows grouped: {len(features)}\n"
        f"Number of clusters: {model.n_clusters}\n"
        f"Cluster sizes: {cluster_sizes}\n\n"
        "Cluster numbers are identifiers only; cluster 0 is not better or worse than cluster 1.\n"
    )
    NOTES_FILE.write_text(notes, encoding="utf-8")
    print(notes)
    print(f"Saved grouped data to {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()

