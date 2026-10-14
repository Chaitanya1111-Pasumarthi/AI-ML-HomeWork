import csv
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path(__file__).parent
iris = load_iris()
scaled_data = StandardScaler().fit_transform(iris.data)
pca_data = PCA(n_components=2).fit_transform(scaled_data)

original_labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(scaled_data)
pca_labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(pca_data)

original_score = silhouette_score(scaled_data, original_labels)
pca_score = silhouette_score(pca_data, pca_labels)
agreement = adjusted_rand_score(original_labels, pca_labels)

with (OUTPUT_DIR / "pca_cluster_assignments.csv").open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["sample", "species", "pca_1", "pca_2",
                     "original_feature_cluster", "pca_cluster"])
    for number, (target, point, original, reduced) in enumerate(
            zip(iris.target, pca_data, original_labels, pca_labels), start=1):
        writer.writerow([number, iris.target_names[target], round(point[0], 4),
                         round(point[1], 4), int(original), int(reduced)])

(OUTPUT_DIR / "clustering_comparison.txt").write_text(
    "CLUSTERING WITH PCA\n"
    f"Original 4D silhouette score: {original_score:.3f}\n"
    f"PCA 2D silhouette score: {pca_score:.3f}\n"
    f"Cluster agreement (Adjusted Rand Index): {agreement:.3f}\n\n"
    "The PCA version uses only two features, is easy to visualize, and produces "
    "clusters similar to those from the original four standardized features.\n"
)
print("Compared KMeans clusters from the original 4D data and 2D PCA data.")
