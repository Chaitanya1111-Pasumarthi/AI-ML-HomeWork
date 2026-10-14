import csv
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path(__file__).parent
iris = load_iris()
scaled_data = StandardScaler().fit_transform(iris.data)
pca = PCA(n_components=2)
projected_data = pca.fit_transform(scaled_data)

with (OUTPUT_DIR / "pca_transformed_data.csv").open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["principal_component_1", "principal_component_2", "species"])
    for point, target in zip(projected_data, iris.target):
        writer.writerow([round(point[0], 4), round(point[1], 4), iris.target_names[target]])

retained = pca.explained_variance_ratio_.sum() * 100
(OUTPUT_DIR / "pca_basics_summary.txt").write_text(
    "PCA BASICS\n"
    "Dataset: Iris (150 rows and 4 numerical features)\n"
    "Original dimensions: 4\n"
    "Reduced dimensions: 2\n"
    f"Information retained: {retained:.2f}%\n\n"
    "PCA combines correlated features into new principal components. "
    "It reduces dimensions while keeping as much variation as possible.\n"
)

print(f"Reduced the Iris dataset from 4 dimensions to 2 ({retained:.2f}% retained).")
