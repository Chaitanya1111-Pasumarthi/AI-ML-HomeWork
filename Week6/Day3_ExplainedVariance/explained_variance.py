import csv
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path(__file__).parent
iris = load_iris()
scaled_data = StandardScaler().fit_transform(iris.data)
pca = PCA().fit(scaled_data)
ratios = pca.explained_variance_ratio_

with (OUTPUT_DIR / "explained_variance.csv").open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["component", "explained_variance_percent", "cumulative_percent"])
    cumulative = 0
    for number, ratio in enumerate(ratios, start=1):
        cumulative += ratio * 100
        writer.writerow([number, round(ratio * 100, 2), round(cumulative, 2)])

two_components = ratios[:2].sum() * 100
(OUTPUT_DIR / "explained_variance_summary.txt").write_text(
    "EXPLAINED VARIANCE\n"
    + "\n".join(
        f"Component {number}: {ratio * 100:.2f}%"
        for number, ratio in enumerate(ratios, start=1)
    )
    + f"\n\nThe first two components retain {two_components:.2f}% of the information.\n"
    + "Therefore, two dimensions provide a useful visual summary of the dataset.\n"
)
print(f"The first two PCA components retain {two_components:.2f}% of the variance.")
