import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path(__file__).parent
DATA_FILE = OUTPUT_DIR.parent / "Day2_RealDatasetClustering" / "customer_data.csv"


with DATA_FILE.open() as file:
    rows = list(csv.DictReader(file))
values = np.array([[row["age"], row["annual_income_k"], row["spending_score"]]
                   for row in rows], dtype=float)
values = StandardScaler().fit_transform(values)

k_values = list(range(1, 11))
inertias = []
for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(values)
    inertias.append(model.inertia_)

# The point farthest from the line joining the first and last values is the elbow.
points = np.column_stack((k_values, inertias))
start, end = points[0], points[-1]
line = end - start
distances = np.abs(line[0] * (start[1] - points[:, 1])
                   - line[1] * (start[0] - points[:, 0])) / np.linalg.norm(line)
best_k = k_values[int(np.argmax(distances))]

with (OUTPUT_DIR / "elbow_results.csv").open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["k", "inertia"])
    writer.writerows(zip(k_values, [round(value, 3) for value in inertias]))

width, height, margin = 800, 520, 70
image = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(image)
draw.line((margin, margin, margin, height - margin), fill="black", width=2)
draw.line((margin, height - margin, width - margin, height - margin), fill="black", width=2)
max_inertia = max(inertias)
plot_points = []
for k, inertia in zip(k_values, inertias):
    x = margin + (k - 1) * (width - 2 * margin) / 9
    y = height - margin - inertia / max_inertia * (height - 2 * margin)
    plot_points.append((x, y))
draw.line(plot_points, fill="#2864dc", width=4)
for k, (x, y) in zip(k_values, plot_points):
    color = "#dc3c32" if k == best_k else "#2864dc"
    draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color)
    draw.text((x - 4, height - margin + 12), str(k), fill="black")
draw.text((width // 2 - 60, 20), "KMeans Elbow Method", fill="black")
draw.text((width // 2 - 40, height - 35), "Number of clusters (K)", fill="black")
draw.text((margin + 15, margin), "Inertia", fill="black")
draw.text((margin + 15, margin + 25), f"Suggested K = {best_k}", fill="#dc3c32")
image.save(OUTPUT_DIR / "elbow_method.png")

(OUTPUT_DIR / "elbow_summary.txt").write_text(
    "ELBOW METHOD\n"
    "Tested K values: 1 through 10\n"
    f"Suggested elbow: K={best_k}\n"
    "The largest bend indicates a useful balance between compact clusters and simplicity.\n"
)
print(f"Tested K=1 to 10. The suggested elbow is K={best_k}.")
