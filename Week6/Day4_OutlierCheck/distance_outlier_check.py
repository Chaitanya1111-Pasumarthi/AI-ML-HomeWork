import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


OUTPUT_DIR = Path(__file__).parent
points = np.array([
    [2.0, 2.2], [2.4, 1.8], [1.7, 2.5], [2.2, 2.8], [1.8, 1.6],
    [2.7, 2.3], [1.5, 2.0], [2.3, 1.5], [8.5, 8.0], [-3.5, 7.5],
])
center = np.median(points, axis=0)
distances = np.linalg.norm(points - center, axis=1)
threshold = distances.mean() + distances.std()
is_outlier = distances > threshold

with (OUTPUT_DIR / "distance_results.csv").open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["point", "x", "y", "distance_from_center", "is_outlier"])
    for number, (point, distance, result) in enumerate(
            zip(points, distances, is_outlier), start=1):
        writer.writerow([number, point[0], point[1], round(distance, 3), bool(result)])

width, height, margin = 760, 520, 60
image = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(image)
draw.line((margin, height - margin, width - margin, height - margin), fill="black", width=2)
draw.line((margin, margin, margin, height - margin), fill="black", width=2)
minimum = points.min(axis=0) - 1
maximum = points.max(axis=0) + 1
for point, result in zip(points, is_outlier):
    x = margin + (point[0] - minimum[0]) / (maximum[0] - minimum[0]) * (width - 2 * margin)
    y = height - margin - (point[1] - minimum[1]) / (maximum[1] - minimum[1]) * (height - 2 * margin)
    color = "#d83b36" if result else "#2864dc"
    draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color)
draw.text((width // 2 - 95, 20), "Distance-Based Outlier Check", fill="black")
draw.text((margin + 10, margin), "Red = outlier", fill="#d83b36")
image.save(OUTPUT_DIR / "distance_outliers.png")

outlier_points = points[is_outlier].tolist()
(OUTPUT_DIR / "distance_summary.txt").write_text(
    "DISTANCE-BASED OUTLIER CHECK\n"
    f"Center (median): ({center[0]:.2f}, {center[1]:.2f})\n"
    f"Distance threshold: {threshold:.2f}\n"
    f"Outlier points: {outlier_points}\n\n"
    "Points farther than the mean distance plus one standard deviation are marked as outliers.\n"
)
print(f"Detected {len(outlier_points)} far-away points: {outlier_points}.")
