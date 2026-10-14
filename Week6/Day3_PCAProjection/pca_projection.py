import csv
from pathlib import Path

from PIL import Image, ImageDraw


OUTPUT_DIR = Path(__file__).parent
DATA_FILE = OUTPUT_DIR.parent / "Day3_PCABasics" / "pca_transformed_data.csv"
COLORS = {"setosa": "#2864dc", "versicolor": "#e28b24", "virginica": "#3a9d5d"}

with DATA_FILE.open() as file:
    rows = list(csv.DictReader(file))

x_values = [float(row["principal_component_1"]) for row in rows]
y_values = [float(row["principal_component_2"]) for row in rows]
width, height, margin = 820, 560, 70
image = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(image)
draw.line((margin, margin, margin, height - margin), fill="black", width=2)
draw.line((margin, height - margin, width - margin, height - margin), fill="black", width=2)

def scale(value, low, high, start, end):
    return start + (value - low) * (end - start) / (high - low)

for row in rows:
    x = scale(float(row["principal_component_1"]), min(x_values), max(x_values),
              margin, width - margin)
    y = scale(float(row["principal_component_2"]), min(y_values), max(y_values),
              height - margin, margin)
    draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=COLORS[row["species"]])

draw.text((width // 2 - 70, 20), "Iris PCA 2D Projection", fill="black")
draw.text((width // 2 - 55, height - 35), "Principal Component 1", fill="black")
draw.text((margin + 10, margin - 25), "Principal Component 2", fill="black")
for index, (species, color) in enumerate(COLORS.items()):
    x = width - 180
    y = 75 + index * 24
    draw.rectangle((x, y, x + 12, y + 12), fill=color)
    draw.text((x + 20, y - 2), species, fill="black")

image.save(OUTPUT_DIR / "pca_2d_projection.png")
(OUTPUT_DIR / "projection_summary.txt").write_text(
    "2D PCA PROJECTION\n"
    "The original four Iris measurements were reduced to two axes.\n"
    "Setosa is clearly separated; versicolor and virginica overlap slightly.\n"
)
print("Created the 2D PCA projection for 150 Iris samples.")
