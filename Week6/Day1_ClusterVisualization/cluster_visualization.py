"""Create a two-feature scatter plot of KMeans clusters."""

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


BASE_DIR = Path(__file__).resolve().parent
PLOT_FILE = BASE_DIR / "kmeans_clusters.png"
SUMMARY_FILE = BASE_DIR / "visualization_summary.txt"
RANDOM_STATE = 42


def scale_values(
    values: np.ndarray,
    start: int,
    end: int,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> np.ndarray:
    minimum = float(values.min()) if minimum is None else minimum
    maximum = float(values.max()) if maximum is None else maximum
    return start + (values - minimum) * (end - start) / (maximum - minimum)


def main() -> None:
    features, _ = make_blobs(
        n_samples=120,
        centers=[(-5, -2), (0, 5), (5, 0)],
        cluster_std=[0.9, 1.0, 0.8],
        random_state=RANDOM_STATE,
    )
    model = KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE)
    assignments = model.fit_predict(features)

    image = Image.new("RGB", (1100, 760), "white")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default(size=34)
    label_font = ImageFont.load_default(size=23)
    legend_font = ImageFont.load_default(size=20)
    draw.text(
        (550, 45),
        "KMeans Clusters for Two-Feature Data",
        fill="#111827",
        font=title_font,
        anchor="mm",
    )

    plot_left, plot_right, plot_top, plot_bottom = 120, 980, 100, 650
    draw.rectangle(
        (plot_left, plot_top, plot_right, plot_bottom),
        outline="#475569",
        width=2,
    )
    x_min, x_max = float(features[:, 0].min()), float(features[:, 0].max())
    y_min, y_max = float(features[:, 1].min()), float(features[:, 1].max())
    x_pixels = scale_values(features[:, 0], plot_left + 25, plot_right - 25, x_min, x_max)
    y_pixels = scale_values(features[:, 1], plot_bottom - 25, plot_top + 25, y_min, y_max)
    center_x = scale_values(
        model.cluster_centers_[:, 0], plot_left + 25, plot_right - 25, x_min, x_max
    )
    center_y = scale_values(
        model.cluster_centers_[:, 1], plot_bottom - 25, plot_top + 25, y_min, y_max
    )
    colors = ["#2563eb", "#dc2626", "#16a34a"]

    for x, y, cluster in zip(x_pixels, y_pixels, assignments):
        radius = 7
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=colors[int(cluster)],
            outline="white",
            width=1,
        )

    for cluster, (x, y) in enumerate(zip(center_x, center_y)):
        radius = 14
        draw.line((x - radius, y - radius, x + radius, y + radius), fill="#111827", width=5)
        draw.line((x - radius, y + radius, x + radius, y - radius), fill="#111827", width=5)
        draw.text(
            (x + 18, y - 18),
            f"Center {cluster}",
            fill="#111827",
            font=legend_font,
        )

    draw.text((550, 705), "Feature 1", fill="#111827", font=label_font, anchor="mm")
    draw.multiline_text(
        (55, 375), "Feature\n2", fill="#111827", font=label_font, anchor="mm", align="center"
    )
    for cluster, color in enumerate(colors):
        x = 335 + cluster * 215
        draw.ellipse((x, 75, x + 18, 93), fill=color)
        draw.text((x + 26, 84), f"Cluster {cluster}", fill="#111827", font=legend_font, anchor="lm")

    image.save(PLOT_FILE)
    summary = (
        "Cluster Visualization\n"
        "=====================\n"
        f"Points plotted: {len(features)}\n"
        "Colors show KMeans cluster assignments.\n"
        "Black X markers show the learned cluster centers.\n"
    )
    SUMMARY_FILE.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Saved visualization to {PLOT_FILE.name}")


if __name__ == "__main__":
    main()
