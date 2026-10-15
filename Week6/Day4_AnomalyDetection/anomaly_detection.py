import csv
from pathlib import Path

import numpy as np


OUTPUT_DIR = Path(__file__).parent
values = np.array([42, 45, 47, 43, 46, 44, 48, 41, 45, 46, 120, 5], dtype=float)

z_scores = np.abs((values - values.mean()) / values.std())
q1, q3 = np.percentile(values, [25, 75])
iqr = q3 - q1
lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr

with (OUTPUT_DIR / "anomaly_results.csv").open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["value", "z_score", "z_score_anomaly", "iqr_anomaly"])
    for value, score in zip(values, z_scores):
        writer.writerow([
            int(value), round(score, 3), bool(score > 2),
            bool(value < lower_bound or value > upper_bound),
        ])

z_anomalies = values[z_scores > 2].astype(int).tolist()
iqr_anomalies = values[(values < lower_bound) | (values > upper_bound)].astype(int).tolist()
(OUTPUT_DIR / "anomaly_summary.txt").write_text(
    "ANOMALY DETECTION\n"
    f"Z-score anomalies (threshold > 2): {z_anomalies}\n"
    f"IQR bounds: {lower_bound:.2f} to {upper_bound:.2f}\n"
    f"IQR anomalies: {iqr_anomalies}\n\n"
    "A z-score measures distance from the mean in standard deviations. "
    "IQR detects values outside the middle-data range and is less affected by extremes.\n"
)
print(f"Z-score found {z_anomalies}; IQR found {iqr_anomalies}.")
