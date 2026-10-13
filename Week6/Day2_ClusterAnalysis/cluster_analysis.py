import csv
from collections import defaultdict
from pathlib import Path


OUTPUT_DIR = Path(__file__).parent
DATA_FILE = OUTPUT_DIR.parent / "Day2_RealDatasetClustering" / "clustered_customers.csv"


with DATA_FILE.open() as file:
    customers = list(csv.DictReader(file))

groups = defaultdict(list)
for customer in customers:
    groups[int(customer["cluster"])].append(customer)

summary_rows = []
report = ["CLUSTER LABEL ANALYSIS", ""]
for cluster, members in sorted(groups.items()):
    age = sum(float(row["age"]) for row in members) / len(members)
    income = sum(float(row["annual_income_k"]) for row in members) / len(members)
    spending = sum(float(row["spending_score"]) for row in members) / len(members)
    income_level = "high-income" if income >= 55 else "lower-income"
    spending_level = "high-spending" if spending >= 50 else "low-spending"
    label = f"{income_level}, {spending_level} customers"
    summary_rows.append([cluster, len(members), round(age, 1), round(income, 1),
                         round(spending, 1), label])
    report.append(
        f"Cluster {cluster}: {len(members)} customers, average age {age:.1f}, "
        f"income ${income:.1f}k, spending score {spending:.1f}. "
        f"This group represents {label}."
    )

with (OUTPUT_DIR / "cluster_summary.csv").open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["cluster", "customers", "average_age", "average_income_k",
                     "average_spending_score", "group_label"])
    writer.writerows(summary_rows)

(OUTPUT_DIR / "cluster_analysis.txt").write_text("\n".join(report) + "\n")
print("Added meaningful labels and analyzed all customer clusters.")
