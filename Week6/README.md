# Week 6 - Clustering

This folder contains the four Monday exercises:

1. A short test covering preprocessing, Decision Trees, and Random Forests.
2. An introduction to grouping data without target labels.
3. Basic KMeans training and cluster assignment inspection.
4. A two-feature cluster visualization.

## Day 2 - Improving and Analyzing Clusters

The four Tuesday exercises are:

1. Test K values from 1 to 10 and create an elbow plot.
2. Add cluster labels and analyze the customer groups.
3. Compare KMeans before and after standardization.
4. Apply clustering to a small sample customer dataset.

## Day 3 - Principal Component Analysis

The four Wednesday exercises are:

1. Apply PCA to the four-feature Iris dataset.
2. Project the data into two dimensions and create a scatter plot.
3. Inspect the explained variance retained by each component.
4. Run KMeans on the PCA data and compare it with the original features.

## Day 4 - Anomaly and Outlier Detection

The four Thursday exercises are:

1. Detect unusual values using z-scores and the IQR method.
2. Find points that are far from the center using Euclidean distance.
3. Cap outliers in customer data and compare results before and after cleaning.
4. Summarize the week's unsupervised-learning concepts in one notebook.

All examples use deterministic random seeds so the results can be reproduced.

## Run

From the repository root:

```bash
.venv/bin/python Week6/Day1_Test/tree_preprocessing_test.py
.venv/bin/python Week6/Day1_UnsupervisedIntro/unsupervised_intro.py
.venv/bin/python Week6/Day1_KMeansBasics/kmeans_basics.py
.venv/bin/python Week6/Day1_ClusterVisualization/cluster_visualization.py
```

Run the Day 2 programs in this order:

```bash
.venv/bin/python Week6/Day2_RealDatasetClustering/real_dataset_clustering.py
.venv/bin/python Week6/Day2_ElbowMethod/elbow_method.py
.venv/bin/python Week6/Day2_ClusterAnalysis/cluster_analysis.py
.venv/bin/python Week6/Day2_ScalingForClusters/scaling_for_clusters.py
```

Run the Day 3 programs:

```bash
.venv/bin/python Week6/Day3_PCABasics/pca_basics.py
.venv/bin/python Week6/Day3_PCAProjection/pca_projection.py
.venv/bin/python Week6/Day3_ExplainedVariance/explained_variance.py
.venv/bin/python Week6/Day3_ClusteringWithPCA/clustering_with_pca.py
```

Run the Day 4 programs:

```bash
.venv/bin/python Week6/Day4_AnomalyDetection/anomaly_detection.py
.venv/bin/python Week6/Day4_OutlierCheck/distance_outlier_check.py
.venv/bin/python Week6/Day4_RealOutlierHandling/real_outlier_handling.py
```

Open `Week6/Day4_SummaryNotebook/unsupervised_learning_summary.ipynb` in
Jupyter or VS Code to review and run the complete summary.
