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
