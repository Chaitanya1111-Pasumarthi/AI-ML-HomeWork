# Week 5 - Day 1: Decision Trees

This folder contains the three Day 1 exercises after the completed test:

1. Train and evaluate a Decision Tree classifier.
2. Compare several `max_depth` settings.
3. Visualize the trained model's feature importance.

All exercises use scikit-learn's built-in Iris dataset and a reproducible,
stratified train/test split.

## Day 2: Random Forest

Day 2 uses scikit-learn's built-in breast-cancer dataset to:

1. Train and evaluate a Random Forest classifier.
2. Compare a Decision Tree with a Random Forest.
3. Compare training and testing accuracy to identify overfitting.
4. Extract and visualize the Random Forest's most important features.

## Run

From the repository root:

```bash
.venv/bin/python Week5/Day1_DecisionTree/decision_tree.py
.venv/bin/python Week5/Day1_TreeDepthExperiment/tree_depth_experiment.py
.venv/bin/python Week5/Day1_TreeResults/tree_results.py
.venv/bin/python Week5/Day2_RandomForest/random_forest.py
.venv/bin/python Week5/Day2_ModelCompare/model_compare.py
.venv/bin/python Week5/Day2_OverfittingCheck/overfitting_check.py
.venv/bin/python Week5/Day2_FeatureImportance/feature_importance.py
```

