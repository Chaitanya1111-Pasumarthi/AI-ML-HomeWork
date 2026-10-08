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

## Day 3: Validation and Tuning

Day 3 continues with the breast-cancer dataset to:

1. Measure model reliability with five-fold cross-validation.
2. Tune a Decision Tree with `GridSearchCV`.
3. Inspect and explain the best parameters.
4. Compare model accuracy in a structured CSV sheet.

## Day 4: Preprocessing and Pipelines

Day 4 uses a small student-success CSV containing numerical columns,
categorical columns, and missing values to:

1. Impute missing values, encode categories, scale features, and split data.
2. Practice label encoding and one-hot encoding.
3. Combine preprocessing and classification in a scikit-learn pipeline.
4. Run one complete workflow from raw CSV data through evaluation.

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
.venv/bin/python Week5/Day3_CrossValidation/cross_validation.py
.venv/bin/python Week5/Day3_HyperparameterTuning/hyperparameter_tuning.py
.venv/bin/python Week5/Day3_BestParameters/best_parameters.py
.venv/bin/python Week5/Day3_AccuracyComparison/accuracy_comparison.py
.venv/bin/python Week5/Day4_PreprocessingRevision/preprocessing_revision.py
.venv/bin/python Week5/Day4_CategoricalEncoding/categorical_encoding.py
.venv/bin/python Week5/Day4_PipelineBasics/pipeline_basics.py
.venv/bin/python Week5/Day4_EndToEndPractice/end_to_end_practice.py
```
