# Mini ML Project: Iris Species Classification

## Objective

Predict whether an Iris flower is **setosa**, **versicolor**, or **virginica**
from its sepal and petal measurements.

## Workflow

1. Export and load the Iris dataset as `iris_dataset.csv`.
2. Separate the four input features from the target species.
3. Create a stratified 80/20 training and testing split.
4. Standardize features using training data only.
5. Train a multiclass logistic regression model.
6. Generate predictions and confidence scores.
7. Evaluate accuracy, precision, recall, F1-score, and the confusion matrix.

## Run

From the repository root:

```bash
.venv/bin/python Week4/Day5_MiniMLProject/mini_ml_project.py
```

## Outputs

- `iris_dataset.csv`: project dataset
- `predictions.csv`: held-out test predictions
- `evaluation_report.txt`: metrics and conclusion
- `confusion_matrix.png`: visual evaluation of the predictions

