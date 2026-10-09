# Student Performance Prediction Project

## Problem statement

Predict a student's numerical final score using study habits, attendance,
previous performance, sleep, tutoring, and learning mode.

## Dataset summary

- 120 reproducibly generated student records
- Four numerical features and two categorical features
- Final score is the regression target
- Selected values are missing intentionally to practice realistic preprocessing

## Preprocessing

1. Split data into 80% training and 20% testing sets.
2. Fill missing numerical values with training-set medians.
3. Standardize numerical features.
4. Fill missing categories with the most frequent training value.
5. One-hot encode categorical features.
6. Keep preprocessing and modeling together in scikit-learn pipelines.

## Models and results

| Model | MAE | MSE | R-squared |
|---|---:|---:|---:|
| Linear Regression | 2.7310 | 14.1757 | 0.8518 |
| Random Forest | 3.2425 | 20.5800 | 0.7848 |
| Decision Tree | 5.3681 | 60.7594 | 0.3647 |

## Conclusion

**Linear Regression** was selected because it produced the lowest test MAE of
**2.7310**. Its MSE was **14.1757** and R-squared was
**0.8518**. MAE was the selection metric because it directly describes
the average prediction error in score points. These results apply to this
educational synthetic dataset and should not be treated as a real academic
decision system.
