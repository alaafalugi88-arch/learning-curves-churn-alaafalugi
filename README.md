# Learning Curves Diagnostic — Telecom Churn

## Overview

This project uses learning curves to diagnose the performance of a logistic regression model on a telecom churn dataset.

The goal is to determine whether the model suffers from high bias or high variance and to identify the best next steps for improvement.

---

## Methodology

* Built a preprocessing + modeling pipeline using:

  * StandardScaler for numeric features
  * OneHotEncoder for categorical features
* Used Logistic Regression with `class_weight="balanced"`
* Applied StratifiedKFold cross-validation (5 folds)
* Evaluated using **F1-score** due to class imbalance
* Generated learning curves using 5 different training set sizes
* Plotted training and validation scores with confidence intervals

---

## Results

The learning curve shows:

* Training score starts relatively higher (~0.53) and decreases as training size increases
* Validation score starts lower (~0.31) and gradually increases
* Both curves **converge around ~0.35–0.38**
* The gap between training and validation becomes **small at larger dataset sizes**
* The validation curve **plateaus early and shows limited improvement**

---

## Analysis

The model is primarily suffering from **high bias (underfitting)**.

This is evident because:

* Training and validation scores converge to a similar value
* Both scores remain relatively low
* The model is unable to capture complex patterns in the data

Increasing the dataset size is **unlikely to significantly improve performance**, since the validation curve has already plateaued and shows minimal gains with more data.

Instead, the limitation comes from the model itself being too simple.

---

## Recommendation

To improve performance, the next step should be increasing model complexity. Possible approaches include:

* Adding interaction or engineered features
* Introducing polynomial features
* Using more flexible models such as:

  * Random Forest
  * Gradient Boosting

These approaches can help the model capture more complex relationships and reduce bias.

---

## Output

* `learning_curve.png` — visualization of training vs validation performance
* `learning_curve.py` — script to generate the plot

---

## How to Run

```bash
pip install -r requirements.txt
python learning_curve.py
```
