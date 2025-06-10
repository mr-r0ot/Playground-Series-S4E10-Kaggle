Author: Mohammad Taha Gorji

# Implementation and Evaluation of an Ensemble Method for Binary Classification in the Playground Series S4E10 Competition

## Abstract

This paper presents a modular framework combining three state-of-the-art gradient boosting algorithms (LightGBM, XGBoost, and CatBoost) for binary classification in the Playground Series S4E10 competition. Through feature engineering—squared transformations and one-hot encoding—and five-fold stratified cross-validation with early stopping, our ensemble achieves a 95.27% out-of-fold accuracy. Inference latency is measured at 36 ms for CatBoost and 114 ms for LightGBM. The main contribution is a transparent, reproducible pipeline implemented across three Python scripts (`TrainModel.py`, `TestModel.py`, and `UseModel.py`), accessible to both practitioners and researchers.

## 1. Introduction

The Playground Series S4E10 synthetic dataset offers a controlled environment to benchmark gradient boosting algorithms. While LightGBM, XGBoost, and CatBoost individually demonstrate strong performance, their ensemble can further enhance accuracy and robustness against overfitting. This work:

* Provides a step-by-step, modular pipeline for model training and deployment.
* Demonstrates effective overfitting prevention via cross-validation and early stopping.
* Quantifies inference latency in a realistic setting.

## 2. Related Work

Gradient boosting decision trees (GBDT) and their optimized variants—XGBoost \[1], LightGBM \[2], and CatBoost \[3]—are widely adopted for tabular data tasks. Ensemble techniques like majority voting and stacking \[4] often yield improved stability and accuracy in machine learning competitions. This paper builds upon these established methods by delivering an end-to-end, reproducible implementation.

## 3. Methodology

### 3.1. Overall Architecture

```
+---------------+      +-----------------+      +---------------+
| Preprocessing | ---> |  Train Models   | ---> |   Ensemble    |
+---------------+      +-----------------+      +---------------+
         |                       |                       |
         v                       v                       v
    feature.csv              model.pkl           submission.csv
```

### 3.2. Pseudocode

```
# Load data
data_train, data_test = load_csvs()
# Preprocess
X_train, y_train = preprocess(data_train)
X_test = preprocess(data_test)
# Cross-validation training
for train_idx, val_idx in StratifiedKFold(n_splits=5):
    models = train_boosting_models(X_train[train_idx], y_train[train_idx])
    save_oof_preds(models, X_train[val_idx])
    save_fold_models(models)
# Ensemble predictions
oof_accuracy = compute_oof_accuracy()
submission = majority_vote(models, X_test)
save_csv(oof.csv, submission.csv)
```

## 4. Implementation Details

* **Language & Frameworks:** Python 3.10, scikit-learn 1.1, LightGBM 3.3, XGBoost 1.6, CatBoost 1.1
* **Dependencies:** numpy, pandas, joblib, matplotlib
* **Key Hyperparameters:** `learning_rate=0.05`, `num_leaves=64` (LightGBM), `max_depth=7` (XGBoost), `iterations` set to LightGBM's best iteration
* **Parameter Tuning:** Tested `learning_rate` in {0.01, 0.05, 0.1} via cross-validation

## 5. Results & Discussion

### 5.1. OOF Metrics

| Model    | Accuracy | Precision | Recall | F1     |
| -------- | -------- | --------- | ------ | ------ |
| Ensemble | 0.9527   | 0.9370    | 0.7157 | 0.8115 |

### 5.2. Inference Latency

| Model    | Time (ms) |
| -------- | --------- |
| LightGBM | 114.4     |
| XGBoost  | 77.7      |
| CatBoost | 36.6      |

### 5.3. Comparison to Baseline

Logistic Regression and Random Forest typically achieve 0.90–0.93 accuracy on similar data; our ensemble attains 0.9527.

### 5.4. Strengths

* **High Accuracy:** Achieves 95.27% OOF accuracy through intelligent feature engineering and model combination
* **Low Latency:** CatBoost inference at 36 ms enables real-time applications
* **Modular & Reproducible:** Clear separation of preprocessing, training, testing, and inference
* **State-of-the-Art Alignment:** Adopts top-performing GBDT algorithms for competitive results

## 6. Conclusion & Future Work

We present a reusable, scalable pipeline delivering 95.27% out-of-fold accuracy on the Playground Series S4E10 dataset. Future enhancements include pseudo-labeling, advanced feature interactions, and stacking meta-learners to further boost performance.

## 7. References

\[1] Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.

\[2] Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS*.

\[3] Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: unbiased boosting with categorical features. *NeurIPS*.

\[4] Wolpert, D. H. (1992). Stacked Generalization. *Neural Networks*, 5(2), 241–259.

## Appendix

* Complete code scripts available on GitHub repository: https://github.com/mr-r0ot/Playground-Series-S4E10-Kaggle
