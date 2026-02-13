# Classification Models & Streamlit Deployment

## a. Problem Statement

The objective of this project is to build, evaluate, and deploy multiple machine learning
classification models on a real-world dataset.  
An interactive Streamlit web application is developed to allow users to upload test data,
select models, and visualize performance metrics.

This project demonstrates an end-to-end ML workflow including:
- Data preprocessing
- Model training
- Evaluation using multiple metrics
- Web-based deployment using Streamlit Community Cloud

---

## b. Dataset Description

- **Dataset Name:** Heart Failure Clinical Records dataset
- **Source:** Kaggle / UCI Repository  
- **Problem Type:** Binary / Multi-class Classification  
- **Number of Instances:** ≥ 500  
- **Number of Features:** ≥ 12  
- **Target Variable:** <Target Column Name>

The dataset contains structured tabular data suitable for supervised classification tasks.

---

## c. Models Used & Evaluation Metrics

The following six classification models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (kNN)  
4. Naive Bayes (Gaussian / Multinomial)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

Each model was evaluated using the following metrics:
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

### 📊 Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------|---------|-----|-----------|--------|----------|-----|
| Logistic Regression |79.80|0.77|0.87|0.62|0.72|0.59|
| Decision Tree |68.68|0.67|0.67|0.52|0.59|0.35|
| Naive Bayes |56.56|0.72|0.91|00.48|0.62|0.52|
| kNN | 56.56 |0.51|0.47|0.17|0.25|0.04|
| Random Forest (Ensemble) |76.76|0.74|0.85|0.55|0.67|0.53|
| XGBoost (Ensemble) |76.76|0.74|0.81|0.60|0.68|0.52|

---

### 📈 Observations on Model Performance

| ML Model | Observation |
|--------|-------------|
| Logistic Regression | Performs well on linearly separable data but may underperform on complex patterns. |
| Decision Tree | Captures non-linear relationships but is prone to overfitting. |
| kNN | Sensitive to feature scaling and performs better with smaller datasets. |
| Naive Bayes | Fast and efficient but assumes feature independence. |
| Random Forest (Ensemble) | Provides strong performance with reduced overfitting due to averaging. |
| XGBoost (Ensemble) | Achieves the best performance due to boosting and regularization techniques. |

---

## 🚀 Streamlit Application Features

The deployed Streamlit web app includes:
- CSV test dataset upload option
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix / classification report visualization

---

## 🗂️ Project Structure
```
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- model/
│ │-- logistic_regression.pkl
│ │-- decision_tree.pkl
│ │-- knn.pkl
│ │-- naive_bayes.pkl
│ │-- random_forest.pkl
│ │-- xgboost.pkl

```
