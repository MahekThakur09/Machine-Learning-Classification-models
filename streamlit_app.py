import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


st.set_page_config(page_title="ML Classification App", layout="centered")

st.title("Machine Learning Classification App")
st.write("Upload a CSV file to evaluate different classification models.")

# ✅ ALWAYS visible
uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])


if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview")
    st.dataframe(data.head())

    target_column = st.selectbox("Select Target Column", data.columns)

    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_choice = st.selectbox(
        "Select Model",
        (
            "logistic_regression",
            "decision_tree",
            "knn",
            "naive_bayes",
            "random_forest",
            "xgboost"
        )
    )

    if st.button("Run Model"):

        # 🔥 Dynamic Import
        model_module = importlib.import_module(f"model.{model_choice}")

        y_pred, y_prob = model_module.run_model(X_train, X_test, y_train)

        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        mcc = matthews_corrcoef(y_test, y_pred)

        try:
            auc = roc_auc_score(y_test, y_prob) if y_prob is not None else "Not Available"
        except:
            auc = "Not Available"

        st.subheader("Evaluation Metrics")

        st.write(f"Accuracy: {accuracy:.4f}")
        st.write(f"Precision: {precision:.4f}")
        st.write(f"Recall: {recall:.4f}")
        st.write(f"F1 Score: {f1:.4f}")
        st.write(f"MCC: {mcc:.4f}")
        st.write(f"AUC: {auc}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # Classification Report
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred)
        st.text(report)

