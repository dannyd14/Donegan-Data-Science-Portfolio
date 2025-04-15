
#Import all necessary libraries 

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from sklearn.datasets import load_iris, load_diabetes

#Create the title and set page configuratuons
st.set_page_config(page_title="ML App", layout="wide")
st.title("🤖 Interactive Machine Learning App")

# Sidebar layout
st.sidebar.title("🛠️ Settings")
data_source = st.sidebar.radio("📂 Choose your data source", ["Upload your own", "Use a sample dataset"])

#conditional statment that checks if the user wants to upload their own dataset or use a sample dataset
if data_source == "Upload your own":
    uploaded_file = st.sidebar.file_uploader("📄 Upload your CSV dataset", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
else:
    sample_name = st.sidebar.selectbox("📁 Select a sample dataset", ["Iris (classification)", "Diabetes (regression)", "Titanic (classification)"])
    if sample_name == "Iris (classification)":
        iris = load_iris(as_frame=True)
        df = iris.frame
        target_column = "target"
    elif sample_name == "Diabetes (regression)":
        diabetes = load_diabetes(as_frame=True)
        df = diabetes.frame
        target_column = "target"
    elif sample_name == "Titanic (classification)":
        df = sns.load_dataset("titanic")
        target_column = "survived"

# If a dataset has been loaded, display it and preprocess
if 'df' in locals():
    df.dropna(inplace=True)
    st.success(f"✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns (nulls dropped)")

    # Display dataset preview
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())
    #task selections
    task = st.sidebar.radio("📌 Task", ["Classification", "Regression"])
    target_column = st.sidebar.selectbox("🎯 Select Target Column", df.columns)

    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    feature_columns = [col for col in numeric_columns if col != target_column]

    valid_data = False

    if task == "Classification":
        if df[target_column].dtype != 'object' and df[target_column].nunique() > 20:
            st.warning("⚠️ Target seems continuous — consider Regression.")
        else:
            y = df[target_column].astype('category').cat.codes
            X = df[feature_columns]
            valid_data = True

    elif task == "Regression":
        if df[target_column].dtype in ['int64', 'float64']:
            y = df[target_column]
            X = df[feature_columns]
            valid_data = True
        else:
            st.error("❌ Target must be numeric for regression.")

    if valid_data:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.sidebar.markdown("### 🤖 Model Selection")

        # Model Selection, conditional statement based on type of regression 
        if task == "Classification":
            model_name = st.sidebar.selectbox("Choose Classifier", ["Logistic Regression", "Decision Tree", "K Nearest Neighbors"])

            if model_name == "Logistic Regression":
                c = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0, key="logreg_c")
                max_iter = st.sidebar.slider("Max Iterations", 100, 1000, 300, key="logreg_iter")
                model = LogisticRegression(C=c, max_iter=max_iter)

            elif model_name == "Decision Tree":
                max_depth = st.sidebar.slider("Max Depth", 1, 50, 5, key="clf_tree_depth")
                min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, 2, key="clf_tree_split")
                model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)

            elif model_name == "K Nearest Neighbors":
                n_neighbors = st.sidebar.slider("Number of Neighbors", 1, 20, 5, key="clf_knn_k")
                model = KNeighborsClassifier(n_neighbors=n_neighbors)

        else:  # Regression
            model_name = st.sidebar.selectbox("Choose Regressor", ["Linear Regression", "Decision Tree", "K Nearest Neighbors"])

            if model_name == "Linear Regression":
                model = LinearRegression()

            elif model_name == "Decision Tree":
                max_depth = st.sidebar.slider("Max Depth", 1, 50, 5, key="reg_tree_depth")
                min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, 2, key="reg_tree_split")
                model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split)

            elif model_name == "K Nearest Neighbors":
                n_neighbors = st.sidebar.slider("Number of Neighbors", 1, 20, 5, key="reg_knn_k")
                model = KNeighborsRegressor(n_neighbors=n_neighbors)
        #Training model button development
        if st.sidebar.button("🚀 Train Model"):
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            st.markdown("## 🧪 Results")
            tabs = st.tabs(["📈 Evaluation", "📊 Confusion Matrix", "🔍 ROC / PR Curve", "💡 Feature / Residuals"])
          #Model evaluation tabs  
            with tabs[0]:
                st.subheader("📈 Model Evaluation")
                st.markdown("""
                **What you're seeing:**  
                These are key performance metrics for your model.  
                - **Accuracy** tells you what percentage of predictions were correct (for classification).  
                - **Mean Squared Error (MSE)** measures average squared difference between actual and predicted values (for regression).  
                - **R² Score** explains how much of the variability in the data the model accounts for.
                """)
                if task == "Classification":
                    acc = accuracy_score(y_test, predictions)
                    st.write(f"**Accuracy:** {acc:.2f}")
                    st.text("Classification Report:")
                    st.text(classification_report(y_test, predictions))
                else:
                    mse = mean_squared_error(y_test, predictions)
                    r2 = r2_score(y_test, predictions)
                    st.write(f"**Mean Squared Error:** {mse:.2f}")
                    st.write(f"**R² Score:** {r2:.2f}")

            with tabs[1]:
                if task == "Classification":
                    st.subheader("📊 Confusion Matrix")
                    st.markdown("""
                    **What you're seeing:**  
                    This confusion matrix shows how well your model is classifying outcomes.  
                    - Rows represent the actual classes.  
                    - Columns represent the predicted classes.  
                    - Diagonal values are correct predictions; off-diagonal values are mistakes.
                    """)
                    cm = confusion_matrix(y_test, predictions)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    st.pyplot(fig)
                else:
                    st.info("Confusion matrix only applies to classification.")

            with tabs[2]:
                if task == "Classification" and len(y_test.unique()) == 2:
                    st.markdown("""
                    **What you're seeing:**  
                    These curves help evaluate how well your model distinguishes between the two classes (binary classification only).  

                    - **ROC Curve (Receiver Operating Characteristic):**  
                      Plots True Positive Rate vs. False Positive Rate. The closer the curve is to the top-left, the better.  
                      AUC (Area Under Curve) summarizes this performance.

                    - **Precision-Recall Curve:**  
                      Plots Precision vs. Recall, especially useful for imbalanced datasets.  
                      Average Precision gives a single-number summary of model performance.
                    """)
                    if hasattr(model, "predict_proba"):
                        y_proba = model.predict_proba(X_test)[:, 1]
                    else:
                        y_proba = model.decision_function(X_test)

                    # ROC Curve
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    auc_score = roc_auc_score(y_test, y_proba)
                    fig_roc, ax_roc = plt.subplots()
                    ax_roc.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
                    ax_roc.plot([0, 1], [0, 1], 'k--')
                    ax_roc.set_xlabel("False Positive Rate")
                    ax_roc.set_ylabel("True Positive Rate")
                    ax_roc.set_title("ROC Curve")
                    ax_roc.legend()
                    st.pyplot(fig_roc)

                    # PR Curve
                    precision, recall, _ = precision_recall_curve(y_test, y_proba)
                    avg_precision = average_precision_score(y_test, y_proba)
                    fig_pr, ax_pr = plt.subplots()
                    ax_pr.plot(recall, precision, label=f"AP = {avg_precision:.2f}")
                    ax_pr.set_xlabel("Recall")
                    ax_pr.set_ylabel("Precision")
                    ax_pr.set_title("Precision-Recall Curve")
                    ax_pr.legend()
                    st.pyplot(fig_pr)
                else:
                    st.info("ROC/PR curves only apply to binary classification.")

            with tabs[3]:
                if task == "Classification" and hasattr(model, "feature_importances_"):
                    st.subheader("💡 Feature Importance")
                    st.markdown("""
                    **What you're seeing:**  
                    This bar chart shows which features were most important for the model’s decisions (tree-based models only).  
                    Larger bars mean the feature had more influence on the predictions.
                    """)
                    importance = model.feature_importances_
                    sorted_idx = importance.argsort()
                    fig, ax = plt.subplots()
                    ax.barh([feature_columns[i] for i in sorted_idx], importance[sorted_idx])
                    ax.set_title("Feature Importance")
                    st.pyplot(fig)
                elif task == "Regression":
                    st.subheader("📉 Residual Plot")
                    st.markdown("""
                    **What you're seeing:**  
                    This residual plot shows the difference between actual and predicted values.  
                    - Residuals should ideally be randomly scattered around zero.  
                    - Patterns might indicate issues like non-linearity or heteroscedasticity.
                    """)
                    residuals = y_test - predictions
                    fig, ax = plt.subplots()
                    ax.scatter(predictions, residuals)
                    ax.axhline(0, color='red', linestyle='--')
                    ax.set_xlabel("Predicted Values")
                    ax.set_ylabel("Residuals")
                    ax.set_title("Residual Plot")
                    st.pyplot(fig)
                else:
                    st.info("Feature importance only applies to tree-based classifiers.")
else:
    st.info("📄 Upload a CSV file or use a sample dataset to get started.")

#To run the app, type `streamlit run MLStreamlitApp/Streamlit-App.py` in the terminal.

