import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

st.title("🩺 Breast Cancer Classification Predictor")
st.markdown("This app predicts whether a breast mass is benign or malignant using various ML models.")


st.header("1. Download Test Dataset")
st.write("Evaluators: Download the test dataset here, then upload it in Step 2 to test the models.")
try:
    with open("breast_cancer_test.csv", "rb") as file:
        st.download_button(
            label="📥 Download breast_cancer_test.csv",
            data=file,
            file_name="breast_cancer_test.csv",
            mime="text/csv"
        )
except FileNotFoundError:
    st.warning(" Test file not found. Ensure 'breast_cancer_test.csv' is in the root directory.")


st.header("2. Upload Test Data")
uploaded_file = st.file_uploader("Upload your CSV file (must include 'target' column)", type="csv")

if uploaded_file is not None:
    
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    
    if 'target' not in df.columns:
        st.error("Error: The uploaded CSV must contain a 'target' column for evaluation.")
    else:
        
        X_test = df.drop(columns=['target'])
        y_test = df['target']

        
        try:
            scaler = pickle.load(open('model/scaler.pkl', 'rb'))
            X_test_scaled = scaler.transform(X_test)
        except FileNotFoundError:
            st.error("Scaler not found. Please ensure 'scaler.pkl' is in the 'model/' directory.")
            st.stop()

        
        st.header("3. Select Model & Run Inference")
        model_choice = st.selectbox(
            "Choose a Classification Model:",
            ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
        )

        if st.button("Generate Predictions & Metrics"):
            
            model_filename = f"model/{model_choice.replace(' ', '_').lower()}.pkl"
            try:
                model = pickle.load(open(model_filename, 'rb'))
            except FileNotFoundError:
                st.error(f"Model file {model_filename} not found!")
                st.stop()

            
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred

            
            st.subheader(" Evaluation Metrics")
            col1, col2, col3 = st.columns(3)
            col4, col5, col6 = st.columns(3)

            col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
            col2.metric("AUC Score", f"{roc_auc_score(y_test, y_prob):.3f}")
            col3.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
            col4.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")
            col5.metric("F1 Score", f"{f1_score(y_test, y_pred):.3f}")
            col6.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.3f}")

            
            st.subheader(" Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted Labels')
            ax.set_ylabel('True Labels')
            st.pyplot(fig)