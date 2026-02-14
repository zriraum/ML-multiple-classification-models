# Machine Learning Assignment 2: Multiple Classification Models

**a. Problem statement:** 
The objective of this project is to implement, evaluate, and compare six different machine learning classification models on a chosen dataset. The trained models are deployed via an interactive Streamlit web application, allowing users to upload test data, select a model, and view evaluation metrics and confusion matrices in real-time.

**b. Dataset description:**
* **Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
* **Source:** UCI Machine Learning Repository (via sklearn.datasets)
* **Features:** 30 numerical features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image (e.g., radius, texture, perimeter, area, smoothness).
* **Instances:** 569
* **Target:** Binary classification predicting whether a tumor is Malignant (0) or Benign (1).

**c. Models used:** 
Below is the comparison of the evaluation metrics calculated for all 6 implemented models.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.974 | 0.997 | 0.972 | 0.986 | 0.979 | 0.944 |
| Decision Tree | 0.939 | 0.932 | 0.944 | 0.958 | 0.951 | 0.869 |
| KNN | 0.947 | 0.982 | 0.958 | 0.958 | 0.958 | 0.888 |
| Naive Bayes | 0.965 | 0.997 | 0.959 | 0.986 | 0.972 | 0.925 |
| Random Forest (Ensemble) | 0.956 | 0.994 | 0.958 | 0.972 | 0.965 | 0.906 |
| XGBoost (Ensemble) | 0.956 | 0.991 | 0.958 | 0.972 | 0.965 | 0.906 |

**Observations on Model Performance:**

| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Achieved the highest overall performance across almost all metrics (Accuracy: 0.974, F1: 0.979). The scaled dataset provided a perfect environment for gradient descent optimization, making this simple linear model the most effective for this specific task. |
| Decision Tree | Had the lowest overall metrics (Accuracy: 0.939). This is expected, as standalone decision trees are highly prone to overfitting on the training data, leading to slightly weaker generalization on the test set compared to ensemble methods. |
| KNN | Performed reliably well (Accuracy: 0.947) largely because the input features were properly normalized using StandardScaler, preventing features with larger numerical ranges from skewing the distance calculations. |
| Naive Bayes | Showed surprisingly strong results (Accuracy: 0.965, AUC: 0.997), indicating that the features in this dataset likely follow a normal distribution, fitting perfectly with the Gaussian Naive Bayes assumption. |
| Random Forest (Ensemble) | Delivered strong, balanced metrics (Accuracy: 0.956). By building multiple trees and averaging their predictions, it successfully mitigated the overfitting issues seen in the standalone Decision Tree. |
| XGBoost (Ensemble) | Performed identically to Random Forest on this test set (Accuracy: 0.956). While usually a more powerful algorithm, the relatively small size and simplicity of the Breast Cancer dataset meant the complex boosting technique didn't provide a massive advantage over standard bagging. |