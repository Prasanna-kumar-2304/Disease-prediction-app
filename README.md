# Diabetes Prediction App with Machine Learning

## Overview

This project aims to build a **Diabetes Prediction App** using machine learning algorithms. The application leverages data from the **Pima Indians Diabetes Database** to predict whether a person has diabetes based on their medical attributes. The app is built using **Random Forest Classifier** and **XGBoost** models, with **Streamlit** as the user interface. It also provides insightful visualizations such as feature importance and correlation heatmaps to better understand the model's predictions.

---

## Project Features

1. **Data Preprocessing:**
   - The dataset is cleaned by replacing zeros with NaN in certain columns (e.g., glucose concentration, insulin, BMI, etc.).
   - Missing values are imputed using the mean of the respective columns.
   - The target column (`diabetes`) is encoded into binary values: 1 for diabetic and 0 for non-diabetic.

2. **Machine Learning Models:**
   - **Random Forest Classifier** and **XGBoost** models are used to train the data and make predictions.
   - Hyperparameter tuning of the XGBoost model is performed using **RandomizedSearchCV**.
   - Model evaluation includes metrics like accuracy, confusion matrix, classification report, and feature importance.

3. **Streamlit App:**
   - A user-friendly interface built using **Streamlit**.
   - Users can input various health parameters such as glucose concentration, BMI, age, and more.
   - The model predicts whether the user is diabetic or non-diabetic based on the provided inputs.
   - Visualizations include:
     - **Feature Importance**: A bar plot showing the importance of different features in the prediction.
     - **Correlation Heatmap**: A heatmap illustrating the correlation between different features.
     - **Glucose Distribution**: A histogram showing the distribution of glucose levels in the dataset.

---

## Prerequisites

To run this project locally, you'll need to install the following libraries:

- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical operations.
- **seaborn**: For data visualization.
- **matplotlib**: For creating plots.
- **scikit-learn**: For machine learning models and evaluation.
- **xgboost**: For training the XGBoost model.
- **streamlit**: For the app's frontend interface.

You can install these libraries using **pip**:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost streamlit
```

---

## Files

- **diabetes_prediction.ipynb**: The Jupyter notebook where data preprocessing, model training, and evaluation are done.
- **pima-data.csv**: The dataset used for the machine learning model.
- **app.py**: The Streamlit application that allows users to interact with the model and get predictions.
- **requirements.txt**: A list of dependencies for the project.

---


## Results

- **Random Forest Model Accuracy**: 74.03%
- **XGBoost Model Accuracy**: 74.89%
- **Confusion Matrices and Classification Reports**: Provided for both models to compare their performance.
- **Feature Importance Visualization**: Shows which features are most important in determining whether a person has diabetes.

---


## Acknowledgments

- The dataset used in this project is from the **Pima Indians Diabetes Database**.
- Thanks to the open-source libraries like **scikit-learn**, **Streamlit**, and **xgboost** that made this project possible.
- The feature importance visualizations were created using **matplotlib** and **seaborn**.

---
