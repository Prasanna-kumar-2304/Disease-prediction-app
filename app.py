import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("pima-data.csv")
data['diabetes'] = data['diabetes'].astype(int)  # Ensure binary values

# Split data into features and labels
feature_columns = ['num_preg', 'glucose_conc', 'diastolic_bp', 'insulin', 'bmi', 'diab_pred', 'age', 'skin']
X = data[feature_columns]
y = data['diabetes']

# Handle missing values
imputer = SimpleImputer(missing_values=0, strategy="mean")
X = imputer.fit_transform(X)

# Train the model
model = RandomForestClassifier(random_state=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
model.fit(X_train, y_train)

# Streamlit UI
st.title("Diabetes Prediction App with Visualizations")

# Input form
st.sidebar.header("Input Parameters")
num_preg = st.sidebar.number_input("Number of Pregnancies", min_value=0, value=1)
glucose_conc = st.sidebar.slider("Glucose Concentration", 0, 200, 120)
diastolic_bp = st.sidebar.slider("Diastolic Blood Pressure", 0, 140, 70)
insulin = st.sidebar.slider("Insulin", 0, 500, 80)
bmi = st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
diab_pred = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.sidebar.slider("Age", 0, 120, 25)
skin = st.sidebar.slider("Skin Thickness", 0, 100, 20)

# Submit Button
if st.sidebar.button("Submit"):
    # Prepare input data
    input_data = np.array([[num_preg, glucose_conc, diastolic_bp, insulin, bmi, diab_pred, age, skin]])
    prediction = model.predict(input_data)
    prediction_result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"

    # Display prediction
    st.subheader("Prediction Result")
    st.write(f"The patient is likely to be **{prediction_result}**.")

    # Visualizations
    st.subheader("Feature Importance")
    importances = model.feature_importances_
    fig, ax = plt.subplots()
    ax.barh(feature_columns, importances, color='teal')
    ax.set_xlabel("Feature Importance Score")
    ax.set_ylabel("Features")
    ax.set_title("Feature Importance Visualization")
    st.pyplot(fig)

    # Correlation heatmap
    st.subheader("Correlation Heatmap of Dataset")
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Histogram for a feature
    st.subheader("Distribution of Glucose Concentration")
    fig, ax = plt.subplots()
    sns.histplot(data['glucose_conc'], bins=20, kde=True, ax=ax)
    st.pyplot(fig)
