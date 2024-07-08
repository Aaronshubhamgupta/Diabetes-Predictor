import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
def load_data():
    # Load the PIMA Diabetes dataset
    diabetics_dataset = pd.read_csv(r"C:\Users\aaron\OneDrive\Desktop\projects\undertest\Diabetes-Prediction\diabetes.csv")
    # Separate features and target
    x = diabetics_dataset.drop(columns="Outcome", axis=1)
    y = diabetics_dataset["Outcome"]
    # Standardize the features
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(x)
    return standardized_data, y

# Train the models
def train_models(x, y):
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)
    
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(random_state=0),
        'Gradient Boosting': GradientBoostingClassifier(random_state=0)
    }

    trained_models = {}
    accuracies = {}

    # Train each model
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_pred, y_test)
        trained_models[name] = model
        accuracies[name] = accuracy

    return trained_models, accuracies

# Streamlit App
def main():
    # Title and description
    st.title('Diabetes Prediction')
    st.write('This app predicts whether a person is likely to have diabetes based on input features.')

    # Load data
    x, y = load_data()

    # Train models (outside of caching to ensure re-training on every run)
    models, accuracies = train_models(x, y)

    # Display model accuracies
    st.subheader('Model Accuracies')
    for name, accuracy in accuracies.items():
        st.write(f'{name}: Accuracy on test data: {accuracy:.2%}')

    # User input for prediction
    st.sidebar.header('User Input Features')
    pregnancies = st.sidebar.slider('Pregnancies', 0, 20, 0)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    blood_pressure = st.sidebar.slider('Blood Pressure', 0, 140, 70)
    skin_thickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 900, 80)
    bmi = st.sidebar.slider('BMI', 0.0, 70.0, 25.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.5)
    age = st.sidebar.slider('Age', 0, 120, 30)

    # Make predictions for each model
    user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

    # Display predictions and probabilities in columns
    st.subheader('Predictions and Probabilities')
    for name, model in models.items():
        prediction = model.predict(user_input)
        prediction_proba = model.predict_proba(user_input)

        # Display results in columns
        col1, col2 = st.columns(2)
        with col1:
            st.write(name)
        with col2:
            st.write(f'{prediction_proba[0][1]:.2f}')

if __name__ == '__main__':
    main()
