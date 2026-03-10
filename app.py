import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the preprocessor and model
preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("model.pkl")

def main():
    st.title('Heart Attack Prediction Model Deployment')

    # Add user input components for 13 features with sensible ranges
    age = st.number_input('Age', min_value=0, max_value=120, value=50)
    sex = st.number_input('Sex', min_value=0, max_value=1, value=1)
    cp = st.number_input('Chest Pain Type (cp)', min_value=0, max_value=3, value=1)
    trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=50, max_value=250, value=120)
    chol = st.number_input('Serum Cholestoral (chol)', min_value=100, max_value=600, value=200)
    fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl (fbs)', min_value=0, max_value=1, value=0)
    restecg = st.number_input('Resting Electrocardiographic Results (restecg)', min_value=0, max_value=2, value=1)
    thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', min_value=50, max_value=250, value=150)
    exang = st.number_input('Exercise Induced Angina (exang)', min_value=0, max_value=1, value=0)
    oldpeak = st.number_input('ST Depression (oldpeak)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.number_input('Slope of the Peak Exercise ST Segment (slope)', min_value=0, max_value=2, value=1)
    ca = st.number_input('Number of Major Vessels (ca)', min_value=0, max_value=4, value=0)
    thal = st.number_input('Thalassemia (thal)', min_value=0, max_value=3, value=1)
    
    if st.button('Make Prediction'):
        features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        result = make_prediction(features)
        st.success(f'The prediction is: {result}')

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    input_df = pd.DataFrame(input_array, columns=columns)
    
    X = preprocessor.transform(input_df)
    prediction = model.predict(X)
    return prediction[0]

if __name__ == '__main__':
    main()

