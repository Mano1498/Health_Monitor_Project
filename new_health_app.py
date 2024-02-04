# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 15:49:37 2024

@author: ranja
"""

import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

import sklearn
print(sklearn.__version__)

# Load the trained model and scaler
sc = joblib.load('hstd.joblib')
dtc = joblib.load('hdtc.joblib')  # Replace with the actual filename of your model

# Function to preprocess and predict
def calculate_bmi(weight, height):
    # BMI Formula: weight (kg) / (height (m))^2
    height_in_meters = height / 100  # Convert height from centimeters to meters
    bmi = weight / (height_in_meters ** 2)
    return bmi

# Function to preprocess and predict
def predict(input_data):
    mapping = {'Yes': 1, 'No': 0, 'Public Transport': 1, 'Own Vehicle': 0, 'Walk': 2,
               'Afternoon Shift': 0, 'Evening Shift': 1, 'General Shift': 2, 'Multiple Shifts': 3,
               'Night Shift': 4, 'EXTREME STRESS': 0, 'HIGH STRESS': 1, 'LOW STRESS': 2,
               'MINIMAL STRESS': 3, 'MODERATE STRESS': 4}
    # Calculate BMI
    bmi = calculate_bmi(input_data[0], input_data[1])

    # Update the mapping and encoded_data based on the new features
    encoded_data = (
        input_data[0],
        bmi,
        input_data[1],
        mapping[input_data[2]],
        mapping[input_data[3]],
        mapping[input_data[4]],
        input_data[5],
        mapping[input_data[6]],
        mapping[input_data[7]],
        mapping[input_data[8]]
    )

    input_data_as_numpy_array = np.asarray(encoded_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    std_data = sc.transform(input_data_reshaped)

    prediction = dtc.predict(std_data)
    
    # Return prediction and advice
    if prediction[0] == 'Not Healthy':
        return "Not Healthy", "Consider consulting with a healthcare professional for further guidance."
    else:
        return "Healthy", "Congratulations! Keep up the good work and maintain a healthy lifestyle."

# Streamlit app
def main():
    st.title("Health Monitor App")

    # Input fields
    weight = st.number_input("Body weight in Kg")
    
    height = st.number_input("Height in centimeters")

    mode_of_travel = st.selectbox("Mode of Travel", ['Own Vehicle', 'Public Transport', 'Walk'])

    medical_history = st.radio("Medical History", ['Yes', 'No'])

    covid_infected = st.radio("Were you infected by COVID-19", ['Yes', 'No'])

    physical_activity = st.number_input("How often you do physical activity in a week?")

    work_type = st.selectbox("Work Type", ['Afternoon Shift', 'Evening Shift', 'General Shift', 'Multiple Shifts', 'Night Shift'])
    
    stress_level = st.selectbox("Stress Level", ['EXTREME STRESS', 'HIGH STRESS', 'LOW STRESS', 'MINIMAL STRESS', 'MODERATE STRESS'])

    input_data = (weight, height, mode_of_travel, medical_history, covid_infected, physical_activity, work_type, stress_level, stress_level)


    if st.button("Predict"):
        prediction , advice  = predict(input_data)
        st.success(f"The predicted outcome is: {prediction}")
        st.info(f"Advice: {advice}")

if __name__ == '__main__':
    main()
    
    
import sklearn
import streamlit
import numpy
import joblib

print("scikit-learn version:", sklearn.__version__)
print("Streamlit version:", streamlit.__version__)
print("NumPy version:", numpy.__version__)
print("Joblib version:", joblib.__version__)

    


