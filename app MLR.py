# -*- coding: utf-8 -*-
"""
MLR Model Deployment for Predicting Income Based on Age & Experience
"""

import numpy as np
import pickle
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# Load the trained MLR model
loaded_model = pickle.load(open('model', 'rb'))

# Prediction function
def predict_income(age, experience):
    input_data = np.array([[age, experience]])  # Convert input to a 2D array
    prediction = loaded_model.predict(input_data)
    return prediction[0]  # Return the predicted income

# Streamlit UI
def main():
    st.title("Income Prediction Based on Age & Experience")
    st.write("Enter Age and Years of Experience to predict the estimated Income.")

    # User Inputs
    age = st.number_input("Enter Age", min_value=18, step=1)
    experience = st.number_input("Enter Years of Experience", min_value=0, step=1)

    # Predict button
    if st.button("Predict Income"):
        predicted_income = predict_income(age, experience)
        st.success(f"Predicted Income: ${predicted_income:.2f}")

if __name__ == "__main__":
    main()
