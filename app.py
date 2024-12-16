import pandas as pd
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from joblib import load


# Set up page config and navbar
st.set_page_config(page_title="UAS Proyek Data Sains", layout="wide")
from component.nav import navbar
navbar()

# Load the pre-trained LSTM model
model = load_model('./lstm_model.keras')

# Load the previously fitted scaler
# This needs to be the scaler used during model training
scaler = load('scaler.pkl')

# Function for prediction
def predict(input_data):
    # Transform new data with the same scaler used in training
    data_scaled = scaler.transform(input_data)

    # Reshape data for LSTM (3D: samples, timesteps, features)
    data_scaled = data_scaled.reshape((data_scaled.shape[0], 1, data_scaled.shape[1]))

    # Predict with the LSTM model
    predictions = model.predict(data_scaled)

    # Interpret the prediction result
    predicted_class = np.argmax(predictions, axis=1)  # Get the class with the highest probability
    return predicted_class[0], predictions[0]

# Create the prediction form using Streamlit's form
with st.form(key='prediction_form'):
    st.markdown("### Enter the details to predict the class:")

    # Collect user input for the features
    gender = st.radio("What is your gender?", options=["Male", "Female"])
    age = st.number_input("What is your age?", min_value=1, max_value=120, value=25, step=1)
    height = st.number_input("What is your height? (cm)", min_value=50, max_value=250, value=170, step=1)
    weight = st.number_input("What is your weight? (kg)", min_value=30, max_value=200, value=70, step=1)
    family_history_with_overweight = st.radio("Has a family member suffered or suffers from overweight?", options=["Yes", "No"])
    FAVC = st.radio("Do you eat high caloric food frequently?", options=["Yes", "No"])
    FCVC = st.radio("Do you usually eat vegetables in your meals?", options=["Between 1 to 2", "Three", "More than Three"]) #
    NCP = st.radio("How many main meals do you have daily?", options=["No", "Sometimes", "Frequently", "Always"]) #
    CAEC = st.radio("Do you eat any food between meals?", options=["Yes", "No"])
    SMOKE = st.radio("Do you smoke?", options=["Yes", "No"])
    CH2O = st.radio("How much water do you drink daily?", options=["Less than a liter", "Between 1 and 2 liters", "More than 2 liters"])
    SCC = st.radio("Do you monitor the calories you eat daily?", options=["Yes", "No"])
    FAF = st.radio("How often do you have physical activity?", options=["I do not have", "1 or 2 days", "4 or 5 days"]) #
    TUE = st.radio("How much time do you use technological devices such as cell phone, videogames, television, computer and others?", options=["0-2 hours", "3-5 hours", "More than 5 hours"])
    CALC = st.radio("How often do you drink alcohol?", options=["I do not drink", "Sometimes", "Frequently", "Always"])
    MTRANS = st.radio("Which transportation do you usually use? ", options=["AutoMobile", "Bike", "MotorBike", "Public Transportation", "Walking"])

    # Submit button for prediction
    submit_button = st.form_submit_button(label="Predict")

    if submit_button:
        # Create input data in the same order as the training data
        input_data = np.array([[1 if gender == "Male" else 0,
                                age, height / 100.0, weight, 
                                1 if family_history_with_overweight == "Yes" else 0,
                                1 if FAVC == "Yes" else 0,
                                1 if FCVC == "Between 1 to 2" else (2 if FCVC == "Three" else 3),
                                1 if NCP == "No" else (2 if NCP == "Sometimes" else (3 if NCP == "Frequently" else 4)),
                                1 if CAEC == "Yes" else 0,
                                1 if SMOKE == "Yes" else 0,
                                1 if CH2O == "Less than a liter" else (2 if CH2O == "Between 1 and 2 liters" else 3),
                                1 if SCC == "Yes" else 0,
                                0 if FAF == "I do not have" else (1 if FAF == "1 or 2 days" else 2),
                                0 if TUE == "0-2 hours" else (1 if TUE == "3-5 hours" else 2),
                                1 if CALC == "I do not drink" else (2 if CALC == "Sometimes" else (3 if CALC == "Frequently" else 4)),
                                0 if MTRANS == "AutoMobile" else (1 if MTRANS == "Bike" else (2 if MTRANS == "Motorbike" else (3 if MTRANS == "Public Transportation" else 4)))
                              ]])

        # Ensure the input data has 16 features (categorical encoded properly)
        assert input_data.shape[1] == 16, f"Expected 16 features, but got {input_data.shape[1]} features."

        # Transform the input data with the scaler
        data_scaled = scaler.transform(input_data)

        # Reshape data for LSTM (3D: samples, timesteps, features)
        data_scaled = data_scaled.reshape((data_scaled.shape[0], 1, data_scaled.shape[1]))

        # Predict with the LSTM model
        predictions = model.predict(data_scaled)

        # Interpret the prediction result
        predicted_class = np.argmax(predictions, axis=1)

        # Display results
        st.markdown(f"### Prediction Result:")
        st.write(f"Predicted Class: {predicted_class[0]}")
        st.write(f"Prediction Probabilities: {predictions[0]}")

        # Map the predicted class to obesity categories
        obesity_map = {
            0: "Insufficient Weight",
            1: "Normal Weight",
            2: "Overweight Level I",
            3: "Overweight Level II",
            4: "Obesity Type I",
            5: "Obesity Type II",
            6: "Obesity Type III"
        }

        result = obesity_map.get(predicted_class[0], "Unknown")
        st.write(f"Prediction Result: {result}")
