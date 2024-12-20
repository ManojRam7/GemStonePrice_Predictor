import streamlit as st
import pandas as pd
from some_module import PredictPipeline, data  # Replace 'some_module' with your actual module

# Title of the app
st.title("Gemstone Price Prediction")

# Input fields
carat = st.number_input("Enter carat value (float)", step=0.01, format="%.2f")
depth = st.number_input("Enter depth value (float)", step=0.01, format="%.2f")
table = st.number_input("Enter table value (float)", step=0.01, format="%.2f")
x = st.number_input("Enter x value (float)", step=0.01, format="%.2f")
y = st.number_input("Enter y value (float)", step=0.01, format="%.2f")
z = st.number_input("Enter z value (float)", step=0.01, format="%.2f")
cut = st.selectbox("Select cut", options=["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.selectbox("Select color", options=["D", "E", "F", "G", "H", "I", "J"])
clarity = st.selectbox("Select clarity", options=["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"])

# Prediction button
if st.button("Submit"):
    # Prepare data as a DataFrame
    input_data = pd.DataFrame(
        {
            "carat": [carat],
            "depth": [depth],
            "table": [table],
            "x": [x],
            "y": [y],
            "z": [z],
            "cut": [cut],
            "color": [color],
            "clarity": [clarity]
        }
    )
    
    # Run prediction
    predict_pipeline = PredictPipeline()
    prediction = predict_pipeline.predict(input_data)
    
    # Display prediction
    st.success(f"Predicted Gemstone Price is: ${round(prediction[0], 2)}")
