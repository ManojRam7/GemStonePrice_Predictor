import streamlit as st
import pandas as pd
import os
from src.pipeline.predict_pipeline import PredictPipeline

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
    try:
        # Create input DataFrame
        input_data = pd.DataFrame({
            "carat": [carat],
            "depth": [depth],
            "table": [table],
            "x": [x],
            "y": [y],
            "z": [z],
            "cut": [cut],
            "color": [color],
            "clarity": [clarity]
        })

        # Dynamically adjust the working directory
        current_dir = os.getcwd()
        artifact_dir = os.path.join(current_dir, "artifacts")

        # Check if the artifacts directory exists
        if not os.path.exists(artifact_dir):
            st.error(f"Artifacts directory not found: {artifact_dir}")
        else:
            st.write(f"Using artifacts from: {artifact_dir}")
        
        # Temporarily change directory to resolve paths
        os.chdir(current_dir)

        # Load prediction pipeline
        predict_pipeline = PredictPipeline()

        # Perform prediction
        prediction = predict_pipeline.predict(input_data)

        # Display prediction
        st.success(f"Predicted Gemstone Price is: ${round(prediction[0], 2)}")
    except Exception as e:
        st.error(f"Error occurred during prediction: {e}")
