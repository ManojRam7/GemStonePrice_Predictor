import streamlit as st
import pandas as pd
import dill
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

try:
    # Load the preprocessor and model
    with open('artifacts/preprocessor.pkl', 'rb') as file:
        preprocessor = dill.load(file)
    logging.info("Preprocessor loaded successfully.")

    with open('artifacts/model.pkl', 'rb') as file:
        model = dill.load(file)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading preprocessor or model: {e}")
    st.error(f"Error loading preprocessor or model: {e}")

# Streamlit app
st.title('Gemstone Price Predictor')

carat = st.number_input('Carat')
cut = st.selectbox('Cut', ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
color = st.selectbox('Color', ['D', 'E', 'F', 'G', 'H', 'I', 'J'])
clarity = st.selectbox('Clarity', ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
depth = st.number_input('Depth')
table = st.number_input('Table')
x = st.number_input('X')
y = st.number_input('Y')
z = st.number_input('Z')

if st.button('Predict'):
    try:
        # Create input data DataFrame
        input_data = pd.DataFrame([[carat, cut, color, clarity, depth, table, x, y, z]], 
                                  columns=['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z'])
        
        # Make prediction
        scaled_data = preprocessor.transform(input_data)
        prediction = model.predict(scaled_data)
        
        st.success(f'Predicted Gemstone Price: ${prediction[0]:,.2f}')
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        st.error(f'An error occurred: {str(e)}')