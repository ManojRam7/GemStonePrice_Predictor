import pandas as pd
import streamlit as st

from src.pipeline.predict_pipeline import PredictPipeline

st.set_page_config(page_title="Gemstone Price Predictor", page_icon="💎", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;800&display=swap');
    html, body, [class*="css"] {
      font-family: 'Manrope', sans-serif;
    }
    .hero {
      background: linear-gradient(135deg, #062925 0%, #0b4d43 45%, #f3b562 100%);
      color: #fff9ef;
      border-radius: 18px;
      padding: 1.5rem;
      margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>💎 Gemstone Price Predictor</h1>
      <p>Production-ready inference app built on a tuned ensemble workflow.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1])
with left:
    carat = st.number_input("Carat", min_value=0.1, max_value=10.0, value=0.8, step=0.01)
    depth = st.number_input("Depth", min_value=40.0, max_value=80.0, value=62.0, step=0.1)
    table = st.number_input("Table", min_value=40.0, max_value=90.0, value=57.0, step=0.1)
    x = st.number_input("x", min_value=0.0, max_value=20.0, value=5.7, step=0.01)
    y = st.number_input("y", min_value=0.0, max_value=20.0, value=5.7, step=0.01)
    z = st.number_input("z", min_value=0.0, max_value=20.0, value=3.5, step=0.01)

with right:
    cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"], index=4)
    color = st.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"], index=2)
    clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"], index=4)

if st.button("Predict Price", type="primary", use_container_width=True):
    try:
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
                "clarity": [clarity],
            }
        )
        prediction = PredictPipeline().predict(input_data)
        st.success(f"Estimated Price: ${float(prediction[0]):,.2f}")
        st.dataframe(input_data, use_container_width=True)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
