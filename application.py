from flask import Flask, request, render_template,jsonify
from flask_cors import CORS,cross_origin
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

@app.route('/')
@cross_origin()
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
@cross_origin()
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(
            carat = float(request.form.get('carat')),
            depth = float(request.form.get('depth')),
            table = float(request.form.get('table')),
            x = float(request.form.get('x')),
            y = float(request.form.get('y')),
            z = float(request.form.get('z')),
            cut = request.form.get('cut'),
            color= request.form.get('color'),
            clarity = request.form.get('clarity')
        )

        pred_df = data.get_data_as_dataframe()
        
        print(pred_df)

        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(pred_df)
        results = round(pred[0],2)
        return render_template('index.html',results=results,pred_df = pred_df)
    
@app.route('/predictAPI',methods=['POST'])
@cross_origin()
def predict_api():
    if request.method=='POST':
        data = CustomData(
            carat = float(request.json['carat']),
            depth = float(request.json['depth']),
            table = float(request.json['table']),
            x = float(request.json['x']),
            y = float(request.json['y']),
            z = float(request.json['z']),
            cut = request.json['cut'],
            color = request.json['color'],
            clarity = request.json['clarity']
        )

        pred_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(pred_df)

        dct = {'price':round(pred[0],2)}
        return jsonify(dct)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
    
    
    
    
    
    
  #  streamlit previous code
    '''
import streamlit as st
import pandas as pd
import dill

# Load the preprocessor and model
with open('artifacts/preprocessor.pkl', 'rb') as file:
    preprocessor = dill.load(file)

with open('artifacts/model.pkl', 'rb') as file:
    model = dill.load(file)

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
        st.error(f'An error occurred: {str(e)}')
        '''