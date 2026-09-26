# Gemstone Price Analyser

Predicts the price of a gemstone from its carat, cut, colour, clarity, depth, table and x, y, z
dimensions. A modular training pipeline (ingestion, transformation, model selection) tunes several
regressors and keeps the best one by test R²; predictions are served through a Flask app with a
JSON endpoint and through a Streamlit app.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7)
![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

## Results

Kaggle Playground Series S3E8 data, 193,573 gemstones, 80/20 split (154,858 / 38,715):

| Model | Test R² | Test RMSE | Test MAE |
|---|---|---|---|
| **CatBoost (tuned)** | **0.9795** | **575** | **295** |
| XGBoost (tuned) | 0.9794 | 576 | 293 |
| Voting ensemble (CatBoost + KNN + XGBoost) | 0.9795 | 575 | 293 |
| KNN (tuned, k = 16) | 0.9743 | 644 | 336 |
| Random forest | 0.9771 | | |
| Linear regression | 0.9373 | 1,007 | 672 |

Tuning moved CatBoost from 0.9792 to 0.9795; the voting ensemble matched it without improving on it,
so the single tuned CatBoost model is the simpler choice.

## Approach

- **EDA** (`notebook/1_EDA_Gemstone_price.ipynb`): no missing values or duplicates; cut, colour and
  clarity are ordinal, so they are mapped to ordered integers; mutual information shows carat and
  the x, y, z dimensions carry most of the signal.
- **Modelling** (`notebook/2_Model_Training_Gemstone.ipynb`): nine regressors compared, then
  CatBoost, KNN and XGBoost tuned with randomised and grid search, and a voting ensemble tried.
- **Explainability** (`notebook/3_Explainability_with_LIME.ipynb`): LIME explanations of individual
  price predictions.
- **Pipeline** (`src/`): `data_ingestion.py` splits the data, `data_transformation.py` builds the
  preprocessing (ordinal encoding and scaling), `model_trainer.py` tunes the candidates and saves the
  best model and preprocessor to `artifacts/`.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.pipeline.train_pipeline     # retrain; writes artifacts/model.pkl and preprocessor.pkl
streamlit run streamlit_app.py             # Streamlit app
python application.py                      # Flask app on http://127.0.0.1:8000
```

JSON endpoint:

```bash
curl -X POST http://127.0.0.1:8000/predictAPI -H "Content-Type: application/json" \
  -d '{"carat": 1.52, "depth": 62.2, "table": 58.0, "x": 7.27, "y": 7.33, "z": 4.55,
       "cut": "Premium", "color": "F", "clarity": "VS2"}'
```

## Project structure

```text
application.py               Flask app: form at /, JSON at /predictAPI
streamlit_app.py             Streamlit app
src/components/              data_ingestion, data_transformation, model_trainer
src/pipeline/                train_pipeline, predict_pipeline
src/utils.py, logger.py, exception.py
notebook/                    EDA, model training, LIME explainability, data
artifacts/                   trained model and preprocessor
templates/, static/          Flask front end
```
