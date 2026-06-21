# 💎 Gemstone Price Predictor

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-API-000000?logo=flask&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production%20Ready-1f9d55)
![License](https://img.shields.io/badge/License-MIT-blue)

A production-polished, end-to-end machine learning project that predicts **gemstone prices** from physical and quality attributes.

## ✨ Highlights

- ✅ End-to-end ML pipeline: ingestion → transformation → training → inference
- ✅ Model selection with tuned ensemble-style tree-based regressors
- ✅ Two serving interfaces:
  - Streamlit app for demos
  - Flask app + JSON API for integration
- ✅ Clean project structure and reproducible artifacts
- ✅ Portfolio-grade documentation and UI polish

## 🧱 Project Structure

```text
GemStonePrice_Predictor/
├── application.py                 # Flask app entrypoint
├── streamlit_app.py               # Streamlit app entrypoint
├── requirements.txt
├── setup.py
├── artifacts/                     # Generated model/preprocessor/data artifacts
├── notebook/                      # EDA, modeling, explainability notebooks
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   ├── predict_pipeline.py
│   │   └── train_pipeline.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
├── static/
│   └── css/style.css
└── templates/
    └── index.html
```

## 📊 Dataset

- Source: Kaggle Playground Series S3E8 (diamond/gemstone pricing)
- Expected columns:
  - Features: `carat`, `depth`, `table`, `x`, `y`, `z`, `cut`, `color`, `clarity`
  - Target: `price`
- Default data path used in training:
  - `notebook/data/gemstone.csv`

## ⚙️ Tech Stack

- **Core ML**: scikit-learn, CatBoost, XGBoost
- **Data**: pandas, numpy
- **Serving**: Flask, Streamlit
- **Serialization**: joblib
- **Explainability**: LIME (notebook workflow)

## 🚀 Quickstart

### 1. Clone and setup

```bash
git clone https://github.com/ManojRam7/GemStonePrice_Predictor.git
cd GemStonePrice_Predictor
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Train pipeline

```bash
python -m src.pipeline.train_pipeline
```

This creates/updates:

- `artifacts/data.csv`
- `artifacts/train.csv`
- `artifacts/test.csv`
- `artifacts/preprocessor.pkl`
- `artifacts/model.pkl`

### 3. Run Streamlit app

```bash
streamlit run streamlit_app.py
```

### 4. Run Flask app

```bash
python application.py
```

- UI endpoint: `http://127.0.0.1:8000/`
- API endpoint: `POST /predictAPI`

Sample JSON payload:

```json
{
  "carat": 0.8,
  "depth": 62.0,
  "table": 57.0,
  "x": 5.7,
  "y": 5.7,
  "z": 3.5,
  "cut": "Ideal",
  "color": "F",
  "clarity": "VS1"
}
```

## 🧠 ML Workflow

1. **Data Ingestion**
   - Loads source CSV and creates train/test splits.
2. **Data Transformation**
   - Numeric: median imputation + scaling.
   - Categorical: frequent imputation + ordinal encoding + scaling.
3. **Model Training**
   - Trains and tunes multiple tree-based regressors.
   - Selects best model by test-set R².
4. **Inference Pipeline**
   - Applies saved preprocessor and model for predictions.

## 📘 Notebooks

- EDA: `notebook/1_EDA_Gemstone_price.ipynb`
- Model experiments: `notebook/2_Model_Training_Gemstone.ipynb`
- Explainability: `notebook/3_Explainability_with_LIME.ipynb`

## 🧩 Professional Improvements Applied

- Removed stale/legacy code blocks and accidental duplicate scripts
- Added cleaner logging and exception handling
- Replaced brittle serialization approach with joblib
- Refactored training and prediction into maintainable pipelines
- Modernized dependencies and package metadata
- Upgraded frontend styling and responsive UX

## 📌 Portfolio Tips

- Add screenshots/GIFs of Streamlit + Flask UI under a new `assets/` folder
- Publish demo on Streamlit Community Cloud
- Add CI for lint + tests (pytest + ruff) as a next enhancement

---

If you found this useful, give it a ⭐ and use it as a template for production-ready regression projects.
