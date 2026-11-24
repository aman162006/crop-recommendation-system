# AI-Powered Crop Recommendation System

Data-driven crop selection using soil nutrients (N, P, K, pH) and climate conditions
(temperature, humidity, rainfall). A supervised-learning pipeline (Random Forest) with a
Streamlit interface.

**Project Team**

| Name        | Roll No.          |
|-------------|-------------------|
| Aman Singh  | 202410101360217   |

## Project structure

crop_recommendation_system/
├── data/Crop_recommendation.csv   # 2,200 samples, 22 crops, 7 features
├── config.py                      # paths, features, valid ranges, team info
├── train.py                       # train + compare 5 algorithms, save best pipeline
├── predict.py                     # validation + inference (also usable as CLI)
├── app.py                         # Streamlit web app
├── tests/test_pipeline.py         # pytest unit tests
├── models/crop_model.joblib       # saved model (created by train.py)
└── reports/                       # metrics, confusion matrix, charts


## How to run

pip install -r requirements.txt
python train.py            # trains models, writes models/ and reports/
streamlit run app.py       # launches the web app

Optional:

python predict.py --N 90 --P 42 --K 43 --temperature 20.9 --humidity 82 --ph 6.5 --rainfall 203
pip install pytest && pytest tests/

## Results (test set, 20% stratified hold-out, 440 samples)

| Algorithm           | Accuracy | Macro F1 | 5-fold CV F1 |
|---------------------|---------:|---------:|-------------:|
| Random Forest       | 0.9932   | 0.9932   | 0.9937       |
| SVM                 | 0.9841   | 0.9840   | 0.9788       |
| Decision Tree       | 0.9795   | 0.9794   | 0.9852       |
| KNN                 | 0.9795   | 0.9793   | 0.9650       |
| Logistic Regression | 0.9727   | 0.9725   | 0.9678       |

Random Forest is deployed. Re-running `train.py` regenerates all numbers and plots in `reports/`.

## Disclaimer
Decision-support tool only. Validate recommendations with local agronomic expertise,
seasonality and market conditions.
