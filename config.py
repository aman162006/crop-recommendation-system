
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Crop_recommendation.csv"
MODEL_PATH = BASE_DIR / "models" / "crop_model.joblib"
REPORT_DIR = BASE_DIR / "reports"

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_VERSION = "v1.0"


FEATURE_INFO = {
    "N":           (0.0, 140.0, 90.0, 1.0,  "Nitrogen (N)"),
    "P":           (5.0, 145.0, 42.0, 1.0,  "Phosphorus (P)"),
    "K":           (5.0, 205.0, 43.0, 1.0,  "Potassium (K)"),
    "temperature": (5.0, 50.0, 25.5, 0.1,   "Temperature (°C)"),
    "humidity":    (10.0, 100.0, 80.0, 0.1, "Humidity (%)"),
    "ph":          (3.0, 10.0, 6.5, 0.1,    "Soil pH"),
    "rainfall":    (15.0, 300.0, 200.0, 1.0, "Rainfall (mm)"),
}

AUTHORS = [
    ("Aman Singh", "202410101360217"),
    
]
