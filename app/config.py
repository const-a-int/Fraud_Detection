import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MODEL_PATH = os.getenv("MODEL_PATH", "models/fraud_model.pkl")
    SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")
    LE_CAT_PATH = os.getenv("LE_CAT_PATH", "models/le_cat.pkl")
    LE_GEN_PATH = os.getenv("LE_GEN_PATH", "models/le_gen.pkl")
    
    DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.5"))
    
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    PORT = int(os.getenv("PORT", "8000"))