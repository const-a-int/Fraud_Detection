import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from .model import FraudDetector
from .config import Config

# Настройка логирования
logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Создание приложения
app = FastAPI(
    title="Fraud Detection API",
    description="API для обнаружения мошеннических транзакций",
    version="1.0.0"
)

# CORS для интеграции с внешними системами
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузка модели
detector = FraudDetector()
logger.info("Модель загружена")

@app.get("/")
def root():
    return {"message": "Fraud Detection API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
async def predict(transaction: dict):
    """
    Предсказание для одной транзакции
    Ожидает JSON с полями:
    {
        "amt": 150.00,
        "city_pop": 50000,
        "lat": 40.7128,
        "long": -74.0060,
        "merch_lat": 40.7580,
        "merch_long": -73.9855,
        "dob": "1990-01-01",
        "unix_time": 1609459200,
        "category": "grocery_pos",
        "gender": "M",
        "cc_num": 123456789  # опционально
    }
    """
    try:
        df = pd.DataFrame([transaction])
        
        proba = detector.predict_proba(df)[0]
        threshold = Config.DEFAULT_THRESHOLD
        is_fraud = bool(proba >= threshold)
        
        if proba < 0.3:
            risk_level = "low"
        elif proba < 0.7:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        logger.info(f"Prediction: proba={proba:.4f}, is_fraud={is_fraud}")
        
        return {
            "fraud_probability": round(proba, 4),
            "is_fraud": is_fraud,
            "threshold_used": threshold,
            "risk_level": risk_level
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(transactions: list):
    try:
        df = pd.DataFrame(transactions)
        probabilities = detector.predict_proba(df)
        
        return {
            "predictions": [
                {
                    "fraud_probability": round(float(p), 4),
                    "is_fraud": bool(p >= Config.DEFAULT_THRESHOLD)
                }
                for p in probabilities
            ]
        }
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_csv")
async def predict_csv(file: bytes):
    try:
        from io import BytesIO
        df = pd.read_csv(BytesIO(file))
        
        probabilities = detector.predict_proba(df)
        
        df['fraud_probability'] = probabilities
        df['is_fraud'] = probabilities >= Config.DEFAULT_THRESHOLD
        
        return df.to_dict(orient='records')
    except Exception as e:
        logger.error(f"CSV prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))