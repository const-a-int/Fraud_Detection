import numpy as np
import pandas as pd
import joblib
from typing import Tuple
from .config import Config
import os

class FraudDetector:
    def __init__(self):
        print(f"Loading model from: {os.path.abspath(Config.MODEL_PATH)}")
        print(f"File size: {os.path.getsize(Config.MODEL_PATH)} bytes")
        
        self.model = joblib.load(Config.MODEL_PATH)
        self.scaler = joblib.load(Config.SCALER_PATH)
        self.le_cat = joblib.load(Config.LE_CAT_PATH)
        self.le_gen = joblib.load(Config.LE_GEN_PATH)
        
        print(f"Model type: {type(self.model)}")
        print(f"Expected features count: {self.model.n_features_in_}")
        
        self.numeric_features = ['amt', 'city_pop', 'distance', 'age']
    
    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    def preprocess(self, df: pd.DataFrame) -> np.ndarray:       
        # Работаем с копией
        data = df.copy()
        
        # Вычисляем distance
        data['distance'] = self.haversine_distance(
            data['lat'], data['long'],
            data['merch_lat'], data['merch_long']
        )
        
        # Вычисляем age
        data['age'] = 2019 - pd.to_datetime(data['dob']).dt.year
        
        # Вычисляем hour из unix_time
        data['hour'] = pd.to_datetime(data['unix_time'], unit='s').dt.hour
        
        # Кодируем категориальные признаки
        data['category'] = self.le_cat.transform(data['category'])
        data['gender'] = self.le_gen.transform(data['gender'])
        
        # ВАЖНО: порядок признаков ДОЛЖЕН совпадать с обучением!
        # В ноутбуке вы использовали: ['amt', 'city_pop', 'distance', 'age']
        numeric_data = data[['amt', 'city_pop', 'distance', 'age']]
        X_num = self.scaler.transform(numeric_data)
        
        # Категориальные признаки в том же порядке: category, gender, hour
        X_cat = data[['category', 'gender', 'hour']].values
        
        # Объединяем
        X = np.hstack([X_num, X_cat])
        
        # Для отладки - выводим форму и первые 2 строки
        print(f"DEBUG: X shape = {X.shape}")
        print(f"DEBUG: X sample (first row) = {X[0] if len(X) > 0 else 'N/A'}")
        
        return X
        
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self.preprocess(df)
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, df: pd.DataFrame, threshold: float = None) -> Tuple[np.ndarray, np.ndarray]:
        proba = self.predict_proba(df)
        threshold = threshold or Config.DEFAULT_THRESHOLD
        pred = (proba >= threshold).astype(int)
        return pred, proba