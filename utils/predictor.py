import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, List
import warnings
import streamlit as st
warnings.filterwarnings("ignore")

MODEL_DIR = Path(__file__).parent.parent / "model"

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_DIR / "xgb_model.pkl")
    return model

@st.cache_resource
def load_scaler():
    return joblib.load(MODEL_DIR / "scaler.pkl")

@st.cache_resource
def load_encoder():
    return joblib.load(MODEL_DIR / "encoder.pkl")

@st.cache_resource
def load_cat_cols():
    return joblib.load(MODEL_DIR / "cat_cols.pkl")

@st.cache_resource
def load_feature_columns():
    return joblib.load(MODEL_DIR / "feature_columns.pkl")

class NIDSPredictor:
    def __init__(self):
        self.model = load_model()
        self.scaler = load_scaler()
        self.encoder = load_encoder()
        self.cat_cols = load_cat_cols()
        self.feature_columns = load_feature_columns()
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "difficulty" in df.columns:
            df.drop("difficulty", axis=1, inplace=True)
        
        for col in self.cat_cols:
            if col in df.columns:
                try:
                    df[col] = self.encoder.fit_transform(df[col])
                except Exception:
                    df[col] = 0
        
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        df = df[self.feature_columns]
        df_scaled = pd.DataFrame(
            self.scaler.transform(df),
            columns=self.feature_columns
        )
        return df_scaled
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X_processed = self.preprocess(df)
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed)[:, 1]
        return predictions, probabilities
    
    def predict_single(self, features: Dict[str, Any]) -> Tuple[int, float]:
        df = pd.DataFrame([features])
        preds, probs = self.predict(df)
        return int(preds[0]), float(probs[0])
    
    def get_feature_importance(self) -> pd.DataFrame:
        importance = self.model.feature_importances_
        return pd.DataFrame({
            "feature": self.feature_columns,
            "importance": importance
        }).sort_values("importance", ascending=False)

def get_risk_level(probability: float) -> str:
    if probability >= 0.8:
        return "Critical"
    elif probability >= 0.6:
        return "High"
    elif probability >= 0.4:
        return "Medium"
    elif probability >= 0.2:
        return "Low"
    return "Minimal"

def get_attack_type(prediction: int, probability: float) -> str:
    if prediction == 0:
        return "Normal Traffic"
    attack_types = {
        (0.8, 1.0): "DoS/DDoS Attack",
        (0.6, 0.8): "Probe/Scan Attack",
        (0.4, 0.6): "R2L Attack",
        (0.2, 0.4): "U2R Attack",
    }
    for (low, high), label in attack_types.items():
        if low <= probability < high:
            return label
    return "Unknown Attack"