import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import streamlit as st

DB_PATH = Path(__file__).parent.parent / "database" / "history.db"

def init_database():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            prediction INTEGER NOT NULL,
            confidence REAL NOT NULL,
            attack_type TEXT,
            risk_level TEXT,
            method TEXT,
            features_json TEXT,
            shap_values_json TEXT
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_prediction ON predictions(prediction)
    """)
    
    conn.commit()
    conn.close()

def save_prediction(
    prediction: int,
    confidence: float,
    attack_type: str,
    risk_level: str,
    method: str,
    features: Optional[Dict] = None,
    shap_values: Optional[Dict] = None
):
    import json
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO predictions (prediction, confidence, attack_type, risk_level, method, features_json, shap_values_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        prediction,
        confidence,
        attack_type,
        risk_level,
        method,
        json.dumps(features) if features else None,
        json.dumps(shap_values) if shap_values else None
    ))
    
    conn.commit()
    conn.close()

def get_all_predictions(limit: int = 1000, offset: int = 0) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT id, timestamp, prediction, confidence, attack_type, risk_level, method
        FROM predictions
        ORDER BY timestamp DESC
        LIMIT ? OFFSET ?
    """
    df = pd.read_sql_query(query, conn, params=(limit, offset))
    conn.close()
    return df

def get_prediction_stats() -> Dict[str, Any]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 1")
    attacks = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 0")
    normal = cursor.fetchone()[0]
    
    cursor.execute("SELECT AVG(confidence) FROM predictions")
    avg_confidence = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT MAX(timestamp) FROM predictions")
    last_prediction = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "total": total,
        "attacks": attacks,
        "normal": normal,
        "avg_confidence": round(avg_confidence * 100, 2),
        "last_prediction": last_prediction
    }

def search_predictions(
    prediction_filter: Optional[int] = None,
    risk_filter: Optional[str] = None,
    method_filter: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 1000
) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    
    query = """
        SELECT id, timestamp, prediction, confidence, attack_type, risk_level, method
        FROM predictions
        WHERE 1=1
    """
    params = []
    
    if prediction_filter is not None:
        query += " AND prediction = ?"
        params.append(prediction_filter)
    
    if risk_filter:
        query += " AND risk_level = ?"
        params.append(risk_filter)
    
    if method_filter:
        query += " AND method = ?"
        params.append(method_filter)
    
    if date_from:
        query += " AND timestamp >= ?"
        params.append(date_from)
    
    if date_to:
        query += " AND timestamp <= ?"
        params.append(date_to)
    
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def export_predictions_csv() -> str:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
    conn.close()
    return df.to_csv(index=False)

def clear_history():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()

@st.cache_resource
def get_db_connection():
    init_database()
    return True