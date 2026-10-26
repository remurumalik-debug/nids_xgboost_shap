import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.metrics import confusion_matrix
from typing import Dict, List
import warnings
warnings.filterwarnings("ignore")

COLORS = {
    "bg_dark": "#0a0e17",
    "bg_card": "#131a2a",
    "bg_hover": "#1a2332",
    "primary": "#00d4aa",
    "primary_light": "#4ef5c9",
    "secondary": "#007aff",
    "accent_red": "#ff3b5c",
    "accent_green": "#00d4aa",
    "accent_orange": "#ff9f0a",
    "accent_blue": "#007aff",
    "text_primary": "#ffffff",
    "text_secondary": "#8b98ab",
    "text_muted": "#5c6a7d",
    "border": "#1e2a3e",
    "grid": "#1e2a3e"
}

def apply_dark_theme():
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': COLORS["bg_dark"],
        'axes.facecolor': COLORS["bg_card"],
        'axes.edgecolor': COLORS["border"],
        'axes.labelcolor': COLORS["text_primary"],
        'axes.titlecolor': COLORS["text_primary"],
        'xtick.color': COLORS["text_secondary"],
        'ytick.color': COLORS["text_secondary"],
        'grid.color': COLORS["grid"],
        'grid.alpha': 0.3,
        'font.family': 'DejaVu Sans',
        'font.size': 10
    })

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    cm = confusion_matrix(y_true, y_pred)
    labels = ['Normal', 'Attack']
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale=[
            [0, COLORS["bg_card"]],
            [0.5, COLORS["secondary"]],
            [1, COLORS["primary"]]
        ],
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 18, "color": "white"},
        hoverongaps=False,
        showscale=False
    ))
    
    fig.update_layout(
        title=dict(text="Confusion Matrix", font=dict(size=18, color=COLORS["text_primary"])),
        xaxis=dict(title="Predicted", titlefont=dict(size=14, color=COLORS["text_primary"]), tickfont=dict(size=12, color=COLORS["text_primary"])),
        yaxis=dict(title="Actual", titlefont=dict(size=14, color=COLORS["text_primary"]), tickfont=dict(size=12, color=COLORS["text_primary"])),
        paper_bgcolor=COLORS["bg_dark"],
        plot_bgcolor=COLORS["bg_card"],
        font=dict(color=COLORS["text_primary"]),
        margin=dict(l=60, r=40, t=60, b=60),
        height=400
    )
    
    return fig

def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    top_features = importance_df.head(top_n).iloc[::-1]
    
    fig = go.Figure(data=go.Bar(
        y=top_features['feature'],
        x=top_features['importance'],
        orientation='h',
        marker=dict(
            color=top_features['importance'],
            colorscale=[[0, COLORS["secondary"]], [1, COLORS["primary"]]],
            showscale=False,
            line=dict(color=COLORS["border"], width=1)
        ),
        text=[f"{v:.4f}" for v in top_features['importance']],
        textposition='outside',
        textfont=dict(size=11, color=COLORS["text_primary"]),
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=f"Top {top_n} Feature Importance", font=dict(size=18, color=COLORS["text_primary"])),
        xaxis=dict(title="Importance Score", titlefont=dict(size=12, color=COLORS["text_primary"]), tickfont=dict(size=11, color=COLORS["text_secondary"]), gridcolor=COLORS["grid"]),
        yaxis=dict(title="", titlefont=dict(size=12, color=COLORS["text_primary"]), tickfont=dict(size=11, color=COLORS["text_primary"])),
        paper_bgcolor=COLORS["bg_dark"],
        plot_bgcolor=COLORS["bg_card"],
        font=dict(color=COLORS["text_primary"]),
        margin=dict(l=180, r=80, t=60, b=60),
        height=max(400, top_n * 30)
    )
    
    return fig

def plot_model_comparison() -> go.Figure:
    models = ['XGBoost', 'Random Forest', 'LightGBM', 'Logistic Regression', 'SVM']
    accuracy = [78.87, 76.2, 77.5, 72.1, 74.3]
    precision = [96.51, 94.8, 95.2, 91.5, 93.2]
    recall = [65.25, 62.1, 63.8, 58.9, 60.4]
    f1 = [77.86, 75.1, 76.4, 71.8, 73.5]
    
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=("Model Performance Comparison",)
    )
    
    fig.add_trace(go.Bar(
        name='Accuracy',
        x=models,
        y=accuracy,
        marker_color=COLORS["primary"],
        text=[f"{v:.1f}%" for v in accuracy],
        textposition='outside',
        textfont=dict(size=11, color=COLORS["text_primary"])
    ))
    
    fig.add_trace(go.Bar(
        name='Precision',
        x=models,
        y=precision,
        marker_color=COLORS["secondary"],
        text=[f"{v:.1f}%" for v in precision],
        textposition='outside',
        textfont=dict(size=11, color=COLORS["text_primary"])
    ))
    
    fig.add_trace(go.Bar(
        name='Recall',
        x=models,
        y=recall,
        marker_color=COLORS["accent_orange"],
        text=[f"{v:.1f}%" for v in recall],
        textposition='outside',
        textfont=dict(size=11, color=COLORS["text_primary"])
    ))
    
    fig.add_trace(go.Bar(
        name='F1-Score',
        x=models,
        y=f1,
        marker_color=COLORS["accent_blue"],
        text=[f"{v:.1f}%" for v in f1],
        textposition='outside',
        textfont=dict(size=11, color=COLORS["text_primary"])
    ))
    
    fig.update_layout(
        barmode='group',
        paper_bgcolor=COLORS["bg_dark"],
        plot_bgcolor=COLORS["bg_card"],
        font=dict(color=COLORS["text_primary"]),
        title=dict(text="Model Performance Comparison", font=dict(size=18, color=COLORS["text_primary"])),
        xaxis=dict(title="Model", titlefont=dict(size=12, color=COLORS["text_primary"]), tickfont=dict(size=11, color=COLORS["text_primary"]), gridcolor=COLORS["grid"]),
        yaxis=dict(title="Score (%)", titlefont=dict(size=12, color=COLORS["text_primary"]), tickfont=dict(size=11, color=COLORS["text_secondary"]), gridcolor=COLORS["grid"]),
        legend=dict(bgcolor=COLORS["bg_card"], bordercolor=COLORS["border"], font=dict(color=COLORS["text_primary"])),
        margin=dict(l=60, r=40, t=80, b=60),
        height=450
    )
    
    return fig

def plot_metrics_radar(metrics: Dict[str, float]) -> go.Figure:
    categories = list(metrics.keys())
    values = list(metrics.values())
    values += values[:1]
    categories += categories[:1]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(0, 212, 170, 0.2)',
        line=dict(color=COLORS["primary"], width=3),
        marker=dict(size=8, color=COLORS["primary"]),
        name='Model Metrics'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10, color=COLORS["text_secondary"]),
                gridcolor=COLORS["grid"]
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color=COLORS["text_primary"])
            ),
            bgcolor=COLORS["bg_card"]
        ),
        paper_bgcolor=COLORS["bg_dark"],
        font=dict(color=COLORS["text_primary"]),
        title=dict(text="Model Metrics Radar", font=dict(size=18, color=COLORS["text_primary"])),
        showlegend=False,
        margin=dict(l=60, r=40, t=60, b=60),
        height=400
    )
    
    return fig

def plot_prediction_distribution(predictions: List[int], probabilities: List[float]) -> go.Figure:
    normal_probs = [p for i, p in zip(predictions, probabilities) if i == 0]
    attack_probs = [p for i, p in zip(predictions, probabilities) if i == 1]
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=normal_probs,
        name='Normal',
        marker_color=COLORS["accent_green"],
        opacity=0.7,
        nbinsx=20,
        hovertemplate='Confidence: %{x:.2f}<br>Count: %{y}<extra></extra>'
    ))
    
    fig.add_trace(go.Histogram(
        x=attack_probs,
        name='Attack',
        marker_color=COLORS["accent_red"],
        opacity=0.7,
        nbinsx=20,
        hovertemplate='Confidence: %{x:.2f}<br>Count: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        barmode='overlay',
        title=dict(text="Prediction Confidence Distribution", font=dict(size=18, color=COLORS["text_primary"])),
        xaxis=dict(title="Confidence Score", titlefont=dict(size=12, color=COLORS["text_primary"]), tickfont=dict(size=11, color=COLORS["text_secondary"]), gridcolor=COLORS["grid"]),
        yaxis=dict(title="Count", titlefont=dict(size=12, color=COLORS["text_primary"]), tickfont=dict(size=11, color=COLORS["text_secondary"]), gridcolor=COLORS["grid"]),
        paper_bgcolor=COLORS["bg_dark"],
        plot_bgcolor=COLORS["bg_card"],
        font=dict(color=COLORS["text_primary"]),
        legend=dict(bgcolor=COLORS["bg_card"], bordercolor=COLORS["border"], font=dict(color=COLORS["text_primary"])),
        margin=dict(l=60, r=40, t=60, b=60),
        height=400
    )
    
    return fig

def plot_shap_importance(shap_importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    top_features = shap_importance_df.head(top_n).iloc[::-1]
    
    fig = go.Figure(data=go.Bar(
        y=top_features['feature'],
        x=top_features['importance'],
        orientation='h',
        marker=dict(
            color=top_features['importance'],
            colorscale=[[0, COLORS["secondary"]], [1, COLORS["accent_orange"]]],
            showscale=False
        ),
        text=[f"{v:.4f}" for v in top_features['importance']],
        textposition='outside',
        textfont=dict(size=11, color=COLORS["text_primary"]),
        hovertemplate='<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=f"SHAP Feature Importance (Top {top_n})", font=dict(size=18, color=COLORS["text_primary"])),
        xaxis=dict(title="Mean |SHAP Value|", titlefont=dict(size=12, color=COLORS["text_primary"]), tickfont=dict(size=11, color=COLORS["text_secondary"]), gridcolor=COLORS["grid"]),
        yaxis=dict(title="", titlefont=dict(size=12, color=COLORS["text_primary"]), tickfont=dict(size=11, color=COLORS["text_primary"])),
        paper_bgcolor=COLORS["bg_dark"],
        plot_bgcolor=COLORS["bg_card"],
        font=dict(color=COLORS["text_primary"]),
        margin=dict(l=180, r=80, t=60, b=60),
        height=max(400, top_n * 30)
    )
    
    return fig

def get_metric_card_html(title: str, value: str, delta: str = None, color: str = "primary") -> str:
    color_map = {
        "primary": COLORS["primary"],
        "red": COLORS["accent_red"],
        "green": COLORS["accent_green"],
        "orange": COLORS["accent_orange"],
        "blue": COLORS["accent_blue"]
    }
    c = color_map.get(color, COLORS["primary"])
    delta_html = f'<div style="color: {c}; font-size: 0.85rem; margin-top: 4px;">{delta}</div>' if delta else ''
    
    return f"""
    <div style="
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-left: 4px solid {c};
        border-radius: 10px;
        padding: 20px;
        transition: all 0.3s ease;
    ">
        <div style="color: {COLORS['text_muted']}; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">{title}</div>
        <div style="color: {COLORS['text_primary']}; font-size: 2rem; font-weight: 700; margin-bottom: 4px;">{value}</div>
        {delta_html}
    </div>
    """