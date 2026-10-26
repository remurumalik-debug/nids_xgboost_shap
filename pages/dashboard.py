import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from utils.database import get_prediction_stats
from utils.visualizations import (
    COLORS, plot_confusion_matrix, plot_model_comparison, 
    plot_metrics_radar, get_metric_card_html
)
from utils.predictor import NIDSPredictor

def show_dashboard():
    st.markdown("""
        <div class="section-header">
            <div class="icon">🏠</div>
            <h2>Dashboard</h2>
            <div class="divider"></div>
        </div>
    """, unsafe_allow_html=True)
    
    # System Overview
    st.markdown("""
        <div style="background: linear-gradient(135deg, #111827 0%, #1a2332 100%); border: 1px solid #1e293b; border-radius: 16px; padding: 24px; margin-bottom: 24px;">
            <h3 style="margin: 0 0 8px; color: #f8fafc; font-size: 1.25rem;">System Overview</h3>
            <p style="margin: 0; color: #94a3b8;">Real-time network intrusion detection powered by XGBoost with SHAP explainability</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Metrics Row
    stats = get_prediction_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(get_metric_card_html(
            "Total Predictions",
            f"{stats['total']:,}",
            f"+{stats['total']} all time",
            "blue"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(get_metric_card_html(
            "Attacks Detected",
            f"{stats['attacks']:,}",
            f"{stats['attacks'] / max(stats['total'], 1) * 100:.1f}% of traffic",
            "red"
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(get_metric_card_html(
            "Normal Traffic",
            f"{stats['normal']:,}",
            f"{stats['normal'] / max(stats['total'], 1) * 100:.1f}% of traffic",
            "green"
        ), unsafe_allow_html=True)
    
    with col4:
        st.markdown(get_metric_card_html(
            "Avg Confidence",
            f"{stats['avg_confidence']:.1f}%",
            "Model reliability",
            "orange"
        ), unsafe_allow_html=True)
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # Model Info
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
            <div class="section-header">
                <div class="icon" style="background: rgba(16, 185, 129, 0.2); color: #10b981;">✅</div>
                <h2>Model Performance</h2>
                <div class="divider"></div>
            </div>
        """, unsafe_allow_html=True)
        
        metrics = {
            "Accuracy": 78.87,
            "Precision": 96.51,
            "Recall": 65.25,
            "F1-Score": 77.86
        }
        
        fig = plot_metrics_radar(metrics)
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics cards
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Accuracy</div>
                    <div class="value">{metrics['Accuracy']:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Precision</div>
                    <div class="value">{metrics['Precision']:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
        with mcol2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Recall</div>
                    <div class="value">{metrics['Recall']:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
                <div class="metric-card">
                    <div class="label">F1-Score</div>
                    <div class="value">{metrics['F1-Score']:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="section-header">
                <div class="icon" style="background: rgba(59, 130, 246, 0.2); color: #3b82f6;">🤖</div>
                <h2>Model Information</h2>
                <div class="divider"></div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="metric-card" style="margin-bottom: 16px;">
                <div class="label">Algorithm</div>
                <div class="value" style="font-size: 1.25rem;">XGBoost</div>
            </div>
            <div class="metric-card" style="margin-bottom: 16px;">
                <div class="label">Dataset</div>
                <div class="value" style="font-size: 1.25rem;">NSL-KDD</div>
            </div>
            <div class="metric-card" style="margin-bottom: 16px;">
                <div class="label">Features</div>
                <div class="value" style="font-size: 1.25rem;">41</div>
            </div>
            <div class="metric-card" style="margin-bottom: 16px;">
                <div class="label">Classes</div>
                <div class="value" style="font-size: 1.25rem;">Binary (Normal / Attack)</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Hyperparameters
        with st.expander("⚙️ Model Hyperparameters", expanded=False):
            hyperparams = {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.01,
                "subsample": 0.8,
                "scale_pos_weight": 3,
                "eval_metric": "logloss",
                "random_state": 42
            }
            for k, v in hyperparams.items():
                st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #1e293b;">
                        <span style="color: #94a3b8;">{k}</span>
                        <span style="color: #f8fafc; font-weight: 500;">{v}</span>
                    </div>
                """, unsafe_allow_html=True)
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # Quick Actions
    st.markdown("""
        <div class="section-header">
            <div class="icon" style="background: rgba(168, 85, 247, 0.2); color: #a855f7;">⚡</div>
            <h2>Quick Actions</h2>
            <div class="divider"></div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔮 Manual Prediction", use_container_width=True):
            st.session_state.current_page = "prediction"
            st.rerun()
    
    with col2:
        if st.button("📁 Batch Upload", use_container_width=True):
            st.session_state.current_page = "upload"
            st.rerun()
    
    with col3:
        if st.button("🧠 Explainability", use_container_width=True):
            st.session_state.current_page = "explainability"
            st.rerun()
    
    with col4:
        if st.button("📜 View History", use_container_width=True):
            st.session_state.current_page = "history"
            st.rerun()
    
    # Recent Activity
    if stats['total'] > 0:
        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        st.markdown("""
            <div class="section-header">
                <div class="icon" style="background: rgba(245, 158, 11, 0.2); color: #f59e0b;">📋</div>
                <h2>Recent Predictions</h2>
                <div class="divider"></div>
            </div>
        """, unsafe_allow_html=True)
        
        from utils.database import get_all_predictions
        recent = get_all_predictions(limit=5)
        
        if not recent.empty:
            recent['timestamp'] = pd.to_datetime(recent['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            recent['prediction'] = recent['prediction'].map({0: '🟢 Normal', 1: '🔴 Attack'})
            recent['confidence'] = recent['confidence'].apply(lambda x: f"{x*100:.1f}%")
            
            st.dataframe(
                recent[['timestamp', 'prediction', 'confidence', 'attack_type', 'risk_level', 'method']],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No predictions yet. Start by making a prediction!")
    
    # Last prediction time
    if stats['last_prediction']:
        st.markdown(f"""
            <div style="text-align: center; padding: 16px; color: #64748b; font-size: 0.85rem; margin-top: 16px;">
                Last prediction: {stats['last_prediction']}
            </div>
        """, unsafe_allow_html=True)