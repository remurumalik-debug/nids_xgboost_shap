import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
from utils.predictor import NIDSPredictor, get_risk_level, get_attack_type
from utils.shap_utils import get_shap_explainer, generate_shap_waterfall_plot, generate_text_explanation
from utils.visualizations import COLORS
from utils.database import save_prediction
import matplotlib.pyplot as plt

def show_prediction():
    st.markdown("""
        <div class="section-header">
            <div class="icon">🔮</div>
            <h2>Manual Prediction</h2>
            <div class="divider"></div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style="background: #111827; border: 1px solid #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 24px;">
            <p style="margin: 0; color: #94a3b8;">Enter network traffic features manually to get real-time intrusion detection prediction with SHAP explainability.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize predictor
    if "predictor" not in st.session_state:
        with st.spinner("Loading model..."):
            st.session_state.predictor = NIDSPredictor()
    
    predictor = st.session_state.predictor
    feature_columns = predictor.feature_columns
    
    # Feature groups for organized input
    feature_groups = {
        "Basic Connection": [
            "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes"
        ],
        "Content": [
            "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
            "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
            "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login"
        ],
        "Traffic (Same Host)": [
            "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
            "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate"
        ],
        "Traffic (Destination Host)": [
            "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
            "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
            "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
        ]
    }
    
    # Create tabs for feature groups
    tabs = st.tabs(["🔗 Basic", "📄 Content", "📊 Host Traffic", "🌐 Dest Traffic", "⚡ Quick Presets"])
    
    features = {}
    
    # Default values for quick testing
    presets = {
        "Normal Traffic": {
            "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "SF",
            "src_bytes": 232, "dst_bytes": 8153, "land": 0, "wrong_fragment": 0,
            "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 1,
            "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0,
            "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
            "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
            "count": 2, "srv_count": 2, "serror_rate": 0.0, "srv_serror_rate": 0.0,
            "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 1.0,
            "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0, "dst_host_count": 255,
            "dst_host_srv_count": 255, "dst_host_same_srv_rate": 1.0,
            "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 0.01,
            "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0,
            "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.0,
            "dst_host_srv_rerror_rate": 0.0
        },
        "DoS Attack": {
            "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "S0",
            "src_bytes": 0, "dst_bytes": 0, "land": 0, "wrong_fragment": 0,
            "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 0,
            "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0,
            "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
            "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
            "count": 511, "srv_count": 511, "serror_rate": 1.0, "srv_serror_rate": 1.0,
            "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 1.0,
            "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0, "dst_host_count": 255,
            "dst_host_srv_count": 255, "dst_host_same_srv_rate": 1.0,
            "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 0.0,
            "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 1.0,
            "dst_host_srv_serror_rate": 1.0, "dst_host_rerror_rate": 0.0,
            "dst_host_srv_rerror_rate": 0.0
        },
        "Probe Attack": {
            "duration": 0, "protocol_type": "tcp", "service": "ecr_i", "flag": "SF",
            "src_bytes": 44, "dst_bytes": 58, "land": 0, "wrong_fragment": 0,
            "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 0,
            "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0,
            "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
            "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
            "count": 3, "srv_count": 3, "serror_rate": 0.0, "srv_serror_rate": 0.0,
            "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 1.0,
            "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0, "dst_host_count": 255,
            "dst_host_srv_count": 255, "dst_host_same_srv_rate": 1.0,
            "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 0.0,
            "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0,
            "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.0,
            "dst_host_srv_rerror_rate": 0.0
        }
    }
    
    with tabs[0]:  # Basic Connection
        col1, col2, col3 = st.columns(3)
        with col1:
            features["duration"] = st.number_input("Duration", value=0, min_value=0, help="Connection duration in seconds")
            features["protocol_type"] = st.selectbox("Protocol Type", ["tcp", "udp", "icmp"], index=0)
        with col2:
            features["service"] = st.selectbox("Service", ["http", "smtp", "ftp", "dns", "ssh", "telnet", "ecr_i", "other"], index=0)
            features["flag"] = st.selectbox("Flag", ["SF", "S0", "REJ", "RSTR", "RSTO", "SH", "S1", "S2", "S3", "OTH"], index=0)
        with col3:
            features["src_bytes"] = st.number_input("Source Bytes", value=232, min_value=0, help="Bytes from source to destination")
            features["dst_bytes"] = st.number_input("Destination Bytes", value=8153, min_value=0, help="Bytes from destination to source")
    
    with tabs[1]:  # Content
        col1, col2, col3 = st.columns(3)
        with col1:
            features["land"] = st.selectbox("Land", [0, 1], index=0, help="1 if connection is from/to same host/port")
            features["wrong_fragment"] = st.number_input("Wrong Fragment", value=0, min_value=0)
            features["urgent"] = st.number_input("Urgent", value=0, min_value=0, help="Number of urgent packets")
            features["hot"] = st.number_input("Hot", value=0, min_value=0, help="Number of 'hot' indicators")
        with col2:
            features["num_failed_logins"] = st.number_input("Failed Logins", value=0, min_value=0)
            features["logged_in"] = st.selectbox("Logged In", [0, 1], index=1, help="1 if successfully logged in")
            features["num_compromised"] = st.number_input("Compromised", value=0, min_value=0)
            features["root_shell"] = st.selectbox("Root Shell", [0, 1], index=0, help="1 if root shell obtained")
        with col3:
            features["su_attempted"] = st.selectbox("SU Attempted", [0, 1], index=0, help="1 if 'su root' attempted")
            features["num_root"] = st.number_input("Num Root", value=0, min_value=0)
            features["num_file_creations"] = st.number_input("File Creations", value=0, min_value=0)
            features["num_shells"] = st.number_input("Shells", value=0, min_value=0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            features["num_access_files"] = st.number_input("Access Files", value=0, min_value=0)
            features["num_outbound_cmds"] = st.number_input("Outbound Cmds", value=0, min_value=0)
        with col2:
            features["is_host_login"] = st.selectbox("Host Login", [0, 1], index=0)
            features["is_guest_login"] = st.selectbox("Guest Login", [0, 1], index=0)
    
    with tabs[2]:  # Host Traffic
        col1, col2, col3 = st.columns(3)
        with col1:
            features["count"] = st.number_input("Count", value=2, min_value=0, help="Connections to same host in past 2s")
            features["srv_count"] = st.number_input("Srv Count", value=2, min_value=0, help="Connections to same service in past 2s")
            features["serror_rate"] = st.slider("SError Rate", 0.0, 1.0, 0.0, 0.01, help="% connections with SYN errors")
            features["srv_serror_rate"] = st.slider("Srv SError Rate", 0.0, 1.0, 0.0, 0.01)
        with col2:
            features["rerror_rate"] = st.slider("RError Rate", 0.0, 1.0, 0.0, 0.01, help="% connections with REJ errors")
            features["srv_rerror_rate"] = st.slider("Srv RError Rate", 0.0, 1.0, 0.0, 0.01)
            features["same_srv_rate"] = st.slider("Same Srv Rate", 0.0, 1.0, 1.0, 0.01, help="% connections to same service")
            features["diff_srv_rate"] = st.slider("Diff Srv Rate", 0.0, 1.0, 0.0, 0.01, help="% connections to different services")
        with col3:
            features["srv_diff_host_rate"] = st.slider("Srv Diff Host Rate", 0.0, 1.0, 0.0, 0.01, help="% connections to different hosts")
    
    with tabs[3]:  # Destination Traffic
        col1, col2, col3 = st.columns(3)
        with col1:
            features["dst_host_count"] = st.number_input("Dst Host Count", value=255, min_value=0, help="Connections to same dest host")
            features["dst_host_srv_count"] = st.number_input("Dst Host Srv Count", value=255, min_value=0)
            features["dst_host_same_srv_rate"] = st.slider("Dst Host Same Srv Rate", 0.0, 1.0, 1.0, 0.01)
            features["dst_host_diff_srv_rate"] = st.slider("Dst Host Diff Srv Rate", 0.0, 1.0, 0.0, 0.01)
        with col2:
            features["dst_host_same_src_port_rate"] = st.slider("Dst Host Same Src Port Rate", 0.0, 1.0, 0.01, 0.01)
            features["dst_host_srv_diff_host_rate"] = st.slider("Dst Host Srv Diff Host Rate", 0.0, 1.0, 0.0, 0.01)
            features["dst_host_serror_rate"] = st.slider("Dst Host SError Rate", 0.0, 1.0, 0.0, 0.01)
            features["dst_host_srv_serror_rate"] = st.slider("Dst Host Srv SError Rate", 0.0, 1.0, 0.0, 0.01)
        with col3:
            features["dst_host_rerror_rate"] = st.slider("Dst Host RError Rate", 0.0, 1.0, 0.0, 0.01)
            features["dst_host_srv_rerror_rate"] = st.slider("Dst Host Srv RError Rate", 0.0, 1.0, 0.0, 0.01)
    
    with tabs[4]:  # Quick Presets
        st.markdown("### Quick Test Presets")
        st.markdown("Select a preset to auto-fill all fields:")
        
        preset_cols = st.columns(len(presets))
        for i, (name, preset_features) in enumerate(presets.items()):
            with preset_cols[i]:
                if st.button(name, use_container_width=True, key=f"preset_{name}"):
                    for key, value in preset_features.items():
                        st.session_state[f"pred_{key}"] = value
                    st.rerun()
        
        st.markdown("---")
        if st.button("🔄 Reset All Fields", use_container_width=True):
            for key in feature_columns:
                if f"pred_{key}" in st.session_state:
                    del st.session_state[f"pred_{key}"]
            st.rerun()
    
    # Apply preset values to session state
    for key in feature_columns:
        if f"pred_{key}" in st.session_state:
            features[key] = st.session_state[f"pred_{key}"]
    
    # Predict button
    st.markdown("<br>", unsafe_allow_html=True)
    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
    with predict_col2:
        predict_btn = st.button("🚀 Analyze Traffic", type="primary", use_container_width=True)
    
    if predict_btn:
        with st.spinner("Analyzing network traffic..."):
            # Make prediction
            prediction, confidence = predictor.predict_single(features)
            risk_level = get_risk_level(confidence)
            attack_type = get_attack_type(prediction, confidence)
            
            # Save to history
            save_prediction(
                prediction=prediction,
                confidence=confidence,
                attack_type=attack_type,
                risk_level=risk_level,
                method="Manual",
                features=features
            )
            
            # Display results
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Prediction Card
            card_class = "attack" if prediction == 1 else "normal"
            pred_label = "ATTACK DETECTED" if prediction == 1 else "NORMAL TRAFFIC"
            pred_icon = "🔴" if prediction == 1 else "🟢"
            pred_color = COLORS["accent_red"] if prediction == 1 else COLORS["accent_green"]
            
            st.markdown(f"""
                <div class="prediction-card {card_class}">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <span style="font-size: 3rem;">{pred_icon}</span>
                            <div>
                                <div style="font-size: 1.5rem; font-weight: 700; color: {pred_color};">{pred_label}</div>
                                <div style="color: #94a3b8; font-size: 0.9rem;">{attack_type}</div>
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 2.5rem; font-weight: 700; color: {pred_color};">{confidence*100:.1f}%</div>
                            <div style="color: #94a3b8;">Confidence</div>
                        </div>
                    </div>
                    <div style="display: flex; gap: 16px; flex-wrap: wrap;">
                        <span class="risk-badge {risk_level.lower()}">{risk_level} Risk</span>
                        <span style="color: #94a3b8; font-size: 0.85rem; align-self: center;">Analyzed at {datetime.now().strftime('%H:%M:%S')}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Metrics row
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="icon blue">🎯</div>
                        <div class="label">Prediction</div>
                        <div class="value">{'Attack' if prediction == 1 else 'Normal'}</div>
                    </div>
                """, unsafe_allow_html=True)
            with mcol2:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="icon orange">📊</div>
                        <div class="label">Confidence</div>
                        <div class="value">{confidence*100:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            with mcol3:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="icon {'red' if risk_level in ['Critical', 'High'] else 'orange' if risk_level == 'Medium' else 'green'}">⚠️</div>
                        <div class="label">Risk Level</div>
                        <div class="value">{risk_level}</div>
                    </div>
                """, unsafe_allow_html=True)
            with mcol4:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="icon purple">🏷️</div>
                        <div class="label">Type</div>
                        <div class="value">{attack_type}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # SHAP Explanation
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
                <div class="section-header">
                    <div class="icon" style="background: rgba(168, 85, 247, 0.2); color: #a855f7;">🧠</div>
                    <h2>SHAP Explanation</h2>
                    <div class="divider"></div>
                </div>
            """, unsafe_allow_html=True)
            
            with st.spinner("Generating SHAP explanation..."):
                try:
                    # Get processed features for SHAP
                    X_processed = predictor.preprocess(pd.DataFrame([features]))
                    
                    # Get SHAP explainer
                    explainer = get_shap_explainer(predictor.model, feature_columns)
                    shap_values = explainer.get_shap_values(X_processed, max_samples=1)
                    expected_value = explainer.get_expected_value()
                    
                    # Local explanation
                    local_explanation = explainer.get_local_explanation(shap_values, X_processed, idx=0)
                    
                    # Waterfall plot
                    fig = generate_shap_waterfall_plot(shap_values, X_processed, feature_columns, 0, expected_value)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                    
                    # Text explanation
                    explanation = generate_text_explanation(local_explanation, prediction, confidence)
                    st.markdown(f"""
                        <div style="background: #111827; border: 1px solid #1e293b; border-radius: 12px; padding: 20px; margin-top: 16px;">
                            <h4 style="margin: 0 0 12px; color: #f8fafc;">📝 AI Explanation</h4>
                            <p style="margin: 0; color: #e2e8f0; line-height: 1.7;">{explanation}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Top contributing features table
                    st.markdown("#### Top Contributing Features")
                    top_features = local_explanation.head(10)
                    top_features['shap_value'] = top_features['shap_value'].apply(lambda x: f"{x:+.4f}")
                    top_features['feature_value'] = top_features['feature_value'].apply(lambda x: f"{x:.4f}")
                    top_features.columns = ['Feature', 'Feature Value', 'SHAP Value']
                    
                    st.dataframe(top_features, use_container_width=True, hide_index=True)
                    
                except Exception as e:
                    st.error(f"Error generating SHAP explanation: {str(e)}")
            
            st.success("✅ Analysis complete! Results saved to history.")


if __name__ == "__main__":
    show_prediction()