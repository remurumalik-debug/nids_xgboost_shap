import streamlit as st
from utils.visualizations import COLORS

def show_about():
    st.markdown("""
        <div class="section-header">
            <div class="icon">ℹ️</div>
            <h2>About This Project</h2>
            <div class="divider"></div>
        </div>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
        <div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #111827 0%, #1a2332 100%); border: 1px solid #1e293b; border-radius: 20px; margin-bottom: 32px;">
            <div style="font-size: 4rem; margin-bottom: 16px;">🛡️</div>
            <h1 style="margin: 0 0 8px; font-size: 2.5rem; font-weight: 700; background: linear-gradient(135deg, #00d4ff, #0099cc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Network Intrusion Detection System</h1>
            <p style="margin: 0; color: #94a3b8; font-size: 1.1rem;">XGBoost + SHAP Explainability for Cybersecurity</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Project Info
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
            <div class="section-header">
                <div class="icon" style="background: rgba(59, 130, 246, 0.2); color: #3b82f6;">📋</div>
                <h2>Project Information</h2>
                <div class="divider"></div>
            </div>
        """, unsafe_allow_html=True)
        
        info_items = [
            ("Project Title", "Network Intrusion Detection System with Explainable AI"),
            ("Author", "[Your Name]"),
            ("Supervisor", "[Supervisor Name]"),
            ("Institution", "[University/Institution Name]"),
            ("Department", "Computer Science / Cybersecurity"),
            ("Academic Year", "2024-2025"),
            ("Degree", "Bachelor's / Master's Thesis"),
            ("Status", "Complete")
        ]
        
        for label, value in info_items:
            st.markdown(f"""
                <div style="display: flex; justify-content: space-between; padding: 12px 0; border-bottom: 1px solid #1e293b;">
                    <span style="color: #94a3b8;">{label}</span>
                    <span style="color: #f8fafc; font-weight: 500;">{value}</span>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="section-header">
                <div class="icon" style="background: rgba(16, 185, 129, 0.2); color: #10b981;">🎯</div>
                <h2>Project Objectives</h2>
                <div class="divider"></div>
            </div>
        """, unsafe_allow_html=True)
        
        objectives = [
            "Develop an accurate Network Intrusion Detection System using XGBoost",
            "Handle class imbalance in network traffic data using SMOTE",
            "Achieve high precision to minimize false alarms (alert fatigue)",
            "Implement SHAP explainability for transparent AI decisions",
            "Create a professional web interface for real-time threat analysis",
            "Enable batch processing for historical traffic analysis",
            "Provide human-readable explanations for security analysts",
            "Store and manage prediction history for audit trails"
        ]
        
        for i, obj in enumerate(objectives, 1):
            st.markdown(f"""
                <div style="display: flex; gap: 12px; padding: 10px 0; align-items: flex-start;">
                    <span style="color: #00d4ff; font-weight: 700; min-width: 28px;">{i}.</span>
                    <span style="color: #e2e8f0;">{obj}</span>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # Dataset
    st.markdown("""
        <div class="section-header">
            <div class="icon" style="background: rgba(245, 158, 11, 0.2); color: #f59e0b;">📊</div>
            <h2>Dataset: NSL-KDD</h2>
            <div class="divider"></div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <div class="icon orange">📁</div>
                <div class="label">Dataset</div>
                <div class="value">NSL-KDD</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="metric-card">
                <div class="icon orange">📈</div>
                <div class="label">Samples (Train)</div>
                <div class="value">125,973</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="metric-card">
                <div class="icon orange">📈</div>
                <div class="label">Samples (Test)</div>
                <div class="value">22,544</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style="background: #111827; border: 1px solid #1e293b; border-radius: 12px; padding: 20px; margin-top: 16px;">
            <h4 style="margin: 0 0 12px; color: #f8fafc;">About NSL-KDD</h4>
            <p style="margin: 0 0 12px; color: #e2e8f0;">The NSL-KDD dataset is an improved version of the original KDD Cup 1999 dataset, addressing issues like redundant records and class imbalance.</p>
            <h5 style="margin: 16px 0 8px; color: #f8fafc;">Attack Categories:</h5>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; color: #94a3b8;">
                <div>• DoS (Denial of Service)</div>
                <div>• Probe (Scanning/Probing)</div>
                <div>• R2L (Remote to Local)</div>
                <div>• U2R (User to Root)</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # Technologies
    st.markdown("""
        <div class="section-header">
            <div class="icon" style="background: rgba(168, 85, 247, 0.2); color: #a855f7;">🛠️</div>
            <h2>Technologies & Libraries</h2>
            <div class="divider"></div>
        </div>
    """, unsafe_allow_html=True)
    
    tech_categories = {
        "Machine Learning": ["XGBoost", "scikit-learn", "imbalanced-learn (SMOTE)"],
        "Explainability": ["SHAP (TreeExplainer)"],
        "Data Processing": ["pandas", "numpy"],
        "Visualization": ["matplotlib", "seaborn", "plotly"],
        "Web Framework": ["Streamlit"],
        "Database": ["SQLite"],
        "Model Persistence": ["joblib"]
    }
    
    for category, techs in tech_categories.items():
        st.markdown(f"""
            <div style="background: #111827; border: 1px solid #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 16px;">
                <h5 style="margin: 0 0 12px; color: #00d4ff;">{category}</h5>
                <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                    {''.join([f'<span style="background: #1e293b; border: 1px solid #334155; color: #e2e8f0; padding: 6px 12px; border-radius: 20px; font-size: 0.85rem;">{t}</span>' for t in techs])}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # System Workflow
    st.markdown("""
        <div class="section-header">
            <div class="icon" style="background: rgba(16, 185, 129, 0.2); color: #10b981;">⚙️</div>
            <h2>System Workflow</h2>
            <div class="divider"></div>
        </div>
    """, unsafe_allow_html=True)
    
    steps = [
        ("1️⃣", "Data Loading", "Load NSL-KDD train/test datasets (41 features + label)"),
        ("2️⃣", "Preprocessing", "Label encoding (binary: normal/attack), categorical encoding, MinMax scaling"),
        ("3️⃣", "Class Balancing", "SMOTE oversampling to handle severe class imbalance (attack vs normal)"),
        ("4️⃣", "Model Training", "XGBoost classifier with optimized hyperparameters (n_estimators=300, max_depth=6)"),
        ("5️⃣", "Evaluation", "Accuracy, Precision, Recall, F1-Score, Confusion Matrix on test set"),
        ("6️⃣", "Explainability", "SHAP TreeExplainer for global and local feature importance"),
        ("7️⃣", "Deployment", "Streamlit web app with real-time prediction, batch upload, history tracking"),
    ]
    
    for icon, title, desc in steps:
        st.markdown(f"""
            <div style="display: flex; gap: 16px; padding: 16px; background: #111827; border: 1px solid #1e293b; border-radius: 12px; margin-bottom: 12px;">
                <div style="font-size: 2rem; min-width: 48px;">{icon}</div>
                <div style="flex: 1;">
                    <div style="color: #f8fafc; font-weight: 600; font-size: 1.1rem;">{title}</div>
                    <div style="color: #94a3b8; margin-top: 4px;">{desc}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # Model Info
    st.markdown("""
        <div class="section-header">
            <div class="icon" style="background: rgba(239, 68, 68, 0.2); color: #ef4444;">🤖</div>
            <h2>Model Information</h2>
            <div class="divider"></div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="section-header">
                <h3 style="margin: 0; color: #f8fafc;">XGBoost Hyperparameters</h3>
            </div>
        """, unsafe_allow_html=True)
        
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
                <div style="display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #1e293b;">
                    <span style="color: #94a3b8;">{k}</span>
                    <span style="color: #f8fafc; font-weight: 500; font-family: monospace;">{v}</span>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="section-header">
                <h3 style="margin: 0; color: #f8fafc;">Performance Metrics</h3>
            </div>
        """, unsafe_allow_html=True)
        
        metrics = {
            "Accuracy": "78.87%",
            "Precision": "96.51%",
            "Recall": "65.25%",
            "F1-Score": "77.86%",
            "AUC-ROC": "~0.92"
        }
        
        for k, v in metrics.items():
            st.markdown(f"""
                <div style="display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #1e293b;">
                    <span style="color: #94a3b8;">{k}</span>
                    <span style="color: #00d4ff; font-weight: 700; font-size: 1.1rem;">{v}</span>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # Features
    st.markdown("""
        <div class="section-header">
            <div class="icon" style="background: rgba(59, 130, 246, 0.2); color: #3b82f6;">🔧</div>
            <h2>Features (41 Total)</h2>
            <div class="divider"></div>
        </div>
    """, unsafe_allow_html=True)
    
    feature_categories = {
        "Basic (6)": ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes"],
        "Content (13)": ["land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login"],
        "Host Traffic (9)": ["count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate"],
        "Destination Traffic (10)": ["dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]
    }
    
    for cat, feats in feature_categories.items():
        with st.expander(f"{cat} ({len(feats)} features)", expanded=False):
            cols = st.columns(3)
            for i, feat in enumerate(feats):
                with cols[i % 3]:
                    st.markdown(f"`{feat}`")
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # Acknowledgments
    st.markdown("""
        <div class="section-header">
            <div class="icon" style="background: rgba(168, 85, 247, 0.2); color: #a855f7;">🙏</div>
            <h2>Acknowledgments</h2>
            <div class="divider"></div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style="background: #111827; border: 1px solid #1e293b; border-radius: 12px; padding: 24px;">
            <p style="margin: 0 0 12px; color: #e2e8f0;">This project was developed as part of academic research in cybersecurity and machine learning.</p>
            <h5 style="margin: 16px 0 8px; color: #f8fafc;">Special Thanks:</h5>
            <ul style="margin: 0; color: #94a3b8; line-height: 2;">
                <li>NSL-KDD dataset creators (University of New Brunswick)</li>
                <li>XGBoost developers (DMLC)</li>
                <li>SHAP library authors (Scott Lundberg et al.)</li>
                <li>Streamlit team for the amazing web framework</li>
                <li>Open source community for all supporting libraries</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
        <div class="app-footer">
            Network Intrusion Detection System v2.0.0<br>
            Built with &#10084; using XGBoost, SHAP, and Streamlit<br>
            <span style="color: #64748b;">For educational and research purposes</span>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    show_about()