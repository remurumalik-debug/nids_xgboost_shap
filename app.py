import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="NIDS - Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --bg-dark: #0a0f1a;
        --bg-card: #111827;
        --bg-card-hover: #1a2332;
        --primary: #00d4ff;
        --primary-glow: #00d4ff40;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --accent-orange: #f59e0b;
        --accent-blue: #3b82f6;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --border: #1e293b;
        --grid: #1e293b;
        --shadow: rgba(0, 212, 255, 0.1);
    }
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: var(--bg-dark);
        color: var(--text-primary);
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: var(--bg-dark);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary);
    }
    
    /* Sidebar */
    .css-1d391kg, .css-1lcbmhc, .css-17lntkn {
        background: var(--bg-card) !important;
        border-right: 1px solid var(--border);
    }
    
    .css-1d391kg .stMarkdown, .css-1lcbmhc .stMarkdown {
        color: var(--text-primary) !important;
    }
    
    /* Navigation */
    .stRadio > label {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    .stRadio > div[role="radiogroup"] {
        gap: 8px !important;
    }
    
    .stRadio > div[role="radiogroup"] > label {
        background: transparent !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        gap: 12px !important;
    }
    
    .stRadio > div[role="radiogroup"] > label:hover {
        border-color: var(--primary) !important;
        background: var(--primary-glow) !important;
    }
    
    .stRadio > div[role="radiogroup"] > label[data-checked="true"] {
        border-color: var(--primary) !important;
        background: var(--primary-glow) !important;
        box-shadow: 0 0 20px var(--shadow) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary), #0099cc) !important;
        color: var(--bg-dark) !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 20px var(--shadow) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px var(--shadow) !important;
        background: linear-gradient(135deg, #00eaff, #00aaff) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    .stButton > button[kind="secondary"] {
        background: transparent !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
        box-shadow: none !important;
    }
    
    .stButton > button[kind="secondary"]:hover {
        border-color: var(--primary) !important;
        background: var(--primary-glow) !important;
    }
    
    /* Inputs */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stTextArea > div > div > textarea {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        padding: 10px 14px !important;
        transition: all 0.2s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px var(--primary-glow) !important;
    }
    
    .stTextInput > label,
    .stNumberInput > label,
    .stSelectbox > label,
    .stTextArea > label {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    
    /* File uploader */
    .stFileUploader > div {
        background: var(--bg-card) !important;
        border: 2px dashed var(--border) !important;
        border-radius: 12px !important;
        padding: 24px !important;
        transition: all 0.2s ease !important;
    }
    
    .stFileUploader > div:hover {
        border-color: var(--primary) !important;
        background: var(--primary-glow) !important;
    }
    
    .stFileUploader label {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }
    
    /* Dataframes */
    .stDataFrame {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    
    .stDataFrame [data-testid="stTable"] {
        background: var(--bg-card) !important;
    }
    
    .stDataFrame th {
        background: var(--bg-dark) !important;
        color: var(--primary) !important;
        font-weight: 600 !important;
        border-bottom: 2px solid var(--border) !important;
        padding: 12px 16px !important;
    }
    
    .stDataFrame td {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border-bottom: 1px solid var(--border) !important;
        padding: 10px 16px !important;
    }
    
    .stDataFrame tr:hover td {
        background: var(--bg-card-hover) !important;
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 20px !important;
        transition: all 0.2s ease !important;
    }
    
    [data-testid="stMetric"]:hover {
        border-color: var(--primary) !important;
        box-shadow: 0 8px 30px var(--shadow) !important;
    }
    
    [data-testid="stMetric"] > div {
        color: var(--text-primary) !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: var(--accent-green) !important;
        font-weight: 600 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-card) !important;
        border-radius: 10px !important;
        padding: 4px !important;
        gap: 4px !important;
        border: 1px solid var(--border) !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-secondary) !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        padding: 10px 20px !important;
        transition: all 0.2s ease !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary) !important;
        background: var(--primary-glow) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary), #0099cc) !important;
        color: var(--bg-dark) !important;
        box-shadow: 0 4px 20px var(--shadow) !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }
    
    .streamlit-expanderContent {
        background: var(--bg-dark) !important;
        border: 1px solid var(--border) !important;
        border-top: none !important;
        border-radius: 0 0 10px 10px !important;
        padding: 16px !important;
    }
    
    /* Alerts */
    .stAlert {
        border-radius: 10px !important;
        border: none !important;
    }
    
    .stAlert[data-baseweb="notification"][kind="success"] {
        background: linear-gradient(135deg, var(--accent-green), #059669) !important;
        color: var(--bg-dark) !important;
    }
    
    .stAlert[data-baseweb="notification"][kind="error"] {
        background: linear-gradient(135deg, var(--accent-red), #dc2626) !important;
        color: var(--bg-dark) !important;
    }
    
    .stAlert[data-baseweb="notification"][kind="warning"] {
        background: linear-gradient(135deg, var(--accent-orange), #d97706) !important;
        color: var(--bg-dark) !important;
    }
    
    .stAlert[data-baseweb="notification"][kind="info"] {
        background: linear-gradient(135deg, var(--primary), #0099cc) !important;
        color: var(--bg-dark) !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, var(--primary), #0099cc) !important;
        border-radius: 4px !important;
    }
    
    /* Slider */
    .stSlider > div > div > div > div {
        background: var(--primary) !important;
    }
    
    /* Plotly charts */
    .js-plotly-plot .plotly .modebar {
        background: var(--bg-card) !important;
        border-radius: 8px !important;
    }
    
    /* Custom cards */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .metric-card:hover {
        border-color: var(--primary);
        box-shadow: 0 8px 30px var(--shadow);
        transform: translateY(-2px);
    }
    
    .metric-card .icon {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        margin-bottom: 12px;
    }
    
    .metric-card .icon.green { background: rgba(16, 185, 129, 0.2); color: var(--accent-green); }
    .metric-card .icon.red { background: rgba(239, 68, 68, 0.2); color: var(--accent-red); }
    .metric-card .icon.blue { background: rgba(59, 130, 246, 0.2); color: var(--accent-blue); }
    .metric-card .icon.orange { background: rgba(245, 158, 11, 0.2); color: var(--accent-orange); }
    .metric-card .icon.purple { background: rgba(168, 85, 247, 0.2); color: #a855f7; }
    
    .metric-card .label {
        color: var(--text-secondary);
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    
    .metric-card .value {
        color: var(--text-primary);
        font-size: 1.75rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    
    .metric-card .delta {
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .metric-card .delta.positive { color: var(--accent-green); }
    .metric-card .delta.negative { color: var(--accent-red); }
    
    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 24px;
    }
    
    .section-header .icon {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        background: var(--primary-glow);
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--primary);
        font-size: 1.25rem;
    }
    
    .section-header h2 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .section-header .divider {
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, var(--primary), transparent);
    }
    
    /* Risk badges */
    .risk-badge {
        display: inline-flex;
        align-items: center;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .risk-badge.critical { background: rgba(239, 68, 68, 0.2); color: var(--accent-red); border: 1px solid var(--accent-red); }
    .risk-badge.high { background: rgba(245, 158, 11, 0.2); color: var(--accent-orange); border: 1px solid var(--accent-orange); }
    .risk-badge.medium { background: rgba(59, 130, 246, 0.2); color: var(--accent-blue); border: 1px solid var(--accent-blue); }
    .risk-badge.low { background: rgba(16, 185, 129, 0.2); color: var(--accent-green); border: 1px solid var(--accent-green); }
    .risk-badge.minimal { background: rgba(100, 116, 139, 0.2); color: var(--text-secondary); border: 1px solid var(--text-secondary); }
    
    /* Prediction card */
    .prediction-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
    }
    
    .prediction-card.attack {
        border-left: 4px solid var(--accent-red);
    }
    
    .prediction-card.normal {
        border-left: 4px solid var(--accent-green);
    }
    
    /* Code blocks */
    .stCodeBlock {
        background: var(--bg-dark) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }
    
    /* Markdown */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: var(--text-primary) !important;
    }
    
    .stMarkdown a {
        color: var(--primary) !important;
    }
    
    /* Selectbox dropdown */
    .stSelectbox [data-baseweb="select"] > div {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
    }
    
    .stSelectbox [data-baseweb="popover"] [role="listbox"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
    }
    
    .stSelectbox [role="option"] {
        color: var(--text-primary) !important;
    }
    
    .stSelectbox [role="option"]:hover {
        background: var(--primary-glow) !important;
    }
    
    /* Multiselect */
    .stMultiSelect [data-baseweb="tag"] {
        background: var(--primary-glow) !important;
        border: 1px solid var(--primary) !important;
        color: var(--primary) !important;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-color: var(--primary) !important;
        border-top-color: transparent !important;
    }
    
    /* Success/Error messages */
    .element-container:has(.stSuccess) {
        animation: fadeIn 0.3s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Table container */
    .table-container {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Custom divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border), transparent);
        margin: 24px 0;
    }
    
    /* Footer */
    .app-footer {
        text-align: center;
        padding: 24px;
        color: var(--text-muted);
        font-size: 0.85rem;
        border-top: 1px solid var(--border);
        margin-top: 48px;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .metric-card .value { font-size: 1.5rem; }
        .section-header h2 { font-size: 1.25rem; }
    }
</style>
"""

# Apply CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Import pages
from pages.dashboard import show_dashboard
from pages.prediction import show_prediction
from pages.upload import show_upload
from pages.explainability import show_explainability
from pages.analytics import show_analytics
from pages.history import show_history
from pages.about import show_about

# Initialize database
from utils.database import init_database
init_database()

# Sidebar navigation
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <div style="font-size: 2.5rem; margin-bottom: 8px;">🛡️</div>
            <h1 style="margin: 0; font-size: 1.5rem; font-weight: 700; background: linear-gradient(135deg, #00d4ff, #0099cc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">NIDS</h1>
            <p style="margin: 4px 0 0; color: #94a3b8; font-size: 0.8rem;">Network Intrusion Detection System</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    pages = {
        "🏠 Dashboard": "dashboard",
        "🔮 Manual Prediction": "prediction",
        "📁 Batch Upload": "upload",
        "🧠 Explainability": "explainability",
        "📊 Analytics": "analytics",
        "📜 History": "history",
        "ℹ️ About": "about"
    }
    
    # Initialize session state for page
    if "current_page" not in st.session_state:
        st.session_state.current_page = "dashboard"
    
    # Radio buttons for navigation
    selected_label = st.radio(
        "Navigation",
        options=list(pages.keys()),
        index=list(pages.values()).index(st.session_state.current_page),
        label_visibility="collapsed"
    )
    
    st.session_state.current_page = pages[selected_label]
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # System info
    st.markdown("""
        <div style="padding: 16px; background: #111827; border-radius: 10px; border: 1px solid #1e293b;">
            <div style="color: #94a3b8; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;">System Status</div>
            <div style="display: flex; align-items: center; gap: 8px; color: #10b981; font-size: 0.85rem;">
                <span style="width: 8px; height: 8px; background: #10b981; border-radius: 50%; animation: pulse 2s infinite;"></span>
                <span>Model Loaded</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px; color: #3b82f6; font-size: 0.85rem; margin-top: 8px;">
                <span style="width: 8px; height: 8px; background: #3b82f6; border-radius: 50%;"></span>
                <span>SHAP Ready</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px; color: #94a3b8; font-size: 0.85rem; margin-top: 8px;">
                <span style="width: 8px; height: 8px; background: #94a3b8; border-radius: 50%;"></span>
                <span>DB Connected</span>
            </div>
        </div>
        <style>
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    st.markdown("""
        <div style="text-align: center; color: #64748b; font-size: 0.75rem;">
            <p style="margin: 0;">v2.0.0</p>
            <p style="margin: 4px 0 0;">XGBoost + SHAP</p>
        </div>
    """, unsafe_allow_html=True)

# Main content area
page = st.session_state.current_page

if page == "dashboard":
    show_dashboard()
elif page == "prediction":
    show_prediction()
elif page == "upload":
    show_upload()
elif page == "explainability":
    show_explainability()
elif page == "analytics":
    show_analytics()
elif page == "history":
    show_history()
elif page == "about":
    show_about()