import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime
from utils.predictor import NIDSPredictor, get_risk_level, get_attack_type
from utils.shap_utils import get_shap_explainer, generate_text_explanation
from utils.visualizations import COLORS, plot_prediction_distribution, get_metric_card_html
from utils.database import save_prediction

def show_upload():
    st.markdown("""
        <div class="section-header">
            <div class="icon">📁</div>
            <h2>Batch Upload & Prediction</h2>
            <div class="divider"></div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style="background: #111827; border: 1px solid #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 24px;">
            <p style="margin: 0; color: #94a3b8;">Upload a CSV file with network traffic data to perform batch intrusion detection. 
            The file should contain the 41 NSL-KDD features (protocol_type, service, flag will be auto-encoded).</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize predictor
    if "predictor" not in st.session_state:
        with st.spinner("Loading model..."):
            st.session_state.predictor = NIDSPredictor()
    
    predictor = st.session_state.predictor
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=["csv"],
        help="Upload a CSV file with network traffic features. Required columns: duration, protocol_type, service, flag, src_bytes, dst_bytes, land, wrong_fragment, urgent, hot, num_failed_logins, logged_in, num_compromised, root_shell, su_attempted, num_root, num_file_creations, num_shells, num_access_files, num_outbound_cmds, is_host_login, is_guest_login, count, srv_count, serror_rate, srv_serror_rate, rerror_rate, srv_rerror_rate, same_srv_rate, diff_srv_rate, srv_diff_host_rate, dst_host_count, dst_host_srv_count, dst_host_same_srv_rate, dst_host_diff_srv_rate, dst_host_same_src_port_rate, dst_host_srv_diff_host_rate, dst_host_serror_rate, dst_host_srv_serror_rate, dst_host_rerror_rate, dst_host_srv_rerror_rate"
    )
    
    # Sample CSV download
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("📥 Download Sample CSV", use_container_width=True):
            sample_df = create_sample_csv()
            csv = sample_df.to_csv(index=False)
            st.download_button(
                label="Download Sample",
                data=csv,
                file_name="nids_sample.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.markdown("#### File Preview")
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)
            
            # Validate columns
            required_cols = predictor.feature_columns
            missing_cols = [c for c in required_cols if c not in df.columns]
            extra_cols = [c for c in df.columns if c not in required_cols and c not in ['label', 'difficulty']]
            
            if missing_cols:
                st.warning(f"Missing columns (will be filled with 0): {missing_cols[:10]}{'...' if len(missing_cols) > 10 else ''}")
            
            if extra_cols:
                st.info(f"Extra columns (will be ignored): {extra_cols[:10]}{'...' if len(extra_cols) > 10 else ''}")
            
            # Prediction button
            if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
                run_batch_prediction(df, predictor)
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

def create_sample_csv():
    """Create a sample CSV with the correct columns"""
    cols = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
        "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
        "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
        "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
        "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
    ]
    
    # Create sample rows
    sample_data = [
        # Normal traffic
        [0, "tcp", "http", "SF", 181, 5450, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 9, 9, 1.0, 0.0, 0.11, 0.0, 0.0, 0.0],
        # Attack traffic (DoS-like)
        [0, "tcp", "http", "S0", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 100, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 255, 255, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        # Normal traffic
        [0, "udp", "dns", "SF", 35, 142, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 5, 5, 1.0, 0.0, 0.2, 0.0, 0.0, 0.0],
    ]
    
    return pd.DataFrame(sample_data, columns=cols)

def run_batch_prediction(df: pd.DataFrame, predictor: NIDSPredictor):
    """Run predictions on uploaded dataframe"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Preprocessing data...")
    progress_bar.progress(25)
    
    # Make predictions
    status_text.text("Running predictions...")
    progress_bar.progress(50)
    
    predictions, probabilities = predictor.predict(df)
    
    status_text.text("Generating results...")
    progress_bar.progress(75)
    
    # Create results dataframe
    results_df = df.copy()
    results_df['prediction'] = predictions
    results_df['prediction_label'] = results_df['prediction'].map({0: 'Normal', 1: 'Attack'})
    results_df['confidence'] = probabilities
    results_df['confidence_pct'] = results_df['confidence'].apply(lambda x: f"{x*100:.1f}%")
    results_df['attack_type'] = results_df.apply(lambda r: get_attack_type(r['prediction'], r['confidence']), axis=1)
    results_df['risk_level'] = results_df['confidence'].apply(get_risk_level)
    results_df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    results_df['method'] = 'CSV Upload'
    
    # Save to database
    for idx, row in results_df.iterrows():
        save_prediction(
            prediction=int(row['prediction']),
            confidence=float(row['confidence']),
            attack_type=row['attack_type'],
            risk_level=row['risk_level'],
            method='CSV Upload',
            features=row.to_dict()
        )
    
    progress_bar.progress(100)
    status_text.text("Complete!")
    
    # Display results
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown("#### Prediction Results")
    
    # Summary metrics
    total = len(results_df)
    attacks = (results_df['prediction'] == 1).sum()
    normal = (results_df['prediction'] == 0).sum()
    avg_conf = results_df['confidence'].mean()
    
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.markdown(get_metric_card_html("Total Samples", f"{total:,}", "", "blue"), unsafe_allow_html=True)
    with mcol2:
        st.markdown(get_metric_card_html("Attacks Detected", f"{attacks:,}", f"{attacks/total*100:.1f}%", "red"), unsafe_allow_html=True)
    with mcol3:
        st.markdown(get_metric_card_html("Normal Traffic", f"{normal:,}", f"{normal/total*100:.1f}%", "green"), unsafe_allow_html=True)
    with mcol4:
        st.markdown(get_metric_card_html("Avg Confidence", f"{avg_conf*100:.1f}%", "", "orange"), unsafe_allow_html=True)
    
    # Confidence distribution chart
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    fig = plot_prediction_distribution(results_df['confidence'].values, results_df['prediction'].values)
    st.plotly_chart(fig, use_container_width=True)
    
    # Results table
    st.markdown("#### Detailed Results")
    display_cols = ['timestamp', 'prediction_label', 'confidence_pct', 'attack_type', 'risk_level', 'method']
    if len(df.columns) <= 10:
        display_cols = list(df.columns) + display_cols
    
    st.dataframe(
        results_df[display_cols].head(100),
        use_container_width=True,
        hide_index=True
    )
    
    if len(results_df) > 100:
        st.info(f"Showing first 100 of {len(results_df)} results.")
    
    # Download button
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Results as CSV",
        data=csv,
        file_name=f"nids_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.success(f"✅ Batch prediction complete! {total} samples processed, {attacks} attacks detected.")


if __name__ == "__main__":
    show_upload()