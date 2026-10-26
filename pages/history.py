import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from utils.database import get_prediction_stats, get_all_predictions, search_predictions, export_predictions_csv, clear_history
from utils.visualizations import COLORS, get_metric_card_html, plot_prediction_distribution

def show_history():
    st.markdown("""
        <div class="section-header">
            <div class="icon">📜</div>
            <h2>Prediction History</h2>
            <div class="divider"></div>
        </div>
    """, unsafe_allow_html=True)
    
    # Stats
    stats = get_prediction_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(get_metric_card_html("Total", f"{stats['total']:,}", "", "blue"), unsafe_allow_html=True)
    with col2:
        st.markdown(get_metric_card_html("Attacks", f"{stats['attacks']:,}", f"{stats['attacks']/max(stats['total'],1)*100:.1f}%", "red"), unsafe_allow_html=True)
    with col3:
        st.markdown(get_metric_card_html("Normal", f"{stats['normal']:,}", f"{stats['normal']/max(stats['total'],1)*100:.1f}%", "green"), unsafe_allow_html=True)
    with col4:
        st.markdown(get_metric_card_html("Avg Confidence", f"{stats['avg_confidence']:.1f}%", "", "orange"), unsafe_allow_html=True)
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # Filters
    st.markdown("#### Filters")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pred_filter = st.selectbox("Prediction", ["All", "Attack", "Normal"], index=0)
    with col2:
        risk_filter = st.selectbox("Risk Level", ["All", "Critical", "High", "Medium", "Low", "Minimal"], index=0)
    with col3:
        method_filter = st.selectbox("Method", ["All", "Manual", "CSV Upload"], index=0)
    with col4:
        limit = st.selectbox("Limit", [50, 100, 500, 1000], index=1)
    
    # Date filters
    col1, col2 = st.columns(2)
    with col1:
        date_from = st.date_input("From Date", value=None)
    with col2:
        date_to = st.date_input("To Date", value=None)
    
    # Apply filters
    pred_val = None if pred_filter == "All" else (1 if pred_filter == "Attack" else 0)
    risk_val = None if risk_filter == "All" else risk_filter
    method_val = None if method_filter == "All" else method_filter
    date_from_val = date_from.strftime('%Y-%m-%d') if date_from else None
    date_to_val = date_to.strftime('%Y-%m-%d') if date_to else None
    
    if st.button("🔍 Apply Filters", type="primary", use_container_width=True):
        st.session_state.history_filters = {
            'prediction': pred_val,
            'risk': risk_val,
            'method': method_val,
            'date_from': date_from_val,
            'date_to': date_to_val,
            'limit': limit
        }
        st.rerun()
    
    # Get filtered results
    filters = st.session_state.get('history_filters', {
        'prediction': None,
        'risk': None,
        'method': None,
        'date_from': None,
        'date_to': None,
        'limit': 100
    })
    
    df = search_predictions(
        prediction_filter=filters['prediction'],
        risk_filter=filters['risk'],
        method_filter=filters['method'],
        date_from=filters['date_from'],
        date_to=filters['date_to'],
        limit=filters['limit']
    )
    
    if not df.empty:
        # Format for display
        df_display = df.copy()
        df_display['timestamp'] = pd.to_datetime(df_display['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        df_display['prediction'] = df_display['prediction'].map({0: '🟢 Normal', 1: '🔴 Attack'})
        df_display['confidence'] = df_display['confidence'].apply(lambda x: f"{x*100:.1f}%")
        
        st.markdown(f"#### Results ({len(df)} records)")
        st.dataframe(
            df_display[['timestamp', 'prediction', 'confidence', 'attack_type', 'risk_level', 'method']],
            use_container_width=True,
            hide_index=True
        )
        
        # Export
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            csv = export_predictions_csv()
            st.download_button(
                label="📥 Export All as CSV",
                data=csv,
                file_name=f"nids_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col2:
            if st.button("🗑️ Clear History", type="secondary", use_container_width=True):
                if st.checkbox("Confirm clear all history"):
                    clear_history()
                    st.success("History cleared!")
                    st.rerun()
        
        # Confidence distribution for filtered results
        if len(df) > 1:
            st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
            st.markdown("#### Confidence Distribution")
            fig = plot_prediction_distribution(df['confidence'].values, df['prediction'].map({'🟢 Normal': 0, '🔴 Attack': 1}).values)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No predictions found. Make some predictions to see history!")
    
    # Show all history button
    if filters['limit'] < 1000:
        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        if st.button("📋 Show More (1000)", use_container_width=True):
            filters['limit'] = 1000
            st.session_state.history_filters = filters
            st.rerun()


if __name__ == "__main__":
    show_history()