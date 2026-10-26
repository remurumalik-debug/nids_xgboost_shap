import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from utils.predictor import NIDSPredictor
from utils.shap_utils import get_shap_explainer
from utils.visualizations import COLORS

def show_explainability():
    st.markdown("""
        <div class="section-header">
            <div class="icon" style="background: rgba(168, 85, 247, 0.2); color: #a855f7;">🧠</div>
            <h2>SHAP Explainability</h2>
            <div class="divider"></div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style="background: #111827; border: 1px solid #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 24px;">
            <p style="margin: 0; color: #94a3b8;">SHAP (SHapley Additive exPlanations) provides game-theoretic explanations for model predictions. 
            Each feature's contribution to moving the prediction from the expected value to the actual output is calculated using Shapley values.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize predictor
    if "predictor" not in st.session_state:
        with st.spinner("Loading model..."):
            st.session_state.predictor = NIDSPredictor()
    
    predictor = st.session_state.predictor
    feature_columns = predictor.feature_columns
    
    # Sample selector
    st.markdown("#### Sample Selection")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("Select a test sample to explain, or use the manual prediction from the previous page.")
    with col2:
        sample_idx = st.number_input("Test Sample Index", 0, 499, 0, help="Index of test sample to explain")
    
    if st.button("🔍 Generate SHAP Explanation", type="primary", use_container_width=True):
        with st.spinner("Computing SHAP values..."):
            try:
                # Get test data
                X_test = get_test_data(predictor, 500)
                
                if X_test is None:
                    st.error("Could not load test data")
                    return
                
                # Get SHAP explainer
                explainer = get_shap_explainer(predictor.model, feature_columns)
                shap_values = explainer.get_shap_values(X_test, max_samples=500)
                expected_value = explainer.get_expected_value()
                
                # Get sample
                sample = X_test.iloc[sample_idx]
                
                # Local explanation
                local_explanation = explainer.get_local_explanation(shap_values, X_test, sample_idx)
                
                # Prediction
                pred, prob = predictor.predict_single(sample.to_dict())
                
                # Display results
                st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
                
                # Prediction summary
                pred_label = "ATTACK" if pred == 1 else "NORMAL"
                pred_color = COLORS["accent_red"] if pred == 1 else COLORS["accent_green"]
                
                st.markdown(f"""
                    <div class="prediction-card {'attack' if pred == 1 else 'normal'}">
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
                            <div style="display: flex; align-items: center; gap: 12px;">
                                <span style="font-size: 3rem;">{'🔴' if pred == 1 else '🟢'}</span>
                                <div>
                                    <div style="font-size: 1.5rem; font-weight: 700; color: {pred_color};">{pred_label} TRAFFIC</div>
                                    <div style="color: #94a3b8;">Sample #{sample_idx} • Confidence: {prob*100:.1f}%</div>
                                </div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 2rem; font-weight: 700; color: {pred_color};">SHAP</div>
                                <div style="color: #94a3b8;">Expected Value: {expected_value:.4f}</div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Tabs for different visualizations
                tab1, tab2, tab3, tab4 = st.tabs(["💧 Waterfall", "📊 Feature Contribution", "⚡ Force Plot", "📝 Explanation"])
                
                with tab1:
                    st.markdown("#### SHAP Waterfall Plot")
                    st.markdown("Shows how each feature contributes to pushing the prediction from the expected value to the final output.")
                    
                    fig = plt.figure(figsize=(12, 7))
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=shap_values[sample_idx],
                            base_values=expected_value,
                            data=sample.values,
                            feature_names=feature_columns
                        ),
                        show=False
                    )
                    plt.title(f"SHAP Waterfall Plot - Sample {sample_idx}", fontsize=14, fontweight='bold', color='white')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                
                with tab2:
                    st.markdown("#### Feature Contributions")
                    st.markdown("Top features contributing to this prediction (positive = pushes toward attack, negative = pushes toward normal).")
                    
                    # Create contribution chart
                    contrib_df = local_explanation.head(15).iloc[::-1]
                    
                    fig = plt.figure(figsize=(10, 8))
                    colors = [COLORS["accent_red"] if x > 0 else COLORS["accent_green"] for x in contrib_df['shap_value']]
                    plt.barh(contrib_df['feature'], contrib_df['shap_value'], color=colors, edgecolor=COLORS["border"], linewidth=0.5)
                    plt.xlabel('SHAP Value (Impact on Prediction)', color='white')
                    plt.title('Feature Contributions to Prediction', fontsize=14, fontweight='bold', color='white')
                    plt.axvline(x=0, color=COLORS["border"], linewidth=1)
                    plt.gca().set_facecolor(COLORS["bg_card"])
                    plt.gcf().set_facecolor(COLORS["bg_dark"])
                    plt.tick_params(colors='white')
                    for spine in plt.gca().spines.values():
                        spine.set_color(COLORS["border"])
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Table
                    display_df = local_explanation.head(15).copy()
                    display_df['shap_value'] = display_df['shap_value'].apply(lambda x: f"{x:+.4f}")
                    display_df['feature_value'] = display_df['feature_value'].apply(lambda x: f"{x:.4f}")
                    display_df.columns = ['Feature', 'Feature Value', 'SHAP Value']
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                with tab3:
                    st.markdown("#### SHAP Force Plot")
                    st.markdown("Interactive visualization showing feature contributions as forces pushing the prediction.")
                    
                    try:
                        shap.initjs()
                        force_plot = shap.force_plot(
                            expected_value,
                            shap_values[sample_idx],
                            sample,
                            feature_names=feature_columns,
                            matplotlib=False,
                            show=False
                        )
                        st.components.v1.html(shap.getjs() + force_plot.html(), height=400)
                    except Exception as e:
                        st.warning("Interactive force plot requires JavaScript. Showing matplotlib version instead.")
                        fig = plt.figure(figsize=(14, 4))
                        shap.force_plot(
                            expected_value,
                            shap_values[sample_idx],
                            sample,
                            feature_names=feature_columns,
                            matplotlib=True,
                            show=False
                        )
                        plt.title(f"SHAP Force Plot - Sample {sample_idx}", fontsize=14, fontweight='bold', color='white')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                
                with tab4:
                    st.markdown("#### Human-Readable Explanation")
                    
                    # Generate text explanation
                    explanation = generate_text_explanation(local_explanation, pred, prob)
                    
                    st.markdown(f"""
                        <div style="background: #111827; border: 1px solid #1e293b; border-radius: 12px; padding: 24px; margin: 16px 0;">
                            <h4 style="margin: 0 0 16px; color: #f8fafc;">📝 AI Explanation</h4>
                            <p style="margin: 0; color: #e2e8f0; line-height: 1.8; font-size: 1.05rem;">{explanation}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Additional details
                    with st.expander("🔍 Detailed Feature Analysis", expanded=False):
                        for _, row in local_explanation.head(10).iterrows():
                            direction = "increases attack likelihood" if row['shap_value'] > 0 else "decreases attack likelihood"
                            st.markdown(f"""
                                <div style="padding: 12px; background: #111827; border: 1px solid #1e293b; border-radius: 8px; margin: 8px 0;">
                                    <strong style="color: #f8fafc;">{row['feature']}</strong> 
                                    <span style="color: #94a3b8;">= {row['feature_value']:.4f}</span>
                                    <span style="color: {'#ef4444' if row['shap_value'] > 0 else '#10b981'}; margin-left: 12px;">
                                        {'▲' if row['shap_value'] > 0 else '▼'} {abs(row['shap_value']):.4f}
                                    </span>
                                    <br>
                                    <small style="color: #64748b;">This feature {direction} by {abs(row['shap_value'])*100:.2f}%</small>
                                </div>
                            """, unsafe_allow_html=True)
                
                st.success("✅ SHAP explanation generated successfully!")
                
            except Exception as e:
                st.error(f"Error generating SHAP explanation: {str(e)}")
                st.exception(e)


def get_test_data(predictor: NIDSPredictor, n_samples: int = 500):
    """Load and preprocess test data for SHAP"""
    import joblib
    from pathlib import Path
    
    try:
        test_cols = ["duration","protocol_type","service","flag","src_bytes","dst_bytes",
            "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
            "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
            "num_shells","num_access_files","num_outbound_cmds","is_host_login",
            "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
            "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
            "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
            "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
            "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]
        
        df_test = pd.read_csv("KDDTest+.txt", names=test_cols)
        df_test["label"] = df_test["label"].apply(lambda x: 0 if x == "normal" else 1)
        df_test.drop("difficulty", axis=1, inplace=True)
        
        cat_cols = joblib.load(Path(__file__).parent.parent / "model" / "cat_cols.pkl")
        le = joblib.load(Path(__file__).parent.parent / "model" / "encoder.pkl")
        
        for col in cat_cols:
            df_test[col] = le.fit_transform(df_test[col])
        
        X_test = df_test.drop("label", axis=1)
        
        for col in predictor.feature_columns:
            if col not in X_test.columns:
                X_test[col] = 0
        X_test = X_test[predictor.feature_columns]
        
        # Scale
        X_test_scaled = pd.DataFrame(
            predictor.scaler.transform(X_test),
            columns=X_test.columns
        )
        
        return X_test_scaled.iloc[:n_samples]
        
    except Exception as e:
        st.error(f"Error loading test data: {e}")
        return None


if __name__ == "__main__":
    show_explainability()