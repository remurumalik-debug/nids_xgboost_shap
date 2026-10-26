import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings("ignore")

class SHAPExplainer:
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self._init_explainer()
    
    def _init_explainer(self):
        self.explainer = shap.TreeExplainer(
            self.model,
            feature_perturbation="tree_path_dependent"
        )
    
    def get_shap_values(self, X: pd.DataFrame, max_samples: int = 100) -> np.ndarray:
        sample_size = min(max_samples, len(X))
        X_sample = X.iloc[:sample_size]
        shap_values = self.explainer.shap_values(X_sample)
        return shap_values
    
    def get_expected_value(self) -> float:
        return self.explainer.expected_value
    
    def get_feature_importance(self, shap_values: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": mean_abs_shap
        }).sort_values("importance", ascending=False)
        return importance_df
    
    def get_local_explanation(self, shap_values: np.ndarray, X: pd.DataFrame, idx: int = 0) -> pd.DataFrame:
        local_shap = shap_values[idx]
        local_features = X.iloc[idx].values
        explanation_df = pd.DataFrame({
            "feature": self.feature_names,
            "feature_value": local_features,
            "shap_value": local_shap
        }).sort_values("shap_value", key=abs, ascending=False)
        return explanation_df

@st.cache_resource
def get_shap_explainer(model, feature_names):
    return SHAPExplainer(model, feature_names)

def generate_shap_summary_plot(shap_values: np.ndarray, X: pd.DataFrame, feature_names: List[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, 
        X, 
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.title("SHAP Feature Importance (Global)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def generate_shap_waterfall_plot(shap_values: np.ndarray, X: pd.DataFrame, feature_names: List[str], idx: int = 0, expected_value: float = 0) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[idx],
            base_values=expected_value,
            data=X.iloc[idx].values,
            feature_names=feature_names
        ),
        show=False
    )
    plt.title(f"SHAP Waterfall Plot - Sample {idx}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def generate_shap_force_plot(shap_values: np.ndarray, X: pd.DataFrame, feature_names: List[str], idx: int = 0, expected_value: float = 0) -> str:
    shap.initjs()
    force_plot = shap.force_plot(
        expected_value,
        shap_values[idx],
        X.iloc[idx],
        feature_names=feature_names,
        matplotlib=False,
        show=False
    )
    return shap.getjs() + force_plot.html()

def generate_text_explanation(shap_explanation: pd.DataFrame, prediction: int, probability: float, top_n: int = 5) -> str:
    top_features = shap_explanation.head(top_n)
    
    if prediction == 0:
        base_text = "The model classified this traffic as **Normal** with {:.1%} confidence. ".format(probability)
        direction = "support normal classification"
    else:
        base_text = "The model classified this traffic as an **Attack** with {:.1%} confidence. ".format(probability)
        direction = "contributed to attack classification"
    
    feature_texts = []
    for _, row in top_features.iterrows():
        feat = row['feature']
        val = row['feature_value']
        shap_val = row['shap_value']
        impact = "increased" if shap_val > 0 else "decreased"
        feature_texts.append(f"**{feat}** (value: {val:.3f}) {impact} the prediction")
    
    explanation = base_text + f"The top {top_n} features that {direction} are: " + ", ".join(feature_texts) + "."
    
    return explanation