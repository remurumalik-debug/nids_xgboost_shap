import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load NSL-KDD ──────────────────────────────────────────────
cols = ["duration","protocol_type","service","flag","src_bytes","dst_bytes",
        "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
        "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
        "num_shells","num_access_files","num_outbound_cmds","is_host_login",
        "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
        "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
        "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
        "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
        "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]

df = pd.read_csv("KDDTrain+.txt", names=cols)
df_test = pd.read_csv("KDDTest+.txt", names=cols)

# ── 2. Label encoding (binary: normal vs attack) ─────────────────
df["label"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)
df_test["label"] = df_test["label"].apply(lambda x: 0 if x == "normal" else 1)
df.drop("difficulty", axis=1, inplace=True)
df_test.drop("difficulty", axis=1, inplace=True)

# ── 3. Encode categorical features ──────────────────────────────
cat_cols = ["protocol_type", "service", "flag"]
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])
    df_test[col] = le.fit_transform(df_test[col])

# ── 4. Split features / labels ──────────────────────────────────
X = df.drop("label", axis=1)
y = df["label"]
X_test = df_test.drop("label", axis=1)
y_test = df_test["label"]

# ── 5. Normalize ─────────────────────────────────────────────────
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# ── 6. SMOTE for class imbalance ─────────────────────────────────
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print(f"After SMOTE: {pd.Series(y_res).value_counts().to_dict()}")

# ── 7. Train XGBoost ─────────────────────────────────────────────
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.01,
    subsample=0.8,
    scale_pos_weight=3,        # add this line
    eval_metric="logloss",
    random_state=42
)
model.fit(X_res, y_res)

# ── 8. Evaluate ──────────────────────────────────────────────────
y_pred = model.predict(X_test)
print("\n===== EVALUATION RESULTS =====")
print(f"Accuracy:  {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred)*100:.2f}%")
print(f"Recall:    {recall_score(y_test, y_pred)*100:.2f}%")
print(f"F1-Score:  {f1_score(y_test, y_pred)*100:.2f}%")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# ── 9. SHAP Explainability ───────────────────────────────────────
print("\nGenerating SHAP explanations...")
explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
shap_values = explainer.shap_values(X_test[:500])  # sample for speed

# Summary plot (global)
shap.summary_plot(shap_values, X_test[:500], plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Global)")
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150)
plt.close()
print("Saved: shap_summary.png")

# Force plot for first prediction (local)
try:
    shap.initjs()
except AssertionError:
    pass

force = shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_test.iloc[0],
    matplotlib=True,
    show=False
)
plt.savefig("shap_force_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_force_plot.png")

print("\nDone! Check shap_summary.png and shap_force_plot.png")