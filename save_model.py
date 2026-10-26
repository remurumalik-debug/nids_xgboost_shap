import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings("ignore")

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

df["label"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)
df_test["label"] = df_test["label"].apply(lambda x: 0 if x == "normal" else 1)
df.drop("difficulty", axis=1, inplace=True)
df_test.drop("difficulty", axis=1, inplace=True)

cat_cols = ["protocol_type", "service", "flag"]
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])
    df_test[col] = le.fit_transform(df_test[col])

X = df.drop("label", axis=1)
y = df["label"]
X_test = df_test.drop("label", axis=1)
y_test = df_test["label"]

scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print(f"After SMOTE: {pd.Series(y_res).value_counts().to_dict()}")

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.01,
    subsample=0.8,
    scale_pos_weight=3,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_res, y_res)

y_pred = model.predict(X_test)
print("\n===== EVALUATION RESULTS =====")
print(f"Accuracy:  {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred)*100:.2f}%")
print(f"Recall:    {recall_score(y_test, y_pred)*100:.2f}%")
print(f"F1-Score:  {f1_score(y_test, y_pred)*100:.2f}%")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

joblib.dump(model, "model/xgb_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(le, "model/encoder.pkl")
joblib.dump(cat_cols, "model/cat_cols.pkl")
joblib.dump(X.columns.tolist(), "model/feature_columns.pkl")

print("\nModel and preprocessors saved successfully!")