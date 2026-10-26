"""
Train an XGBoost intrusion-detection classifier on NSL-KDD and
precompute a SHAP explainer + background data for fast, explainable
inference in the web app.
"""
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix,
)
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
]

# Human-readable descriptions used in the UI tooltips
FEATURE_DESCRIPTIONS = {
    "duration": "Length of the connection (seconds)",
    "protocol_type": "Transport protocol (tcp/udp/icmp)",
    "service": "Destination network service (http, ftp, etc.)",
    "flag": "Status flag of the connection",
    "src_bytes": "Bytes sent from source to destination",
    "dst_bytes": "Bytes sent from destination to source",
    "land": "1 if connection is from/to the same host/port",
    "wrong_fragment": "Number of wrong fragments",
    "urgent": "Number of urgent packets",
    "hot": "Number of 'hot' indicators (suspicious activity)",
    "num_failed_logins": "Number of failed login attempts",
    "logged_in": "1 if successfully logged in",
    "num_compromised": "Number of 'compromised' conditions",
    "root_shell": "1 if root shell obtained",
    "su_attempted": "1 if 'su root' command attempted",
    "num_root": "Number of root accesses",
    "num_file_creations": "Number of file creation operations",
    "num_shells": "Number of shell prompts",
    "num_access_files": "Number of operations on access control files",
    "num_outbound_cmds": "Number of outbound commands in an ftp session",
    "is_host_login": "1 if login belongs to 'host' list",
    "is_guest_login": "1 if login is a 'guest' login",
    "count": "Connections to same host in last 2s",
    "srv_count": "Connections to same service in last 2s",
    "serror_rate": "% of connections with SYN errors",
    "srv_serror_rate": "% of connections with SYN errors (same service)",
    "rerror_rate": "% of connections with REJ errors",
    "srv_rerror_rate": "% of connections with REJ errors (same service)",
    "same_srv_rate": "% of connections to the same service",
    "diff_srv_rate": "% of connections to different services",
    "srv_diff_host_rate": "% of connections to different hosts (same service)",
    "dst_host_count": "Connections to same destination host",
    "dst_host_srv_count": "Connections to same destination host + service",
    "dst_host_same_srv_rate": "% same service at destination host",
    "dst_host_diff_srv_rate": "% different services at destination host",
    "dst_host_same_src_port_rate": "% same source port at destination host",
    "dst_host_srv_diff_host_rate": "% different hosts, same service, dest host",
    "dst_host_serror_rate": "% SYN errors at destination host",
    "dst_host_srv_serror_rate": "% SYN errors, same service, destination host",
    "dst_host_rerror_rate": "% REJ errors at destination host",
    "dst_host_srv_rerror_rate": "% REJ errors, same service, destination host",
}

CATEGORICAL = ["protocol_type", "service", "flag"]


def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, "KDDTrain.txt"), names=COLUMNS)
    test = pd.read_csv(os.path.join(DATA_DIR, "KDDTest.txt"), names=COLUMNS)
    train = train.drop(columns=["difficulty"])
    test = test.drop(columns=["difficulty"])
    return train, test


def main():
    train, test = load_data()

    # Binary target: normal vs attack
    train["binary_label"] = (train["label"] != "normal").astype(int)
    test["binary_label"] = (test["label"] != "normal").astype(int)

    full = pd.concat([train, test], ignore_index=True)

    encoders = {}
    for col in CATEGORICAL:
        le = LabelEncoder()
        full[col] = le.fit_transform(full[col].astype(str))
        encoders[col] = le

    train_n = len(train)
    train = full.iloc[:train_n].reset_index(drop=True)
    test = full.iloc[train_n:].reset_index(drop=True)

    feature_cols = [c for c in COLUMNS if c not in ("label", "difficulty")]

    X = train[feature_cols]
    y = train["binary_label"]
    X_test = test[feature_cols]
    y_test = test["binary_label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]

    print("=== Validation ===")
    print(classification_report(y_val, val_pred, target_names=["normal", "attack"]))
    print("=== Held-out NSL-KDD Test+ ===")
    print(classification_report(y_test, test_pred, target_names=["normal", "attack"]))
    print("Test accuracy:", accuracy_score(y_test, test_pred))
    print("Test ROC-AUC:", roc_auc_score(y_test, test_proba))

    val_proba = model.predict_proba(X_val)[:, 1]
    train_pred_full = model.predict(X_train)
    train_proba_full = model.predict_proba(X_train)[:, 1]

    cm = confusion_matrix(y_test, test_pred).tolist()  # [[TN, FP], [FN, TP]]

    # SHAP explainer (TreeExplainer is exact + fast for XGBoost)
    explainer = shap.TreeExplainer(model)

    # Background/reference sample for global summary + realistic sample flows in the UI
    rng = np.random.RandomState(42)
    bg_idx = rng.choice(len(X_train), size=min(200, len(X_train)), replace=False)
    background = X_train.iloc[bg_idx]

    global_shap = explainer.shap_values(background)
    global_importance = pd.Series(
        np.abs(global_shap).mean(axis=0), index=feature_cols
    ).sort_values(ascending=False)

    # Save artifacts
    # Use the underlying booster to avoid XGBoost wrapper issues with _estimator_type.
    model.get_booster().save_model(os.path.join(MODEL_DIR, "xgb_model.json"))
    # explainer is recreated at startup in app.py (avoids pickle issues)
    encoder_classes = {col: encoders[col].classes_.tolist() for col in CATEGORICAL}
    with open(os.path.join(MODEL_DIR, "encoders.json"), "w") as f:
        json.dump(encoder_classes, f)

    with open(os.path.join(MODEL_DIR, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f)

    with open(os.path.join(MODEL_DIR, "feature_descriptions.json"), "w") as f:
        json.dump(FEATURE_DESCRIPTIONS, f)

    with open(os.path.join(MODEL_DIR, "global_importance.json"), "w") as f:
        json.dump(
            [{"feature": k, "importance": float(v)} for k, v in global_importance.items()],
            f,
        )

    def split_metrics(y_true, y_pred, y_proba):
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred)),
            "recall": float(recall_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred)),
            "auc": float(roc_auc_score(y_true, y_proba)),
        }

    # attack-type breakdown on the test set: recall per NSL-KDD attack label
    attack_type_recall = []
    test_labels = test["label"]
    for lab in sorted(test_labels.unique()):
        mask = test_labels == lab
        if mask.sum() == 0:
            continue
        y_true_sub = test.loc[mask, "binary_label"]
        y_pred_sub = test_pred[mask.values]
        detected = int((y_pred_sub == 1).sum()) if lab != "normal" else int((y_pred_sub == 0).sum())
        attack_type_recall.append({
            "label": lab,
            "count": int(mask.sum()),
            "detected": detected,
            "rate": round(detected / int(mask.sum()), 4),
        })
    attack_type_recall.sort(key=lambda d: -d["count"])

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(
            {
                "train": split_metrics(y_train, train_pred_full, train_proba_full),
                "val": split_metrics(y_val, val_pred, val_proba),
                "test": split_metrics(y_test, test_pred, test_proba),
                "test_accuracy": float(accuracy_score(y_test, test_pred)),
                "test_auc": float(roc_auc_score(y_test, test_proba)),
                "n_train": int(len(X_train)),
                "n_val": int(len(X_val)),
                "n_test": int(len(X_test)),
                "confusion_matrix": {
                    "labels": ["normal", "attack"],
                    "matrix": cm,  # [[TN, FP],[FN, TP]]
                },
                "attack_type_breakdown": attack_type_recall,
                "hyperparameters": {
                    "n_estimators": 200,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "objective": "binary:logistic",
                },
            },
            f,
        )

    # ---- SHAP summary (beeswarm-style) data for the top features ----
    top_features = list(global_importance.index[:12])
    top_idx = [feature_cols.index(f) for f in top_features]
    summary_bg_idx = rng.choice(len(X_train), size=min(150, len(X_train)), replace=False)
    summary_bg = X_train.iloc[summary_bg_idx]
    summary_shap = explainer.shap_values(summary_bg)

    shap_summary = []
    for f, i in zip(top_features, top_idx):
        vals = summary_bg[f].values.astype(float)
        vmin, vmax = float(vals.min()), float(vals.max())
        rng_span = (vmax - vmin) or 1.0
        norm_vals = ((vals - vmin) / rng_span).tolist()
        shap_summary.append({
            "feature": f,
            "shap_values": summary_shap[:, i].astype(float).tolist(),
            "normalized_feature_values": [round(v, 3) for v in norm_vals],
        })

    with open(os.path.join(MODEL_DIR, "shap_summary.json"), "w") as f:
        json.dump(shap_summary, f)

    # Save a pool of real sample flows (with labels + attack type) for the UI's
    # "load example" feature - mix of normal + a variety of attack categories.
    sample_pool = test.copy()
    sample_pool["pred_proba"] = test_proba
    sample_pool["attack_type"] = test["label"]

    samples = []
    # a handful of normal + a spread of distinct attack labels
    normal_samples = sample_pool[sample_pool["binary_label"] == 0].sample(
        6, random_state=1
    )
    attack_labels = sample_pool[sample_pool["binary_label"] == 1]["attack_type"].unique()
    attack_samples_frames = []
    for lab in attack_labels[:10]:
        subset = sample_pool[sample_pool["attack_type"] == lab]
        attack_samples_frames.append(subset.sample(1, random_state=1))
    attack_samples = pd.concat(attack_samples_frames) if attack_samples_frames else sample_pool.iloc[0:0]

    combined = pd.concat([normal_samples, attack_samples]).reset_index(drop=True)

    # decode categoricals back to human-readable strings for display
    for col in CATEGORICAL:
        combined[col + "_label"] = encoders[col].inverse_transform(combined[col].astype(int))

    for _, row in combined.iterrows():
        rec = {c: (row[c] if not isinstance(row[c], np.generic) else row[c].item()) for c in feature_cols}
        rec["protocol_type_label"] = row["protocol_type_label"]
        rec["service_label"] = row["service_label"]
        rec["flag_label"] = row["flag_label"]
        rec["true_label"] = row["attack_type"]
        rec["true_binary"] = int(row["binary_label"])
        samples.append(rec)

    with open(os.path.join(MODEL_DIR, "sample_flows.json"), "w") as f:
        json.dump(samples, f)

    print("\nSaved model artifacts to", MODEL_DIR)


if __name__ == "__main__":
    main()
