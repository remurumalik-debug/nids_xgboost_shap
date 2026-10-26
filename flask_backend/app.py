import json
import os
import numpy as np
import shap
import xgboost as xgb
import joblib
from flask import Flask, jsonify, request, send_from_directory
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")

app = Flask(__name__, static_folder="static", template_folder="templates")

model_json_path = os.path.join(MODEL_DIR, "xgb_model.json")
model_joblib_path = os.path.join(MODEL_DIR, "xgb_model.joblib")

model = None
if os.path.exists(model_json_path):
    # load into the sklearn wrapper so TreeExplainer and predict_proba behave as expected
    model = xgb.XGBClassifier()
    model.load_model(model_json_path)
elif os.path.exists(model_joblib_path):
    model = joblib.load(model_joblib_path)
else:
    raise FileNotFoundError(
        "No model artifact found. Expected xgb_model.json or xgb_model.joblib in model/"
    )

# If a raw Booster was saved/loaded (not the sklearn wrapper), wrap it so callers can use
# the sklearn-like API. Also ensure attributes like `n_classes_` and `classes_` exist because
# xgboost's `load_model` may not set them on the wrapper.
if isinstance(model, xgb.core.Booster):
    clf = xgb.XGBClassifier()
    clf._Booster = model
    model = clf

# Fall back to reasonable defaults for sklearn-like attributes when missing
if not hasattr(model, "n_classes_"):
    model.n_classes_ = 2
if not hasattr(model, "classes_"):
    model.classes_ = np.array([0, 1])

# If we somehow don't have a predict_proba method (e.g. older raw Booster), attach a
# lightweight fallback that returns probabilities for binary classifiers.
if not hasattr(model, "predict_proba"):
    booster = getattr(model, "_Booster", None)
    if booster is not None:
        def _predict_proba(X):
            dmat = xgb.DMatrix(X)
            ypred = booster.predict(dmat)
            ypred = np.array(ypred)
            if ypred.ndim == 1:
                probs = np.vstack([1 - ypred, ypred]).T
            else:
                probs = ypred
            return probs

        model.predict_proba = _predict_proba

# Create explainer at startup (avoids pickle/matplotlib issues on Python 3.13)
explainer = shap.TreeExplainer(model)

encoders_json_path = os.path.join(MODEL_DIR, "encoders.json")
encoders_joblib_path = os.path.join(MODEL_DIR, "encoders.joblib")

if os.path.exists(encoders_json_path):
    with open(encoders_json_path) as f:
        encoder_classes = json.load(f)

    encoders = {}
    for col, classes in encoder_classes.items():
        le = LabelEncoder()
        le.classes_ = np.array(classes, dtype=object)
        encoders[col] = le
elif os.path.exists(encoders_joblib_path):
    import joblib
    encoders = joblib.load(encoders_joblib_path)
else:
    raise FileNotFoundError(
        "No encoders artifact found. Expected encoders.json or encoders.joblib in model/"
    )

with open(os.path.join(MODEL_DIR, "feature_cols.json")) as f:
    FEATURE_COLS = json.load(f)
with open(os.path.join(MODEL_DIR, "feature_descriptions.json")) as f:
    FEATURE_DESCRIPTIONS = json.load(f)
with open(os.path.join(MODEL_DIR, "global_importance.json")) as f:
    GLOBAL_IMPORTANCE = json.load(f)
with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
    METRICS = json.load(f)
with open(os.path.join(MODEL_DIR, "sample_flows.json")) as f:
    SAMPLE_FLOWS = json.load(f)
with open(os.path.join(MODEL_DIR, "shap_summary.json")) as f:
    SHAP_SUMMARY = json.load(f)

CATEGORICAL = ["protocol_type", "service", "flag"]
CATEGORY_OPTIONS = {c: list(encoders[c].classes_) for c in CATEGORICAL}

PIPELINE = [
    {
        "id": "data_acquisition",
        "title": "Data Acquisition",
        "icon": "db",
        "summary": "NSL-KDD train/test connection records (41 features + label).",
        "detail": f"{METRICS['n_train'] + METRICS.get('n_val', 0):,} labeled training connections and "
                  f"{METRICS['n_test']:,} held-out test connections, spanning normal traffic and dozens of "
                  "attack sub-types across DoS, Probe, R2L and U2R categories.",
    },
    {
        "id": "preprocessing",
        "title": "Preprocessing",
        "icon": "filter",
        "summary": "Label-encode categoricals, binarize target (normal vs attack).",
        "detail": "protocol_type, service and flag are label-encoded to integers. The multi-class attack "
                  "label is collapsed to a binary target (0 = normal, 1 = attack) for the primary classifier, "
                  "while the original attack-type string is retained for evaluation breakdowns.",
    },
    {
        "id": "feature_selection",
        "title": "Feature Selection",
        "icon": "layers",
        "summary": "All 41 NSL-KDD features retained; importance ranked post-hoc via SHAP.",
        "detail": "Rather than filtering features up front, all 41 are passed to the tree ensemble and "
                  "ranked afterward by mean |SHAP value| — letting the model and its explanation decide "
                  "what matters instead of a manual cut.",
    },
    {
        "id": "model_training",
        "title": "Model Training",
        "icon": "cpu",
        "summary": "XGBoost gradient-boosted trees, 85/15 stratified train/validation split.",
        "detail": "200 trees, max depth 6, learning rate 0.1, subsample/colsample 0.9. Trained with logloss "
                  "eval on a held-out validation split before ever touching the NSL-KDD Test+ set.",
    },
    {
        "id": "shap_analysis",
        "title": "SHAP Analysis",
        "icon": "eye",
        "summary": "TreeExplainer computes exact Shapley values per prediction.",
        "detail": "Unlike model-agnostic SHAP approximations, TreeExplainer computes exact contributions "
                  "for tree ensembles in polynomial time — fast enough to explain every single prediction "
                  "live, not just a sampled subset.",
    },
    {
        "id": "evaluation",
        "title": "Evaluation",
        "icon": "check",
        "summary": "Scored on NSL-KDD Test+, which includes attack types never seen in training.",
        "detail": "Test+ is deliberately adversarial to the model — it injects attack sub-types absent from "
                  "training specifically to test generalization. That's why per-attack-type recall varies "
                  "so widely below.",
    },
]


def encode_row(payload):
    """Turn a UI payload (with human-readable categoricals) into a model row."""
    row = []
    for col in FEATURE_COLS:
        if col in CATEGORICAL:
            val = payload.get(col)
            le = encoders[col]
            if val not in le.classes_:
                val = le.classes_[0]
            row.append(int(le.transform([val])[0]))
        else:
            row.append(float(payload.get(col, 0)))
    return row


@app.route("/")
def index():
    return send_from_directory(app.template_folder, "index.html")


@app.route("/api/meta")
def meta():
    """Everything the frontend needs to build the input form."""
    return jsonify({
        "feature_cols": FEATURE_COLS,
        "descriptions": FEATURE_DESCRIPTIONS,
        "categorical": CATEGORICAL,
        "category_options": CATEGORY_OPTIONS,
        "global_importance": GLOBAL_IMPORTANCE,
        "metrics": METRICS,
    })


@app.route("/api/samples")
def samples():
    return jsonify(SAMPLE_FLOWS)


@app.route("/api/pipeline")
def pipeline():
    return jsonify(PIPELINE)


@app.route("/api/shap_summary")
def shap_summary():
    return jsonify(SHAP_SUMMARY)


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    row = encode_row(payload)
    X = np.array([row])

    proba = float(model.predict_proba(X)[0, 1])
    pred = int(proba >= 0.5)

    shap_vals = explainer.shap_values(X)[0]
    base_value = float(explainer.expected_value)

    contributions = []
    for i, col in enumerate(FEATURE_COLS):
        raw_val = payload.get(col, row[i])
        contributions.append({
            "feature": col,
            "value": raw_val,
            "shap": float(shap_vals[i]),
        })
    contributions.sort(key=lambda d: abs(d["shap"]), reverse=True)

    return jsonify({
        "prediction": "attack" if pred == 1 else "normal",
        "probability": proba,
        "base_value": base_value,
        "contributions": contributions[:12],
        "all_contributions": contributions,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
