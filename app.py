import os
import uuid
import datetime
import logging
import json
import re
import time
import sqlite3
import hashlib
import shap
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from collections import Counter

from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for, flash, Response, stream_with_context
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from imblearn.over_sampling import SMOTE
from werkzeug.utils import secure_filename
from sklearn.metrics import (accuracy_score, f1_score, mean_squared_error,
                              r2_score, roc_curve, auc, confusion_matrix,
                              precision_recall_curve, average_precision_score)
import joblib

# ---------------------------------------------------------------------------
# Logging  (replaces all print() debug calls)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask app setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
# In production set FLASK_SECRET_KEY in .env — never leave it random
_secret = os.environ.get("FLASK_SECRET_KEY")
if not _secret:
    if os.environ.get("FLASK_ENV") == "production":
        raise RuntimeError("FLASK_SECRET_KEY must be set in production")
    _secret = "dev-only-insecure-key-change-in-production"
app.secret_key = _secret

# FIX #10 — enforce a 50 MB upload cap
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MODEL_FOLDER"]  = "models"
app.config["SCALER_FOLDER"] = "scalers"
app.config["GRAPH_FOLDER"]  = "graphs"
app.config["REPORT_FOLDER"] = "reports"

for folder in ["UPLOAD_FOLDER", "MODEL_FOLDER", "SCALER_FOLDER",
               "GRAPH_FOLDER", "REPORT_FOLDER"]:
    os.makedirs(app.config[folder], exist_ok=True)

DB_PATH = "automl.db"
ALLOWED_EXTENSIONS = {"csv"}

# ---------------------------------------------------------------------------
# Database — auth + run history
# ---------------------------------------------------------------------------

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created  TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS runs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER NOT NULL,
            created       TEXT NOT NULL,
            dataset_name  TEXT,
            target_column TEXT,
            task_type     TEXT,
            best_model    TEXT,
            best_score    REAL,
            n_models      INTEGER,
            report_id     TEXT,
            meta_path     TEXT,
            results_json  TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
    """)
    conn.commit()
    conn.close()

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def current_user_id():
    return session.get("user_id")

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user_id():
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

init_db()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    """FIX #10 — backend file-type validation."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def unique_filename(original: str) -> str:
    """FIX #9 — prepend a UUID so concurrent uploads never collide."""
    safe = secure_filename(original)
    return f"{uuid.uuid4().hex}_{safe}"


def detect_classification(y: pd.Series) -> bool:
    """
    FIX #5 — single authoritative place for task-type detection.
    Uses dtype + unique-value count so numeric regression targets with
    fewer than 10 distinct values are not misclassified.
    """
    n_unique = len(y.dropna().unique())
    if pd.api.types.is_float_dtype(y) and n_unique > 10:
        return False
    return n_unique < 10


def _encode_plot() -> str:
    """Save the current matplotlib figure to a base64 PNG string."""
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()
    return encoded


# ---------------------------------------------------------------------------
# Performance bar chart
# ---------------------------------------------------------------------------

def generate_performance_plot(results: dict, is_classification: bool) -> str:
    # Use accuracy for classification, R2 for regression (both higher = better)
    metric = "accuracy" if is_classification else "r2"
    labels = list(results.keys())
    scores = [v[metric] for v in results.values()]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, scores, color="skyblue")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Accuracy (%)" if is_classification else "R2 Score")
    plt.title("Model Performance Comparison")
    if not is_classification:
        plt.axhline(y=0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
        plt.ylim(min(min(scores) - 0.1, -0.1), 1.05)
    plt.tight_layout()

    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.4f}",
            ha="center",
            va="bottom",
        )

    return _encode_plot()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    available_models = {
        "random_forest":      "Random Forest",
        "xgboost":            "XGBoost",
        "lightgbm":           "LightGBM",
        "logistic_regression":"Logistic Regression",
        "knn":                "KNN",
        "svm":                "SVM",
        "decision_tree":      "Decision Tree",
        "gradient_boosting":  "Gradient Boosting",
    }
    return render_template("index.html",
                           available_models=available_models,
                           logged_in=(current_user_id() is not None),
                           username=session.get("username", ""))


@app.route("/get_columns", methods=["POST"])
def get_columns():
    """Return cleaned column names + lightweight EDA stats for the upload page."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "Only CSV files are accepted"}), 400

        df = pd.read_csv(file)
        df.columns = (
            df.columns
            .str.replace(r"[^a-zA-Z0-9_]", "", regex=True)
            .str.strip()
        )

        # ── Dataset-level stats ──────────────────────────────────────────────
        n_rows, n_cols = df.shape
        missing_counts = df.isnull().sum()
        total_missing  = int(missing_counts.sum())
        n_duplicates   = int(df.duplicated().sum())

        # ── Per-column summary ───────────────────────────────────────────────
        col_info = []
        for col in df.columns:
            col_info.append({
                "name":    col,
                "dtype":   str(df[col].dtype),
                "missing": int(missing_counts[col]),
                "unique":  int(df[col].nunique()),
            })

        # ── First 5 rows as a list of dicts (JSON-serialisable) ──────────────
        preview = df.head(5).astype(str).to_dict(orient="list")

        # ── Correlation heatmap (numeric cols only) ──────────────────────────
        corr_plot = None
        num_df = df.select_dtypes(include="number")
        if num_df.shape[1] >= 2:
            corr = num_df.corr()
            fig_h = max(4, min(corr.shape[0] * 0.55, 12))
            fig_w = max(5, min(corr.shape[1] * 0.65, 14))
            plt.figure(figsize=(fig_w, fig_h))
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(
                corr, mask=mask, annot=corr.shape[0] <= 12,
                fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.4, annot_kws={"size": 8},
            )
            plt.title("Feature Correlation Heatmap", pad=10)
            plt.tight_layout()
            corr_plot = _encode_plot()

        return jsonify({
            "columns":       df.columns.tolist(),
            "n_rows":        n_rows,
            "n_cols":        n_cols,
            "total_missing": total_missing,
            "n_duplicates":  n_duplicates,
            "col_info":      col_info,
            "preview":       preview,
            "corr_plot":     corr_plot,
        })

    except Exception as exc:
        logger.exception("get_columns failed")
        return jsonify({"error": f"Error reading file: {exc}"}), 500


@app.route("/eda_target", methods=["POST"])
def eda_target():
    """Return a target-distribution chart after the user selects a target column."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files["file"]
        target_column = request.form.get("target_column", "").strip()
        if not allowed_file(file.filename):
            return jsonify({"error": "Only CSV files are accepted"}), 400

        df = pd.read_csv(file)
        df.columns = (
            df.columns
            .str.replace(r"[^a-zA-Z0-9_]", "", regex=True)
            .str.strip()
        )
        if target_column not in df.columns:
            return jsonify({"error": "Target column not found"}), 400

        y = df[target_column].dropna()
        is_cls = detect_classification(y)

        plt.figure(figsize=(7, 3.5))
        if is_cls:
            counts = y.astype(str).value_counts().sort_index()
            bars = plt.bar(counts.index.astype(str), counts.values, color="steelblue")
            for bar, val in zip(bars, counts.values):
                plt.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.3, str(val),
                         ha="center", va="bottom", fontsize=9)
            plt.xlabel("Class"); plt.ylabel("Count")
            plt.title(f"Target Distribution — {target_column}")
        else:
            sns.histplot(y, kde=True, color="steelblue", bins=30)
            plt.xlabel(target_column); plt.ylabel("Frequency")
            plt.title(f"Target Distribution — {target_column}")
        plt.tight_layout()
        dist_plot = _encode_plot()

        return jsonify({
            "task_type":  "classification" if is_cls else "regression",
            "dist_plot":  dist_plot,
            "target_min": round(float(y.min()), 4) if not is_cls else None,
            "target_max": round(float(y.max()), 4) if not is_cls else None,
            "target_mean":round(float(y.mean()), 4) if not is_cls else None,
            "n_classes":  int(y.nunique()) if is_cls else None,
        })

    except Exception as exc:
        logger.exception("eda_target failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/check_target_column", methods=["POST"])
def check_target_column():
    """
    Detect task type and return the appropriate model list.
    FIX #5 — uses detect_classification() so logic matches preprocessing.
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files["file"]
        target_column = request.form.get("target_column", "").strip()

        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        if not target_column:
            return jsonify({"error": "Target column is required"}), 400
        # FIX #10 — validate extension on backend
        if not allowed_file(file.filename):
            return jsonify({"error": "Only CSV files are accepted"}), 400

        df = pd.read_csv(file)
        # FIX #12
        df.columns = (
            df.columns
            .str.replace(r"[^a-zA-Z0-9_]", "", regex=True)
            .str.strip()
        )
        if target_column not in df.columns:
            return jsonify({"error": "Target column not found in dataset"}), 400

        # FIX #5 — single shared helper
        is_classification = detect_classification(df[target_column])

        classification_models = [
            {"key": "random_forest",       "name": "Random Forest"},
            {"key": "xgboost",             "name": "XGBoost"},
            {"key": "lightgbm",            "name": "LightGBM"},
            {"key": "logistic_regression", "name": "Logistic Regression"},
            {"key": "knn",                 "name": "KNN"},
            {"key": "svm",                 "name": "SVM"},
            {"key": "decision_tree",       "name": "Decision Tree"},
            {"key": "gradient_boosting",   "name": "Gradient Boosting"},
        ]
        regression_models = [
            {"key": "random_forest",     "name": "Random Forest"},
            {"key": "xgboost",           "name": "XGBoost"},
            {"key": "lightgbm",          "name": "LightGBM"},
            {"key": "knn",               "name": "KNN"},
            {"key": "svr",               "name": "SVR"},
            {"key": "decision_tree",     "name": "Decision Tree"},
            {"key": "gradient_boosting", "name": "Gradient Boosting"},
        ]

        return jsonify({
            "task_type": "classification" if is_classification else "regression",
            "models":    classification_models if is_classification else regression_models,
        })

    except Exception as exc:
        logger.exception("check_target_column failed")
        return jsonify({"error": f"Error processing dataset: {exc}"}), 500




# ---------------------------------------------------------------------------
# SSE — Server-Sent Events job store for live training progress
# ---------------------------------------------------------------------------
import queue
import threading

# Per-job: {"status": str, "log": [str], "result_url": str|None, "error": str|None}
_sse_jobs: dict = {}
_sse_queues: dict = {}   # job_id → queue.Queue  (used to push events)


def _sse_event(q: queue.Queue, msg: str, event: str = "message") -> None:
    """Push a single SSE-formatted message into the queue."""
    q.put(f"event: {event}\ndata: {msg}\n\n")


@app.route("/train_stream", methods=["POST"])
def train_stream():
    """
    Accept multipart form (same fields as /upload), save file, start a
    background training thread, return a job_id immediately.
    The client then connects to /train_progress/<job_id> for SSE.
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
        file            = request.files["file"]
        target_column   = request.form.get("target_column", "").strip()
        selected_models = request.form.getlist("selected_models")

        if not allowed_file(file.filename):
            return jsonify({"error": "Only CSV files accepted"}), 400
        if not target_column:
            return jsonify({"error": "target_column required"}), 400
        if not selected_models:
            return jsonify({"error": "Select at least one model"}), 400

        filename  = unique_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        model_name_mapping = {
            "random_forest": "Random Forest", "xgboost": "XGBoost",
            "lightgbm": "LightGBM", "logistic_regression": "Logistic Regression",
            "knn": "KNN", "svm": "SVM", "decision_tree": "Decision Tree",
            "gradient_boosting": "Gradient Boosting", "svr": "SVR",
        }
        selected_model_names = [
            model_name_mapping[m] for m in selected_models if m in model_name_mapping
        ]

        job_id = uuid.uuid4().hex
        q      = queue.Queue()
        _sse_jobs[job_id]   = {"status": "running", "log": [], "result_url": None, "error": None}
        _sse_queues[job_id] = q

        # Snapshot the user id now (session unavailable in background thread)
        uid      = current_user_id()
        username = session.get("username", "")

        def _bg():
            try:
                _sse_event(q, "📂 Reading and preprocessing dataset…")

                df = pd.read_csv(file_path)
                df.columns = (df.columns.str.replace(r"[^a-zA-Z0-9_]", "", regex=True).str.strip())
                X, y, imputer, scaler, is_classification = clean_and_preprocess_data(df, target_column)
                n_feat = X.shape[1]
                _sse_event(q, f"✅ Preprocessed — {X.shape[0]} rows × {n_feat} features")

                # ── Per-model training with timing ──────────────────────────
                scoring = "f1_weighted" if is_classification else "neg_mean_squared_error"
                stratify = y if is_classification else None
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=stratify)

                if is_classification and len(np.unique(y_train)) > 1:
                    class_counts    = Counter(y_train)
                    min_class_count = min(class_counts.values())
                    if min_class_count >= 2:
                        n_neighbors = max(1, min(5, min_class_count - 1))
                        smote = SMOTE(random_state=42, k_neighbors=n_neighbors)
                        X_train, y_train = smote.fit_resample(X_train, y_train)
                        _sse_event(q, f"⚖️  SMOTE applied (k_neighbors={n_neighbors})")

                from sklearn.model_selection import RandomizedSearchCV
                from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

                if is_classification:
                    param_grids = {
                        "Random Forest":      {"model": RandomForestClassifier(random_state=42),
                                               "params": {"n_estimators": [100, 200], "max_depth": [None, 10]}},
                        "XGBoost":            {"model": XGBClassifier(eval_metric="mlogloss", random_state=42),
                                               "params": {"n_estimators": [100], "learning_rate": [0.01, 0.1]}},
                        "LightGBM":           {"model": LGBMClassifier(random_state=42),
                                               "params": {"n_estimators": [100], "learning_rate": [0.01, 0.1]}},
                        "Logistic Regression":{"model": LogisticRegression(max_iter=1000, random_state=42),
                                               "params": {"C": [0.1, 1.0], "solver": ["liblinear"]}},
                        "KNN":                {"model": KNeighborsClassifier(),
                                               "params": {"n_neighbors": [3, 5], "weights": ["uniform", "distance"]}},
                        "SVM":                {"model": SVC(probability=True, random_state=42),
                                               "params": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}},
                        "Decision Tree":      {"model": DecisionTreeClassifier(random_state=42),
                                               "params": {"max_depth": [None, 10, 20], "min_samples_split": [2, 5]}},
                        "Gradient Boosting":  {"model": GradientBoostingClassifier(random_state=42),
                                               "params": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]}},
                    }
                else:
                    param_grids = {
                        "Random Forest":     {"model": RandomForestRegressor(random_state=42),
                                              "params": {"n_estimators": [100, 200], "max_depth": [None, 10]}},
                        "XGBoost":           {"model": XGBRegressor(eval_metric="rmse", random_state=42),
                                              "params": {"n_estimators": [100], "learning_rate": [0.01, 0.1]}},
                        "LightGBM":          {"model": LGBMRegressor(random_state=42),
                                              "params": {"n_estimators": [100], "learning_rate": [0.01, 0.1]}},
                        "KNN":               {"model": KNeighborsRegressor(),
                                              "params": {"n_neighbors": [3, 5], "weights": ["uniform", "distance"]}},
                        "SVR":               {"model": SVR(),
                                              "params": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}},
                        "Decision Tree":     {"model": DecisionTreeRegressor(random_state=42),
                                              "params": {"max_depth": [None, 10, 20], "min_samples_split": [2, 5]}},
                        "Gradient Boosting": {"model": GradientBoostingRegressor(random_state=42),
                                              "params": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]}},
                    }

                param_grids = {k: v for k, v in param_grids.items() if k in selected_model_names}
                total   = len(param_grids)
                results = {}
                best_score, best_model_obj, best_model_name = -np.inf, None, ""

                for idx, (name, model_info) in enumerate(param_grids.items(), 1):
                    _sse_event(q, f"🔄 [{idx}/{total}] Training {name}…", event="progress")
                    model      = model_info["model"]
                    param_grid = dict(model_info["params"])

                    if name == "KNN":
                        filtered = [n for n in param_grid["n_neighbors"] if n <= len(X_train)]
                        param_grid["n_neighbors"] = filtered or [max(1, len(X_train))]

                    search = RandomizedSearchCV(model, param_grid, n_iter=5, cv=3,
                                               scoring=scoring, n_jobs=-1, random_state=42)
                    t0 = time.perf_counter()
                    search.fit(X_train, y_train)
                    secs       = round(time.perf_counter() - t0, 2)
                    final_model = search.best_estimator_
                    y_pred      = final_model.predict(X_test)

                    if is_classification:
                        acc  = round(accuracy_score(y_test, y_pred) * 100, 2)
                        f1   = round(f1_score(y_test, y_pred, average="weighted") * 100, 2)
                        score = f1 / 100
                        results[name] = {"accuracy": acc, "f1_score": f1,
                                         "best_params": search.best_params_, "train_time": secs}
                        _sse_event(q, f"✅ {name} — Accuracy {acc}%, F1 {f1}% ({secs}s)", event="progress")
                    else:
                        rmse  = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
                        r2    = round(r2_score(y_test, y_pred), 4)
                        score = r2
                        results[name] = {"rmse": rmse, "r2": r2,
                                         "best_params": search.best_params_, "train_time": secs}
                        _sse_event(q, f"✅ {name} — R² {r2}, RMSE {rmse} ({secs}s)", event="progress")

                    if score > best_score:
                        best_score, best_model_obj, best_model_name = score, final_model, name

                # ── Save artefacts ──────────────────────────────────────────
                _sse_event(q, f"💾 Saving best model ({best_model_name})…")
                timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path  = os.path.join(app.config["MODEL_FOLDER"],  f"{best_model_name}_{timestamp}.pkl")
                scaler_path = os.path.join(app.config["SCALER_FOLDER"], f"scaler_{best_model_name}_{timestamp}.pkl")
                imp_path    = os.path.join(app.config["SCALER_FOLDER"], f"imputer_{best_model_name}_{timestamp}.pkl")
                joblib.dump(best_model_obj, model_path)
                joblib.dump(scaler, scaler_path)
                joblib.dump(imputer, imp_path)

                meta = {
                    "feature_names": list(X.columns), "is_classification": is_classification,
                    "model_name": best_model_name, "scaler_path": scaler_path,
                    "imputer_path": imp_path, "model_path": model_path,
                }
                meta_path = model_path.replace(".pkl", "_meta.json")
                with open(meta_path, "w") as mf:
                    json.dump(meta, mf)

                # ── Save to history DB if user was logged in ────────────────
                if uid:
                    best_score_val = (results[best_model_name]["accuracy"]
                                      if is_classification else results[best_model_name]["r2"])
                    report_id = uuid.uuid4().hex
                    report_data = {"best_model": best_model_name,
                                   "model_type": "classification" if is_classification else "regression",
                                   "results": results}
                    report_path = os.path.join(app.config["REPORT_FOLDER"], f"report_{report_id}.json")
                    with open(report_path, "w") as rh:
                        json.dump(report_data, rh, indent=4)
                    conn = get_db()
                    conn.execute(
                        """INSERT INTO runs (user_id, created, dataset_name, target_column,
                           task_type, best_model, best_score, n_models, report_id, meta_path, results_json)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                        (uid, datetime.datetime.now().isoformat(), file.filename, target_column,
                         "classification" if is_classification else "regression",
                         best_model_name, best_score_val, len(results),
                         report_id, meta_path, json.dumps(results))
                    )
                    conn.commit(); conn.close()

                # ── Signal done ─────────────────────────────────────────────
                result_payload = json.dumps({
                    "best_model": best_model_name,
                    "task_type":  "classification" if is_classification else "regression",
                    "meta_path":  meta_path,
                    "results":    results,
                })
                _sse_jobs[job_id]["status"]     = "done"
                _sse_jobs[job_id]["result_url"] = meta_path
                _sse_event(q, result_payload, event="done")

            except Exception as exc:
                logger.exception("SSE training job %s failed", job_id)
                _sse_jobs[job_id]["status"] = "error"
                _sse_jobs[job_id]["error"]  = str(exc)
                _sse_event(q, str(exc), event="error")
            finally:
                q.put(None)   # sentinel — tells generator to stop

        threading.Thread(target=_bg, daemon=True).start()
        return jsonify({"job_id": job_id, "stream_url": f"/train_progress/{job_id}"})

    except Exception as exc:
        logger.exception("train_stream setup failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/train_progress/<job_id>")
def train_progress(job_id):
    """SSE endpoint — streams training log events to the browser."""
    q = _sse_queues.get(job_id)
    if not q:
        return "Job not found", 404

    def _generate():
        while True:
            item = q.get()
            if item is None:
                break
            yield item

    return Response(
        stream_with_context(_generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":  "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        }
    )


@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]
        target_column   = request.form.get("target_column", "").strip()
        selected_models = request.form.getlist("selected_models")

        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        # FIX #10 — validate extension on backend
        if not allowed_file(file.filename):
            return jsonify({"error": "Only CSV files are accepted"}), 400
        if not target_column:
            return jsonify({"error": "Target column is required"}), 400
        if not selected_models:
            return jsonify({"error": "At least one model must be selected"}), 400

        # FIX #9 — unique filename prevents concurrent-upload collisions
        filename  = unique_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        df = pd.read_csv(file_path)
        # FIX #12
        df.columns = (
            df.columns
            .str.replace(r"[^a-zA-Z0-9_]", "", regex=True)
            .str.strip()
        )

        X, y, imputer, scaler, is_classification = clean_and_preprocess_data(df, target_column)
        logger.info("Feature dtypes after preprocessing:\n%s", X.dtypes)

        model_name_mapping = {
            "random_forest":       "Random Forest",
            "xgboost":             "XGBoost",
            "lightgbm":            "LightGBM",
            "logistic_regression": "Logistic Regression",
            "knn":                 "KNN",
            "svm":                 "SVM",
            "decision_tree":       "Decision Tree",
            "gradient_boosting":   "Gradient Boosting",
            "svr":                 "SVR",
        }
        selected_model_names = [
            model_name_mapping[m] for m in selected_models if m in model_name_mapping
        ]

        (results, best_model_name, best_model_obj,
         all_roc_auc_plots, all_confusion_matrix_plots,
         all_pr_curve_plots, regression_plots,
         feature_importance_plots, shap_plots) = train_and_compare_models(
            X, y, is_classification, selected_model_names
        )

        hyperparameters = suggest_hyperparameters(selected_model_names)

        # FIX #15 — use logger instead of print
        if not is_classification:
            logger.info("Regression plot keys: %s", list(regression_plots.keys()))
        else:
            logger.info("Confusion matrix plot keys: %s",
                        list(all_confusion_matrix_plots.keys()))

        # FIX #2 — save both the best model AND its scaler
        timestamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path   = os.path.join(
            app.config["MODEL_FOLDER"],
            f"{best_model_name}_{timestamp}.pkl",
        )
        scaler_path  = os.path.join(
            app.config["SCALER_FOLDER"],
            f"scaler_{best_model_name}_{timestamp}.pkl",
        )
        imputer_path = os.path.join(
            app.config["SCALER_FOLDER"],
            f"imputer_{best_model_name}_{timestamp}.pkl",
        )
        joblib.dump(best_model_obj, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(imputer, imputer_path)
        logger.info("Saved model → %s", model_path)
        logger.info("Saved scaler → %s", scaler_path)
        logger.info("Saved imputer → %s", imputer_path)

        # Save metadata so the prediction page can reconstruct the pipeline
        meta = {
            "feature_names":    list(X.columns),
            "is_classification": is_classification,
            "model_name":       best_model_name,
            "scaler_path":      scaler_path,
            "imputer_path":     imputer_path,
            "model_path":       model_path,
        }
        meta_path = model_path.replace(".pkl", "_meta.json")
        with open(meta_path, "w") as mf:
            json.dump(meta, mf)

        # FIX #4 — store report data server-side; pass only a short key to template
        report_id = uuid.uuid4().hex
        report_data = {
            "best_model":     best_model_name,
            "model_type":     "classification" if is_classification else "regression",
            "results":        results,
            "hyperparameters": hyperparameters,
        }
        report_path = os.path.join(
            app.config["REPORT_FOLDER"], f"report_{report_id}.json"
        )
        with open(report_path, "w") as fh:
            json.dump(report_data, fh, indent=4)

        plot_base64 = generate_performance_plot(results, is_classification) if results else None

        # Save run to history DB
        uid = current_user_id()
        if uid:
            best_score_val = (
                results[best_model_name]["accuracy"]
                if is_classification
                else results[best_model_name]["r2"]
            )
            conn = get_db()
            conn.execute(
                """INSERT INTO runs
                       (user_id, created, dataset_name, target_column, task_type,
                        best_model, best_score, n_models, report_id, meta_path, results_json)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (uid, datetime.datetime.now().isoformat(),
                 file.filename, target_column,
                 "classification" if is_classification else "regression",
                 best_model_name, best_score_val, len(selected_model_names),
                 report_id, meta_path, json.dumps(results))
            )
            conn.commit()
            conn.close()

        return render_template(
            "result.html",
            results=results,
            best_model=best_model_name,
            model_path=model_path,
            scaler_path=scaler_path,
            model_type="classification" if is_classification else "regression",
            hyperparameters=hyperparameters,
            selected_models=selected_model_names,
            roc_auc_plots=all_roc_auc_plots if is_classification else None,
            confusion_matrix_plots=all_confusion_matrix_plots if is_classification else None,
            pr_curve_plots=all_pr_curve_plots if is_classification else None,
            regression_plots=regression_plots if not is_classification else None,
            plot_base64=plot_base64,
            report_id=report_id,
            feature_importance_plots=feature_importance_plots,
            shap_plots=shap_plots,
            meta_path=meta_path,
            logged_in=(uid is not None),
        )

    except Exception as exc:
        logger.exception("Exception in /upload route")
        return jsonify({"error": f"Error processing dataset: {exc}"}), 500



# ---------------------------------------------------------------------------
# Module-level ID/timestamp column detector (also used inside preprocessing)
# Exposed at module level so the test suite can import and unit-test it.
# ---------------------------------------------------------------------------
def _is_id_or_ts_col(col: str) -> bool:
    """Return True if col name looks like a pure-ID or timestamp column."""
    _id_words = {"id", "ids", "timestamp", "timestamps"}
    if any(p in _id_words for p in col.lower().split("_")):
        return True
    camel = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", col)
    camel = re.sub(r"([a-z])([A-Z])", r"\1_\2", camel)
    return any(p in _id_words for p in camel.lower().split("_"))

# ---------------------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------------------

def clean_and_preprocess_data(df: pd.DataFrame, target_column: str):
    """
    Returns (X, y, scaler, is_classification).
    FIX #5  — uses detect_classification() consistently.
    FIX #11 — no local re-import of 're'; already imported at top level.
    FIX #12 — .str.strip() with no argument.
    """
    df = df.copy()
    # FIX #12
    df.columns = (
        df.columns
        .str.replace(r"[^a-zA-Z0-9_]", "", regex=True)
        .str.strip()
    )
    df.drop_duplicates(inplace=True)
    df = df.dropna(subset=[target_column])

    # Drop time-like or fully non-numeric feature columns
    # FIX #11 — re is already imported at module level
    _time_re = re.compile(r"^\d{1,2}:\d{2}\s?(AM|PM|am|pm)?$")

    def is_time_string(s: str) -> bool:
        return bool(_time_re.match(str(s).strip()))

    cols_to_drop = []
    for col in df.columns:
        if col == target_column:
            continue

        # Object columns are handled later by get_dummies — never drop them here.
        # Only check non-object columns that might be un-coercible (e.g. mixed types).
        if df[col].dtype == object:
            # Drop if it looks like a time string (useless feature)
            sample = df[col].dropna().astype(str).head(10).tolist()
            if any(is_time_string(v) for v in sample):
                cols_to_drop.append(col)
                logger.info("Dropping '%s': contains time-like strings.", col)
            # else: keep — get_dummies will encode it
        else:
            # Numeric-ish dtype — try to coerce; drop only if it truly fails
            try:
                pd.to_numeric(df[col], errors="raise")
            except Exception:
                cols_to_drop.append(col)
                logger.info("Dropping '%s': non-numeric, non-object data.", col)

    df.drop(columns=cols_to_drop, errors="ignore", inplace=True)

    # Drop pure-ID / timestamp columns (uses module-level _is_id_or_ts_col)
    id_ts_cols = [
        c for c in df.columns
        if c != target_column and _is_id_or_ts_col(c)
    ]
    if id_ts_cols:
        logger.info("Dropping id/timestamp columns: %s", id_ts_cols)
    df.drop(columns=id_ts_cols, errors="ignore", inplace=True)

    # Drop high-cardinality object columns — too many unique values to encode
    # meaningfully (e.g. Name, Ticket). Threshold: >50% of rows are unique.
    high_card_cols = []
    n_rows = len(df)
    for col in df.columns:
        if col == target_column:
            continue
        if df[col].dtype == object and df[col].nunique() > max(20, n_rows * 0.5):
            high_card_cols.append(col)
            logger.info("Dropping '%s': high cardinality object column (%d unique).",
                        col, df[col].nunique())
    df.drop(columns=high_card_cols, errors="ignore", inplace=True)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found after dropping columns."
        )

    X = df.drop(columns=[target_column])
    if X.empty:
        raise ValueError(
            "Feature dataframe is empty after dropping the target and other columns."
        )

    y = df[target_column]

    # FIX #5 — single helper for task-type detection
    is_classification = detect_classification(y)

    if is_classification:
        if y.dtype == "object":
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), index=y.index)
        else:
            y = pd.to_numeric(y, errors="coerce")

    categorical_cols = X.select_dtypes(include=["object"]).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    X = X.astype("float64")

    # Impute missing values with column median BEFORE scaling
    # (fixes NaN crash with SMOTE and all sklearn estimators)
    imputer  = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    X        = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

    # Confirm no NaNs remain
    if X.isnull().any().any():
        raise ValueError("NaNs remain after imputation — check feature columns.")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X        = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    return X, y, imputer, scaler, is_classification




# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def preprocess_for_clustering(df: pd.DataFrame) -> tuple:
    """Preprocess dataframe for clustering — no target column needed."""
    df = df.copy()
    df.columns = (df.columns.str.replace(r"[^a-zA-Z0-9_]", "", regex=True).str.strip())
    df.drop_duplicates(inplace=True)

    # Drop id/timestamp cols
    def _is_id_or_ts_col(col):
        _id_words = {"id", "ids", "timestamp", "timestamps"}
        if any(p in _id_words for p in col.lower().split("_")):
            return True
        camel = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", col)
        camel = re.sub(r"([a-z])([A-Z])", r"\1_\2", camel)
        return any(p in _id_words for p in camel.lower().split("_"))

    drop_cols = [c for c in df.columns if _is_id_or_ts_col(c)]
    df.drop(columns=drop_cols, errors="ignore", inplace=True)

    # Encode categoricals, keep numerics
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    # Drop high-cardinality text columns
    n_rows = len(df)
    cat_cols = [c for c in cat_cols if df[c].nunique() <= max(20, n_rows * 0.5)]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df = df.select_dtypes(include="number").astype("float64")
    df.dropna(axis=1, how="all", inplace=True)

    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    X = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X, list(df.columns)


def run_clustering(X: pd.DataFrame, feature_names: list,
                   n_clusters_range=(2, 8)) -> dict:
    """Run KMeans, DBSCAN, Agglomerative clustering and return results + plots."""
    results   = {}
    plots     = {}
    X_arr     = X.values

    # ── PCA to 2D for all scatter plots ────────────────────────────────────
    pca       = PCA(n_components=2, random_state=42)
    X_2d      = pca.fit_transform(X_arr)
    var_exp   = pca.explained_variance_ratio_

    # ── Elbow + Silhouette curve for KMeans ────────────────────────────────
    k_range   = list(range(n_clusters_range[0], n_clusters_range[1] + 1))
    inertias  = []
    sil_scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_arr)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_arr, labels) if len(set(labels)) > 1 else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(k_range, inertias, "bo-", linewidth=2)
    ax1.set_xlabel("Number of Clusters (k)"); ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Curve — KMeans")
    ax2.plot(k_range, sil_scores, "rs-", linewidth=2)
    ax2.set_xlabel("Number of Clusters (k)"); ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score vs k")
    plt.tight_layout()
    plots["elbow"] = _encode_plot()

    # ── Best k by silhouette ───────────────────────────────────────────────
    best_k = k_range[int(np.argmax(sil_scores))]

    # ── KMeans with best k ─────────────────────────────────────────────────
    km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    km_labels = km_best.fit_predict(X_arr)
    km_sil    = round(silhouette_score(X_arr, km_labels), 4)
    km_db     = round(davies_bouldin_score(X_arr, km_labels), 4)
    results["KMeans"] = {
        "n_clusters":       best_k,
        "silhouette":       km_sil,
        "davies_bouldin":   km_db,
        "params":           {"n_clusters": best_k, "n_init": 10},
        "cluster_sizes":    {int(k): int(v) for k, v in zip(*np.unique(km_labels, return_counts=True))},
    }
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=km_labels, cmap="tab10", alpha=0.7, s=30)
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel(f"PC1 ({var_exp[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({var_exp[1]*100:.1f}% var)")
    plt.title(f"KMeans Clusters (k={best_k}) — PCA 2D Projection")
    plt.tight_layout()
    plots["kmeans_scatter"] = _encode_plot()

    # ── DBSCAN — auto-tune eps via nearest-neighbour distances ─────────────
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=5).fit(X_arr)
    dists, _ = nbrs.kneighbors(X_arr)
    eps_auto = float(np.percentile(dists[:, -1], 90))

    db = DBSCAN(eps=eps_auto, min_samples=5)
    db_labels = db.fit_predict(X_arr)
    n_db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise       = int(np.sum(db_labels == -1))

    if n_db_clusters >= 2:
        db_sil = round(silhouette_score(X_arr[db_labels != -1], db_labels[db_labels != -1]), 4)
        db_db  = round(davies_bouldin_score(X_arr[db_labels != -1], db_labels[db_labels != -1]), 4)
    else:
        db_sil = db_db = None

    results["DBSCAN"] = {
        "n_clusters":     n_db_clusters,
        "n_noise_points": n_noise,
        "silhouette":     db_sil,
        "davies_bouldin": db_db,
        "params":         {"eps": round(eps_auto, 4), "min_samples": 5},
        "cluster_sizes":  {int(k): int(v) for k, v in zip(*np.unique(db_labels, return_counts=True))},
    }
    plt.figure(figsize=(8, 5))
    palette = plt.cm.tab10(np.linspace(0, 1, max(n_db_clusters + 1, 2)))
    for lbl in np.unique(db_labels):
        mask = db_labels == lbl
        color = "grey" if lbl == -1 else palette[lbl % len(palette)]
        label = "Noise" if lbl == -1 else f"Cluster {lbl}"
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[color], label=label, alpha=0.7, s=30)
    plt.legend(fontsize=8, loc="best")
    plt.xlabel(f"PC1 ({var_exp[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({var_exp[1]*100:.1f}% var)")
    plt.title(f"DBSCAN Clusters (eps={eps_auto:.3f}) — PCA 2D Projection")
    plt.tight_layout()
    plots["dbscan_scatter"] = _encode_plot()

    # ── Agglomerative with best_k ──────────────────────────────────────────
    agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    agg_labels = agg.fit_predict(X_arr)
    agg_sil    = round(silhouette_score(X_arr, agg_labels), 4)
    agg_db     = round(davies_bouldin_score(X_arr, agg_labels), 4)
    results["Agglomerative"] = {
        "n_clusters":     best_k,
        "silhouette":     agg_sil,
        "davies_bouldin": agg_db,
        "params":         {"n_clusters": best_k, "linkage": "ward"},
        "cluster_sizes":  {int(k): int(v) for k, v in zip(*np.unique(agg_labels, return_counts=True))},
    }
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=agg_labels, cmap="tab10", alpha=0.7, s=30)
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel(f"PC1 ({var_exp[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({var_exp[1]*100:.1f}% var)")
    plt.title(f"Agglomerative Clusters (k={best_k}) — PCA 2D Projection")
    plt.tight_layout()
    plots["agg_scatter"] = _encode_plot()

    # Pick best algorithm by silhouette (ignoring None)
    best_algo = max(
        ((name, r["silhouette"]) for name, r in results.items() if r["silhouette"] is not None),
        key=lambda t: t[1], default=("KMeans", 0)
    )[0]

    return results, plots, best_algo, round(float(var_exp[0] + var_exp[1]) * 100, 1)


@app.route("/cluster", methods=["POST"])
def cluster():
    """Handle clustering task — no target column required."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files["file"]
        if not allowed_file(file.filename):
            return jsonify({"error": "Only CSV files are accepted"}), 400

        filename  = unique_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        df = pd.read_csv(file_path)
        if df.shape[0] < 20:
            return jsonify({"error": "Dataset too small for clustering (need ≥ 20 rows)."}), 400

        X, feature_names = preprocess_for_clustering(df)
        if X.shape[1] < 2:
            return jsonify({"error": "Need at least 2 numeric features for clustering."}), 400

        cluster_results, cluster_plots, best_algo, pca_variance = run_clustering(
            X, feature_names,
            n_clusters_range=(2, min(8, X.shape[0] // 5))
        )

        # Save run to DB if logged in
        uid = current_user_id()
        if uid:
            conn = get_db()
            conn.execute(
                """INSERT INTO runs
                   (user_id, created, dataset_name, target_column, task_type,
                    best_model, best_score, n_models, report_id, meta_path, results_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (uid, datetime.datetime.now().isoformat(),
                 file.filename, "N/A", "clustering",
                 best_algo,
                 cluster_results[best_algo]["silhouette"],
                 len(cluster_results), "", "", json.dumps({
                     k: {kk: vv for kk, vv in v.items() if kk != "cluster_sizes"}
                     for k, v in cluster_results.items()
                 }))
            )
            conn.commit()
            conn.close()

        return render_template(
            "cluster.html",
            results=cluster_results,
            plots=cluster_plots,
            best_algo=best_algo,
            pca_variance=pca_variance,
            n_features=X.shape[1],
            n_samples=X.shape[0],
            dataset_name=file.filename,
            logged_in=(uid is not None),
        )

    except Exception as exc:
        logger.exception("cluster route failed")
        return jsonify({"error": f"Clustering failed: {exc}"}), 500


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_and_compare_models(X, y, is_classification: bool, selected_models=None):
    """
    FIX #1  — regression_plots initialised exactly once, before the loop.
    FIX #3  — SMOTE k_neighbors clamped to at least 1, guarded so SMOTE is
               only called when the minority class has ≥ 2 samples.
    FIX #6  — stratify=y added to train_test_split for classification.
    FIX #7  — removed unused max_neighbors variable.
    FIX #8  — removed redundant always-true ternary in binary PR curve.
    FIX #11 — removed duplicate local imports (plt, BytesIO, base64, Counter).
    FIX #15 — replaced print() with logger.
    """
    # FIX #6 — stratify keeps class proportions in the test set
    stratify = y if is_classification else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    # FIX #3 — only apply SMOTE when minority class has ≥ 2 samples
    if is_classification and len(np.unique(y_train)) > 1:
        class_counts   = Counter(y_train)
        min_class_count = min(class_counts.values())
        if min_class_count >= 2:
            # k_neighbors must be < min_class_count and ≥ 1
            n_neighbors = max(1, min(5, min_class_count - 1))
            smote = SMOTE(random_state=42, k_neighbors=n_neighbors)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logger.info("SMOTE applied with k_neighbors=%d", n_neighbors)
        else:
            logger.warning(
                "SMOTE skipped: minority class has only %d sample(s).", min_class_count
            )

    scoring = "f1_weighted" if is_classification else "neg_mean_squared_error"

    if is_classification:
        param_grids = {
            "Random Forest":      {"model": RandomForestClassifier(random_state=42),
                                   "params": {"n_estimators": [100, 200], "max_depth": [None, 10]}},
            "XGBoost":            {"model": XGBClassifier(eval_metric="mlogloss", random_state=42),
                                   "params": {"n_estimators": [100], "learning_rate": [0.01, 0.1]}},
            "LightGBM":           {"model": LGBMClassifier(random_state=42),
                                   "params": {"n_estimators": [100], "learning_rate": [0.01, 0.1]}},
            "Logistic Regression":{"model": LogisticRegression(max_iter=1000, random_state=42),
                                   "params": {"C": [0.1, 1.0], "solver": ["liblinear"]}},
            "KNN":                {"model": KNeighborsClassifier(),
                                   "params": {"n_neighbors": [3, 5], "weights": ["uniform", "distance"]}},
            "SVM":                {"model": SVC(probability=True, random_state=42),
                                   "params": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}},
            "Decision Tree":      {"model": DecisionTreeClassifier(random_state=42),
                                   "params": {"max_depth": [None, 10, 20], "min_samples_split": [2, 5]}},
            "Gradient Boosting":  {"model": GradientBoostingClassifier(random_state=42),
                                   "params": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]}},
        }
    else:
        param_grids = {
            "Random Forest":     {"model": RandomForestRegressor(random_state=42),
                                  "params": {"n_estimators": [100, 200], "max_depth": [None, 10]}},
            "XGBoost":           {"model": XGBRegressor(eval_metric="rmse", random_state=42),
                                  "params": {"n_estimators": [100], "learning_rate": [0.01, 0.1]}},
            "LightGBM":          {"model": LGBMRegressor(random_state=42),
                                  "params": {"n_estimators": [100], "learning_rate": [0.01, 0.1]}},
            "KNN":               {"model": KNeighborsRegressor(),
                                  "params": {"n_neighbors": [3, 5], "weights": ["uniform", "distance"]}},
            "SVR":               {"model": SVR(),
                                  "params": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}},
            "Decision Tree":     {"model": DecisionTreeRegressor(random_state=42),
                                  "params": {"max_depth": [None, 10, 20], "min_samples_split": [2, 5]}},
            "Gradient Boosting": {"model": GradientBoostingRegressor(random_state=42),
                                  "params": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]}},
        }

    if selected_models:
        param_grids = {k: v for k, v in param_grids.items() if k in selected_models}

    best_score               = -np.inf
    best_model_obj           = None
    best_model_name          = ""
    all_roc_auc_plots        = {}
    all_confusion_matrix_plots = {}
    all_pr_curve_plots       = {}
    results                  = {}
    regression_plots         = {}   # single init (FIX #1)
    feature_importance_plots = {}   # NEW — per-model importance charts

    for name, model_info in param_grids.items():
        model      = model_info["model"]
        param_grid = dict(model_info["params"])     # shallow copy so we can mutate safely

        # Clamp KNN n_neighbors to training set size
        # FIX #7 — removed unused max_neighbors variable
        if name in ("KNN",):
            max_allowed = len(X_train)
            filtered    = [n for n in param_grid["n_neighbors"] if n <= max_allowed]
            if not filtered:
                filtered = [max(1, max_allowed)]
            param_grid["n_neighbors"] = filtered

        search = RandomizedSearchCV(
            model, param_grid, n_iter=5, cv=3,
            scoring=scoring, n_jobs=-1, random_state=42
        )
        t_start     = time.perf_counter()
        search.fit(X_train, y_train)
        train_secs  = round(time.perf_counter() - t_start, 2)
        final_model = search.best_estimator_
        y_pred      = final_model.predict(X_test)

        # ── Cross-validation on the full dataset (5-fold) ─────────────────
        cv_scoring = "f1_weighted" if is_classification else "r2"
        cv_splitter = (StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                       if is_classification else KFold(n_splits=5, shuffle=True, random_state=42))
        try:
            cv_scores = cross_val_score(
                final_model, X, y,
                cv=cv_splitter, scoring=cv_scoring, n_jobs=-1
            )
            cv_mean = round(float(cv_scores.mean()), 4)
            cv_std  = round(float(cv_scores.std()), 4)
        except Exception as cv_err:
            logger.warning("CV failed for %s: %s", name, cv_err)
            cv_mean, cv_std = None, None

        if is_classification:
            accuracy = accuracy_score(y_test, y_pred)
            f1       = f1_score(y_test, y_pred, average="weighted")
            results[name] = {
                "accuracy":    round(accuracy * 100, 2),
                "f1_score":    round(f1 * 100, 2),
                "best_params": search.best_params_,
                "train_time":  train_secs,
                "cv_mean":     cv_mean,
                "cv_std":      cv_std,
            }
            score = f1

            n_classes = len(np.unique(y_test))

            # ROC curve
            if n_classes == 2:
                y_proba = final_model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_val = auc(fpr, tpr)
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color="darkorange", lw=2,
                         label=f"ROC curve (AUC = {roc_val:.2f})")
                plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
                plt.xlim([0, 1]); plt.ylim([0, 1.05])
                plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
                plt.title(f"ROC Curve — {name}"); plt.legend(loc="lower right")
                all_roc_auc_plots[name] = _encode_plot()

            elif n_classes > 2:
                y_proba = final_model.predict_proba(X_test)
                fpr_d, tpr_d, roc_d = {}, {}, {}
                plt.figure(figsize=(8, 6))
                for i in range(n_classes):
                    fpr_d[i], tpr_d[i], _ = roc_curve(y_test == i, y_proba[:, i])
                    roc_d[i] = auc(fpr_d[i], tpr_d[i])
                    plt.plot(fpr_d[i], tpr_d[i], label=f"Class {i} (AUC={roc_d[i]:.2f})")
                plt.plot([0, 1], [0, 1], "k--")
                plt.xlim([0, 1]); plt.ylim([0, 1.05])
                plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
                plt.title(f"ROC Curve — {name}"); plt.legend(loc="lower right")
                all_roc_auc_plots[name] = _encode_plot()

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=np.unique(y), yticklabels=np.unique(y))
            plt.xlabel("Predicted Label"); plt.ylabel("True Label")
            plt.title(f"Confusion Matrix — {name}")
            all_confusion_matrix_plots[name] = _encode_plot()

            # Precision-Recall curve
            if n_classes == 2:
                # FIX #8 — removed always-true ternary; y_proba is already 1-D here
                y_proba_bin = final_model.predict_proba(X_test)[:, 1]
                precision_arr, recall_arr, _ = precision_recall_curve(y_test, y_proba_bin)
                avg_prec = average_precision_score(y_test, y_proba_bin)
                plt.figure(figsize=(8, 6))
                plt.plot(recall_arr, precision_arr, label=f"AP = {avg_prec:.2f}")
                plt.xlabel("Recall"); plt.ylabel("Precision")
                plt.title(f"Precision-Recall Curve — {name}"); plt.legend(loc="lower left")
                all_pr_curve_plots[name] = _encode_plot()

            elif n_classes > 2:
                y_proba_mc = final_model.predict_proba(X_test)
                plt.figure(figsize=(8, 6))
                for i in range(n_classes):
                    prec_i, rec_i, _ = precision_recall_curve(y_test == i, y_proba_mc[:, i])
                    ap_i = average_precision_score(y_test == i, y_proba_mc[:, i])
                    plt.plot(rec_i, prec_i, lw=2, label=f"Class {i} (AP={ap_i:.2f})")
                plt.xlabel("Recall"); plt.ylabel("Precision")
                plt.title(f"Precision-Recall Curve — {name}")
                plt.xlim([0, 1]); plt.ylim([0, 1.05])
                plt.legend(loc="lower left"); plt.grid(True)
                all_pr_curve_plots[name] = _encode_plot()

        else:
            mse  = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2   = r2_score(y_test, y_pred)
            results[name] = {
                "rmse":        round(rmse, 4),
                "r2":          round(r2, 4),
                "best_params": search.best_params_,
                "train_time":  train_secs,
                "cv_mean":     cv_mean,
                "cv_std":      cv_std,
            }
            # Use R² as the selection criterion (higher = better, consistent with classification)
            score = r2

            residuals = y_test - y_pred

            plt.figure(figsize=(8, 6))
            plt.scatter(y_pred, residuals, alpha=0.6)
            plt.axhline(y=0, color="r", linestyle="--")
            plt.xlabel("Predicted Values"); plt.ylabel("Residuals")
            plt.title(f"Residual Plot — {name}")
            residual_plot = _encode_plot()

            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred, alpha=0.6)
            plt.xlabel("Actual Values"); plt.ylabel("Predicted Values")
            plt.title(f"Predicted vs Actual — {name}")
            pred_actual_plot = _encode_plot()

            plt.figure(figsize=(8, 6))
            sns.histplot(residuals, kde=True)
            plt.xlabel("Residuals"); plt.title(f"Residuals Distribution — {name}")
            residual_dist_plot = _encode_plot()

            # FIX #1 — regression_plots already initialised before loop; just assign
            regression_plots[name] = {
                "Residual Plot":         residual_plot,
                "Predicted vs Actual":   pred_actual_plot,
                "Residuals Distribution":residual_dist_plot,
            }

        # ── Feature importance (tree-based and linear models) ──────────────
        importances = None
        feat_names  = list(X.columns)
        if hasattr(final_model, "feature_importances_"):
            importances = final_model.feature_importances_
        elif hasattr(final_model, "coef_"):
            coef = final_model.coef_
            importances = np.abs(coef[0] if coef.ndim > 1 else coef)

        if importances is not None and len(feat_names) == len(importances):
            top_n = min(15, len(feat_names))
            idx   = np.argsort(importances)[-top_n:][::-1]
            top_feats = [feat_names[i] for i in idx]
            top_vals  = [importances[i] for i in idx]

            plt.figure(figsize=(8, max(3, top_n * 0.38)))
            colors = ["#2196F3" if i == 0 else "#90CAF9" for i in range(top_n)]
            plt.barh(top_feats[::-1], top_vals[::-1], color=colors[::-1])
            plt.xlabel("Importance Score")
            plt.title(f"Feature Importance — {name} (top {top_n})")
            plt.tight_layout()
            feature_importance_plots[name] = _encode_plot()

        if score > best_score:
            best_score      = score
            best_model_obj  = final_model
            best_model_name = name

    if not results:
        return {}, "No models selected", None, {}, {}, {}, {}, {}, {}

    # ── SHAP explainability for the best model ─────────────────────────────
    shap_plots = {}
    try:
        sample_X = X_test if len(X_test) <= 200 else X_test.sample(200, random_state=42)

        # Choose explainer based on model type
        if hasattr(best_model_obj, "feature_importances_"):
            explainer = shap.TreeExplainer(best_model_obj)
        else:
            explainer = shap.KernelExplainer(
                best_model_obj.predict_proba if is_classification and hasattr(best_model_obj, "predict_proba")
                else best_model_obj.predict,
                shap.sample(sample_X, 50)
            )

        shap_values = explainer(sample_X)

        # Handle multi-output (multi-class): use class-1 or take mean abs
        sv = shap_values
        if hasattr(sv, "values"):
            vals = sv.values
            if vals.ndim == 3:          # (samples, features, classes)
                vals = vals[:, :, 1] if vals.shape[2] == 2 else np.abs(vals).mean(axis=2)
                sv_plot = shap.Explanation(values=vals, base_values=sv.base_values.mean() if hasattr(sv.base_values, "mean") else sv.base_values,
                                           data=sv.data, feature_names=sv.feature_names)
            else:
                sv_plot = sv
        else:
            sv_plot = sv

        # Summary bar plot (mean |SHAP|)
        plt.figure(figsize=(8, max(4, min(len(X.columns), 15) * 0.4)))
        shap.plots.bar(sv_plot, max_display=15, show=False)
        plt.title(f"SHAP Feature Importance — {best_model_name}")
        plt.tight_layout()
        shap_plots["bar"] = _encode_plot()

        # Beeswarm plot (distribution of SHAP values)
        plt.figure(figsize=(8, max(4, min(len(X.columns), 15) * 0.4)))
        shap.plots.beeswarm(sv_plot, max_display=15, show=False)
        plt.title(f"SHAP Beeswarm — {best_model_name}")
        plt.tight_layout()
        shap_plots["beeswarm"] = _encode_plot()

        logger.info("SHAP plots generated for %s", best_model_name)

    except Exception as shap_err:
        logger.warning("SHAP failed for %s: %s", best_model_name, shap_err)

    return (
        results,
        best_model_name,
        best_model_obj,
        all_roc_auc_plots,
        all_confusion_matrix_plots,
        all_pr_curve_plots,
        regression_plots,
        feature_importance_plots,
        shap_plots,
    )


# ---------------------------------------------------------------------------
# Hyperparameter display
# ---------------------------------------------------------------------------

def suggest_hyperparameters(selected_models=None):
    all_hyperparameters = {
        "Random Forest":      {"n_estimators": [100, 200], "max_depth": [None, 10]},
        "XGBoost":            {"n_estimators": [100], "learning_rate": [0.01, 0.1]},
        "LightGBM":           {"n_estimators": [100], "learning_rate": [0.01, 0.1]},
        "Logistic Regression":{"C": [0.1, 1.0], "solver": ["liblinear"]},
        "KNN":                {"n_neighbors": [3, 5], "weights": ["uniform", "distance"]},
        "SVM":                {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
        "Decision Tree":      {"max_depth": [None, 10, 20], "min_samples_split": [2, 5]},
        "Gradient Boosting":  {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]},
    }
    if selected_models:
        return {k: v for k, v in all_hyperparameters.items() if k in selected_models}
    return all_hyperparameters


# ---------------------------------------------------------------------------
# Download report
# FIX #4 — data stored server-side; only a short report_id travels over HTTP
# ---------------------------------------------------------------------------

@app.route("/download_report/<report_id>")
def download_report(report_id: str):
    try:
        # Sanitise the ID so it can only be hex chars (UUID without dashes)
        if not re.fullmatch(r"[0-9a-f]{32}", report_id):
            return "Invalid report ID", 400

        report_path = os.path.join(
            app.config["REPORT_FOLDER"], f"report_{report_id}.json"
        )
        if not os.path.exists(report_path):
            return "Report not found", 404

        return send_file(
            report_path,
            as_attachment=True,
            download_name="model_training_report.json",
        )

    except Exception as exc:
        logger.exception("download_report failed")
        return "Error generating report", 500


# ---------------------------------------------------------------------------
# Model download route (bonus — was missing entirely)
# ---------------------------------------------------------------------------

@app.route("/download_model")
def download_model():
    """Allow the user to download the saved best-model .pkl."""
    try:
        model_path = request.args.get("path", "")
        # Security: only serve files from the models folder
        models_dir = os.path.abspath(app.config["MODEL_FOLDER"])
        requested  = os.path.abspath(model_path)
        if not requested.startswith(models_dir) or not os.path.exists(requested):
            return "Model file not found", 404
        return send_file(requested, as_attachment=True)
    except Exception as exc:
        logger.exception("download_model failed")
        return "Error downloading model", 500





# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user_id():
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        confirm  = request.form.get("confirm", "").strip()
        if not username or not password:
            error = "Username and password are required."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        elif password != confirm:
            error = "Passwords do not match."
        else:
            try:
                conn = get_db()
                conn.execute(
                    "INSERT INTO users (username, password, created) VALUES (?,?,?)",
                    (username, hash_password(password), datetime.datetime.now().isoformat())
                )
                conn.commit()
                conn.close()
                flash("Account created! Please log in.", "success")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                error = "Username already taken."
    return render_template("register.html", error=error)


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user_id():
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username, hash_password(password))
        ).fetchone()
        conn.close()
        if user:
            session["user_id"]  = user["id"]
            session["username"] = user["username"]
            flash(f"Welcome back, {username}!", "success")
            return redirect(url_for("index"))
        error = "Invalid username or password."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


@app.route("/history")
@login_required
def history():
    conn = get_db()
    runs = conn.execute(
        "SELECT * FROM runs WHERE user_id=? ORDER BY created DESC",
        (current_user_id(),)
    ).fetchall()
    conn.close()
    return render_template("history.html",
                           runs=runs,
                           username=session.get("username", ""))


# ---------------------------------------------------------------------------
# Prediction page
# ---------------------------------------------------------------------------

@app.route("/predict_page")
def predict_page():
    """Serve the prediction form, loading feature names from saved metadata."""
    try:
        meta_path = request.args.get("meta", "")
        # Security: only allow files inside MODEL_FOLDER
        models_dir = os.path.abspath(app.config["MODEL_FOLDER"])
        abs_meta   = os.path.abspath(meta_path)
        if not abs_meta.startswith(models_dir) or not os.path.exists(abs_meta):
            return "Model metadata not found. Please train a model first.", 404

        with open(abs_meta) as f:
            meta = json.load(f)

        return render_template("predict.html", meta=meta, meta_path=meta_path)

    except Exception as exc:
        logger.exception("predict_page failed")
        return f"Error loading prediction page: {exc}", 500


@app.route("/predict", methods=["POST"])
def predict():
    """Accept form values, scale them, run the saved model, return a prediction."""
    try:
        meta_path = request.form.get("meta_path", "")
        models_dir = os.path.abspath(app.config["MODEL_FOLDER"])
        abs_meta   = os.path.abspath(meta_path)
        if not abs_meta.startswith(models_dir) or not os.path.exists(abs_meta):
            return jsonify({"error": "Model metadata not found"}), 404

        with open(abs_meta) as f:
            meta = json.load(f)

        # Build input row from posted form values
        feature_names = meta["feature_names"]
        row = {}
        missing_cols = []
        for feat in feature_names:
            val = request.form.get(feat, "").strip()
            if val == "":
                missing_cols.append(feat)
                row[feat] = 0.0          # default to 0 for missing
            else:
                try:
                    row[feat] = float(val)
                except ValueError:
                    return jsonify({"error": f"'{feat}' must be a number. Got: '{val}'"}), 400

        input_df = pd.DataFrame([row], columns=feature_names)

        # Load imputer + scaler + model
        scaler_path  = meta["scaler_path"]
        imputer_path = meta.get("imputer_path", "")
        model_path   = meta["model_path"]
        if not os.path.exists(scaler_path) or not os.path.exists(model_path):
            return jsonify({"error": "Saved model files not found on disk."}), 404

        scaler  = joblib.load(scaler_path)
        model   = joblib.load(model_path)
        input_arr = input_df.astype("float64").values

        # Apply imputer if available (handles any NaNs from blank fields)
        if imputer_path and os.path.exists(imputer_path):
            imputer   = joblib.load(imputer_path)
            input_arr = imputer.transform(input_arr)

        input_scaled = scaler.transform(input_arr)
        prediction   = model.predict(input_scaled)[0]

        # Confidence / probability for classifiers
        confidence = None
        if meta["is_classification"] and hasattr(model, "predict_proba"):
            proba      = model.predict_proba(input_scaled)[0]
            confidence = round(float(max(proba)) * 100, 2)

        return jsonify({
            "prediction":  str(prediction),
            "confidence":  confidence,
            "is_classification": meta["is_classification"],
            "model_name":  meta["model_name"],
            "missing_cols": missing_cols,
        })

    except Exception as exc:
        logger.exception("predict failed")
        return jsonify({"error": str(exc)}), 500




# ===========================================================================
# REST API — v1
# Simple hand-rolled JSON API with Swagger UI served at /api/docs
# ===========================================================================

# ── In-memory job store (replace with Redis/DB in production) ──────────────
_api_jobs: dict = {}   # job_id → {status, result, error, created}

API_MODELS = {
    "classification": ["Random Forest","XGBoost","LightGBM","Logistic Regression",
                        "KNN","SVM","Decision Tree","Gradient Boosting"],
    "regression":     ["Random Forest","XGBoost","LightGBM","KNN","SVR",
                        "Decision Tree","Gradient Boosting"],
    "clustering":     ["KMeans","DBSCAN","Agglomerative"],
}

# ── OpenAPI / Swagger spec ─────────────────────────────────────────────────
SWAGGER_SPEC = {
    "openapi": "3.0.3",
    "info": {
        "title":       "AutoML REST API",
        "version":     "1.0.0",
        "description": "Programmatic access to the AutoML Platform — train models, run predictions, inspect results.",
    },
    "servers": [{"url": "/api/v1", "description": "v1"}],
    "paths": {
        "/models": {
            "get": {
                "summary":     "List available models",
                "operationId": "listModels",
                "parameters": [{
                    "name": "task", "in": "query", "required": False,
                    "schema": {"type": "string", "enum": ["classification","regression","clustering"]},
                    "description": "Filter by task type",
                }],
                "responses": {"200": {"description": "Model list"}},
            }
        },
        "/predict": {
            "post": {
                "summary":     "Run prediction using a saved model",
                "operationId": "predict",
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {
                        "type": "object",
                        "required": ["meta_path", "features"],
                        "properties": {
                            "meta_path": {"type": "string", "description": "Path to the _meta.json saved after training"},
                            "features":  {"type": "object", "description": "Dict of feature_name → numeric value"},
                        }
                    }}}
                },
                "responses": {
                    "200": {"description": "Prediction result"},
                    "400": {"description": "Bad request"},
                    "404": {"description": "Model not found"},
                },
            }
        },
        "/train": {
            "post": {
                "summary":     "Start an async training job",
                "operationId": "train",
                "requestBody": {
                    "required": True,
                    "content": {"multipart/form-data": {"schema": {
                        "type": "object",
                        "required": ["file", "target_column"],
                        "properties": {
                            "file":           {"type": "string", "format": "binary"},
                            "target_column":  {"type": "string"},
                            "selected_models":{"type": "string", "description": "Comma-separated model names"},
                        }
                    }}}
                },
                "responses": {
                    "202": {"description": "Job accepted — poll /jobs/{job_id}"},
                    "400": {"description": "Bad request"},
                },
            }
        },
        "/jobs/{job_id}": {
            "get": {
                "summary":     "Poll a training job",
                "operationId": "getJob",
                "parameters": [{"name":"job_id","in":"path","required":True,"schema":{"type":"string"}}],
                "responses": {
                    "200": {"description": "Job status + results when done"},
                    "404": {"description": "Job not found"},
                },
            }
        },
        "/runs": {
            "get": {
                "summary":     "List saved runs (auth required)",
                "operationId": "listRuns",
                "responses": {
                    "200": {"description": "List of run records"},
                    "401": {"description": "Not authenticated"},
                },
            }
        },
    }
}


@app.route("/api/docs")
def api_docs():
    """Serve the Swagger UI."""
    return render_template("swagger.html")


@app.route("/api/v1/openapi.json")
def api_spec():
    """Serve the OpenAPI spec."""
    return jsonify(SWAGGER_SPEC)


@app.route("/api/v1/models", methods=["GET"])
def api_models():
    task = request.args.get("task", "").lower()
    if task and task in API_MODELS:
        return jsonify({"task": task, "models": API_MODELS[task]})
    return jsonify({"models": API_MODELS})


@app.route("/api/v1/predict", methods=["POST"])
def api_predict():
    """JSON prediction endpoint — mirrors the /predict route but accepts JSON."""
    try:
        body = request.get_json(force=True)
        if not body:
            return jsonify({"error": "JSON body required"}), 400

        meta_path_req = body.get("meta_path", "")
        features_req  = body.get("features", {})

        if not meta_path_req:
            return jsonify({"error": "'meta_path' is required"}), 400
        if not features_req:
            return jsonify({"error": "'features' dict is required"}), 400

        models_dir = os.path.abspath(app.config["MODEL_FOLDER"])
        abs_meta   = os.path.abspath(meta_path_req)
        if not abs_meta.startswith(models_dir) or not os.path.exists(abs_meta):
            return jsonify({"error": "meta_path not found or invalid"}), 404

        with open(abs_meta) as f:
            meta = json.load(f)

        feature_names = meta["feature_names"]
        row = {}
        for feat in feature_names:
            val = features_req.get(feat, 0.0)
            try:
                row[feat] = float(val)
            except (TypeError, ValueError):
                return jsonify({"error": f"Feature '{feat}' must be numeric"}), 400

        input_df  = pd.DataFrame([row], columns=feature_names)
        scaler    = joblib.load(meta["scaler_path"])
        model     = joblib.load(meta["model_path"])
        input_arr = input_df.astype("float64").values

        ip = meta.get("imputer_path", "")
        if ip and os.path.exists(ip):
            input_arr = joblib.load(ip).transform(input_arr)

        input_scaled = scaler.transform(input_arr)
        prediction   = model.predict(input_scaled)[0]

        confidence = None
        if meta["is_classification"] and hasattr(model, "predict_proba"):
            proba      = model.predict_proba(input_scaled)[0]
            confidence = round(float(max(proba)) * 100, 2)

        return jsonify({
            "prediction":        str(prediction),
            "confidence_pct":    confidence,
            "is_classification": meta["is_classification"],
            "model_name":        meta["model_name"],
        })

    except Exception as exc:
        logger.exception("api_predict failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/v1/train", methods=["POST"])
def api_train():
    """
    Accept a CSV upload + params, run training in a background thread,
    return a job_id immediately (202 Accepted).
    """
    import threading

    try:
        if "file" not in request.files:
            return jsonify({"error": "file is required"}), 400
        file          = request.files["file"]
        target_column = request.form.get("target_column", "").strip()
        models_param  = request.form.get("selected_models", "")

        if not allowed_file(file.filename):
            return jsonify({"error": "Only CSV files accepted"}), 400
        if not target_column:
            return jsonify({"error": "target_column is required"}), 400

        # Save the file first (can't pass file object to thread)
        filename  = unique_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        model_name_mapping = {
            "random_forest": "Random Forest", "xgboost": "XGBoost",
            "lightgbm": "LightGBM", "logistic_regression": "Logistic Regression",
            "knn": "KNN", "svm": "SVM", "decision_tree": "Decision Tree",
            "gradient_boosting": "Gradient Boosting", "svr": "SVR",
        }
        if models_param:
            selected = [model_name_mapping[m.strip()] for m in models_param.split(",")
                        if m.strip() in model_name_mapping]
        else:
            selected = list(model_name_mapping.values())

        job_id = uuid.uuid4().hex
        _api_jobs[job_id] = {
            "status":  "running",
            "created": datetime.datetime.now().isoformat(),
            "result":  None,
            "error":   None,
        }

        def _train_job():
            try:
                df = pd.read_csv(file_path)
                df.columns = (df.columns.str.replace(r"[^a-zA-Z0-9_]", "", regex=True).str.strip())
                X, y, imputer, scaler, is_cls = clean_and_preprocess_data(df, target_column)

                (results, best_model_name, best_model_obj,
                 _, _, _, _, _, _) = train_and_compare_models(X, y, is_cls, selected)

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path  = os.path.join(app.config["MODEL_FOLDER"],  f"{best_model_name}_{timestamp}.pkl")
                scaler_path = os.path.join(app.config["SCALER_FOLDER"], f"scaler_{best_model_name}_{timestamp}.pkl")
                imp_path    = os.path.join(app.config["SCALER_FOLDER"], f"imputer_{best_model_name}_{timestamp}.pkl")
                joblib.dump(best_model_obj, model_path)
                joblib.dump(scaler, scaler_path)
                joblib.dump(imputer, imp_path)

                meta = {
                    "feature_names": list(X.columns),
                    "is_classification": is_cls,
                    "model_name": best_model_name,
                    "scaler_path": scaler_path,
                    "imputer_path": imp_path,
                    "model_path": model_path,
                }
                meta_path = model_path.replace(".pkl", "_meta.json")
                with open(meta_path, "w") as mf:
                    json.dump(meta, mf)

                _api_jobs[job_id]["status"] = "done"
                _api_jobs[job_id]["result"] = {
                    "best_model":    best_model_name,
                    "task_type":     "classification" if is_cls else "regression",
                    "meta_path":     meta_path,
                    "model_path":    model_path,
                    "results":       results,
                }
            except Exception as e:
                _api_jobs[job_id]["status"] = "error"
                _api_jobs[job_id]["error"]  = str(e)
                logger.exception("api_train job %s failed", job_id)

        threading.Thread(target=_train_job, daemon=True).start()

        return jsonify({
            "job_id":     job_id,
            "status":     "running",
            "poll_url":   f"/api/v1/jobs/{job_id}",
        }), 202

    except Exception as exc:
        logger.exception("api_train failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/v1/jobs/<job_id>", methods=["GET"])
def api_job_status(job_id):
    """Poll training job status."""
    job = _api_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "job_id":  job_id,
        "status":  job["status"],
        "created": job["created"],
        "result":  job["result"],
        "error":   job["error"],
    })


@app.route("/api/v1/runs", methods=["GET"])
def api_runs():
    """List saved runs for the authenticated user."""
    uid = current_user_id()
    if not uid:
        return jsonify({"error": "Authentication required — log in at /login"}), 401
    conn = get_db()
    rows = conn.execute(
        "SELECT id, created, dataset_name, target_column, task_type, best_model, best_score, n_models FROM runs WHERE user_id=? ORDER BY created DESC LIMIT 50",
        (uid,)
    ).fetchall()
    conn.close()
    return jsonify({"runs": [dict(r) for r in rows]})


if __name__ == "__main__":
    app.run(debug=True)