"""
AutoML Platform — test suite
Run with:  pytest test_app.py -v
"""
import io
import json
import pytest
import pandas as pd
import numpy as np

import app as automl_app
from app import (
    detect_classification,
    clean_and_preprocess_data,
    _is_id_or_ts_col,
)


# ── Helper ──────────────────────────────────────────────────────────────────
def _make_csv(df: pd.DataFrame) -> io.BytesIO:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


def _upload(client, csv_bytes, target, models=("random_forest",)):
    csv_bytes.seek(0)
    data = {
        "file":            (csv_bytes, "data.csv"),
        "target_column":   target,
        "selected_models": list(models),
    }
    return client.post("/upload", data=data, content_type="multipart/form-data")


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests — preprocessing helpers
# ═══════════════════════════════════════════════════════════════════════════

class TestDetectClassification:
    def test_binary_int_target(self):
        y = pd.Series([0, 1, 0, 1, 0, 1])
        assert detect_classification(y) is True

    def test_multiclass_target(self):
        y = pd.Series([0, 1, 2, 0, 1, 2, 0])
        assert detect_classification(y) is True

    def test_float_regression_target(self):
        y = pd.Series([1.2, 3.4, 5.6, 7.8, 9.0, 2.1, 4.3, 6.5, 8.7, 0.9, 1.1])
        assert detect_classification(y) is False

    def test_string_class_labels(self):
        y = pd.Series(["cat", "dog", "cat", "bird", "dog"])
        assert detect_classification(y) is True

    def test_many_unique_integers_is_regression(self):
        y = pd.Series(range(50), dtype=float)
        assert detect_classification(y) is False


class TestIdColumnDetection:
    """Test the word-boundary ID/timestamp column detector."""
    SHOULD_DROP = [
        "PassengerId", "user_id", "order_id", "ID", "id",
        "StudentId", "RecordID", "timestamp", "Timestamp",
        "customer_id", "ids",
    ]
    SHOULD_KEEP = [
        "SepalWidthCm", "PetalWidthCm", "valid", "liquid",
        "period", "Width", "Validity", "avoid", "solid",
        "rapid", "fluid", "candidate", "SibSp", "Fare",
    ]

    def test_drops_id_cols(self):
        for col in self.SHOULD_DROP:
            assert automl_app._is_id_or_ts_col(col), \
                f"Expected '{col}' to be dropped but it was kept"

    def test_keeps_feature_cols(self):
        for col in self.SHOULD_KEEP:
            assert not automl_app._is_id_or_ts_col(col), \
                f"Expected '{col}' to be kept but it was dropped"


class TestPreprocessing:
    def test_basic_preprocessing_shape(self):
        df = pd.DataFrame({
            "age":   [25, 32, 41, 19, 55, 28, 63, 47, 36, 22],
            "score": [70, 85, 90, 60, 95, 75, 88, 82, 78, 65],
            "label": [0, 1, 1, 0, 1, 0, 1, 1, 1, 0],
        })
        X, y, imputer, scaler, is_cls = clean_and_preprocess_data(df, "label")
        assert X.shape[0] == 10
        assert X.shape[1] == 2   # age, score
        assert is_cls is True

    def test_nan_imputation(self):
        df = pd.DataFrame({
            "a":     [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "b":     [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        })
        X, y, imputer, scaler, is_cls = clean_and_preprocess_data(df, "label")
        assert not X.isnull().any().any(), "NaNs should be imputed"

    def test_categorical_encoding(self):
        df = pd.DataFrame({
            "gender": ["M","F","M","F","M","F","M","F","M","F"],
            "age":    [25, 32, 41, 19, 55, 28, 63, 47, 36, 22],
            "label":  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        })
        X, y, imputer, scaler, is_cls = clean_and_preprocess_data(df, "label")
        # gender should be one-hot encoded into gender_M or gender_F
        assert any("gender" in c for c in X.columns)

    def test_id_column_dropped(self):
        df = pd.DataFrame({
            "user_id": range(10),
            "age":     [25, 32, 41, 19, 55, 28, 63, 47, 36, 22],
            "label":   [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        })
        X, y, imputer, scaler, is_cls = clean_and_preprocess_data(df, "label")
        assert "user_id" not in X.columns

    def test_high_cardinality_dropped(self):
        df = pd.DataFrame({
            "name":  [f"Person_{i}" for i in range(20)],
            "score": range(20),
            "label": [i % 2 for i in range(20)],
        })
        X, y, imputer, scaler, is_cls = clean_and_preprocess_data(df, "label")
        assert "name" not in X.columns

    def test_duplicate_rows_removed(self):
        df = pd.DataFrame({
            "a": [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "b": [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        })
        X, y, imputer, scaler, is_cls = clean_and_preprocess_data(df, "y")
        assert X.shape[0] == 10   # one duplicate removed

    def test_returns_correct_tuple_length(self):
        df = pd.DataFrame({
            "x": range(10),
            "y": [0,1]*5,
        })
        result = clean_and_preprocess_data(df, "y")
        assert len(result) == 5  # X, y, imputer, scaler, is_classification

    def test_scaled_features_zero_mean(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "y": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        })
        X, y, imputer, scaler, is_cls = clean_and_preprocess_data(df, "y")
        assert abs(X["a"].mean()) < 1e-9, "StandardScaler should zero-centre features"


# ═══════════════════════════════════════════════════════════════════════════
# Route tests — /get_columns
# ═══════════════════════════════════════════════════════════════════════════

class TestGetColumns:
    def test_returns_columns(self, client, classification_csv):
        resp = client.post("/get_columns",
                           data={"file": (classification_csv, "data.csv")},
                           content_type="multipart/form-data")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "columns" in data
        assert "age" in data["columns"]

    def test_returns_stats(self, client, classification_csv):
        resp = client.post("/get_columns",
                           data={"file": (classification_csv, "data.csv")},
                           content_type="multipart/form-data")
        data = json.loads(resp.data)
        assert data["n_rows"] == 20
        assert data["n_cols"] == 4
        assert "col_info" in data
        assert "preview" in data

    def test_rejects_non_csv(self, client):
        buf = io.BytesIO(b"not a csv")
        resp = client.post("/get_columns",
                           data={"file": (buf, "data.txt")},
                           content_type="multipart/form-data")
        assert resp.status_code == 400
        assert "error" in json.loads(resp.data)

    def test_no_file_returns_400(self, client):
        resp = client.post("/get_columns", data={},
                           content_type="multipart/form-data")
        assert resp.status_code == 400


# ═══════════════════════════════════════════════════════════════════════════
# Route tests — /check_target_column
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckTargetColumn:
    def test_detects_classification(self, client, classification_csv):
        resp = client.post("/check_target_column",
                           data={"file": (classification_csv, "d.csv"),
                                 "target_column": "purchased"},
                           content_type="multipart/form-data")
        data = json.loads(resp.data)
        assert data["task_type"] == "classification"
        assert len(data["models"]) > 0

    def test_detects_regression(self, client, regression_csv):
        resp = client.post("/check_target_column",
                           data={"file": (regression_csv, "d.csv"),
                                 "target_column": "target"},
                           content_type="multipart/form-data")
        data = json.loads(resp.data)
        assert data["task_type"] == "regression"

    def test_missing_target_returns_error(self, client, classification_csv):
        resp = client.post("/check_target_column",
                           data={"file": (classification_csv, "d.csv"),
                                 "target_column": "nonexistent_col"},
                           content_type="multipart/form-data")
        assert resp.status_code == 400


# ═══════════════════════════════════════════════════════════════════════════
# Route tests — authentication
# ═══════════════════════════════════════════════════════════════════════════

class TestAuth:
    def _register(self, client, username="testuser", password="testpass123"):
        return client.post("/register",
                           data={"username": username,
                                 "password": password,
                                 "confirm":  password},
                           follow_redirects=True)

    def test_register_success(self, client):
        resp = self._register(client)
        assert resp.status_code == 200
        assert b"log in" in resp.data.lower() or b"login" in resp.data.lower()

    def test_register_duplicate_username(self, client):
        self._register(client, "dupeuser")
        resp = self._register(client, "dupeuser")
        assert b"taken" in resp.data.lower() or b"already" in resp.data.lower()

    def test_register_short_password(self, client):
        resp = self._register(client, password="abc")
        assert b"6" in resp.data or b"short" in resp.data.lower()

    def test_login_success(self, client):
        self._register(client)
        resp = client.post("/login",
                           data={"username": "testuser", "password": "testpass123"},
                           follow_redirects=True)
        assert resp.status_code == 200

    def test_login_wrong_password(self, client):
        self._register(client)
        resp = client.post("/login",
                           data={"username": "testuser", "password": "wrongpassword"},
                           follow_redirects=True)
        assert b"invalid" in resp.data.lower()

    def test_history_requires_login(self, client):
        resp = client.get("/history", follow_redirects=False)
        assert resp.status_code in (302, 401)

    def test_logout_clears_session(self, client):
        self._register(client)
        client.post("/login",
                    data={"username": "testuser", "password": "testpass123"})
        client.get("/logout")
        resp = client.get("/history", follow_redirects=False)
        assert resp.status_code in (302, 401)


# ═══════════════════════════════════════════════════════════════════════════
# Route tests — /eda_target
# ═══════════════════════════════════════════════════════════════════════════

class TestEdaTarget:
    def test_classification_distribution(self, client, classification_csv):
        resp = client.post("/eda_target",
                           data={"file":          (classification_csv, "d.csv"),
                                 "target_column":  "purchased"},
                           content_type="multipart/form-data")
        data = json.loads(resp.data)
        assert data["task_type"] == "classification"
        assert data["n_classes"] == 2
        assert "dist_plot" in data

    def test_regression_stats_returned(self, client, regression_csv):
        resp = client.post("/eda_target",
                           data={"file":         (regression_csv, "d.csv"),
                                 "target_column": "target"},
                           content_type="multipart/form-data")
        data = json.loads(resp.data)
        assert data["task_type"] == "regression"
        assert data["target_min"] is not None
        assert data["target_max"] > data["target_min"]


# ═══════════════════════════════════════════════════════════════════════════
# Route tests — API endpoints
# ═══════════════════════════════════════════════════════════════════════════

class TestRestApi:
    def test_models_endpoint_all(self, client):
        resp = client.get("/api/v1/models")
        data = json.loads(resp.data)
        assert "models" in data
        assert "classification" in data["models"]
        assert "regression" in data["models"]

    def test_models_endpoint_filtered(self, client):
        resp = client.get("/api/v1/models?task=classification")
        data = json.loads(resp.data)
        assert "Random Forest" in data["models"]

    def test_openapi_spec_returns_json(self, client):
        resp = client.get("/api/v1/openapi.json")
        assert resp.status_code == 200
        spec = json.loads(resp.data)
        assert spec["openapi"].startswith("3.")
        assert "/predict" in spec["paths"]

    def test_runs_requires_auth(self, client):
        resp = client.get("/api/v1/runs")
        assert resp.status_code == 401

    def test_predict_missing_body(self, client):
        resp = client.post("/api/v1/predict",
                           data=json.dumps({}),
                           content_type="application/json")
        assert resp.status_code == 400

    def test_predict_invalid_meta_path(self, client):
        resp = client.post("/api/v1/predict",
                           data=json.dumps({
                               "meta_path": "../../etc/passwd",
                               "features":  {"x": 1}
                           }),
                           content_type="application/json")
        assert resp.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════
# Security tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSecurity:
    def test_path_traversal_blocked_predict(self, client):
        # Send a valid-shaped body — the path traversal should be caught
        # by the path security check (404) not the body validation (400)
        resp = client.post("/api/v1/predict",
                           data=json.dumps({
                               "meta_path": "../../../etc/passwd",
                               "features":  {"x": 1.0}
                           }),
                           content_type="application/json")
        assert resp.status_code == 404

    def test_path_traversal_blocked_download(self, client):
        resp = client.get("/download_model?path=../../etc/passwd")
        assert resp.status_code == 404

    def test_invalid_report_id_blocked(self, client):
        resp = client.get("/download_report/../../etc/passwd")
        assert resp.status_code in (400, 404)

    def test_non_csv_upload_blocked(self, client):
        buf = io.BytesIO(b"<script>alert(1)</script>")
        resp = client.post("/get_columns",
                           data={"file": (buf, "evil.html")},
                           content_type="multipart/form-data")
        assert resp.status_code == 400