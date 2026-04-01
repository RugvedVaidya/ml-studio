"""
pytest configuration and shared fixtures for AutoML Platform tests.
"""
import io
import os
import sys
import sqlite3
import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as automl_app


class _NoCloseConn:
    """
    Wraps a sqlite3.Connection and makes .close() a no-op.
    Routes call conn.close() after every operation; in tests we need
    the same connection to stay alive across multiple requests.
    """
    def __init__(self, conn):
        self._conn = conn

    def execute(self, *a, **kw):
        return self._conn.execute(*a, **kw)

    def executescript(self, *a, **kw):
        return self._conn.executescript(*a, **kw)

    def commit(self):
        return self._conn.commit()

    def close(self):
        pass  # intentional no-op — fixture closes the real conn after the test

    def __getattr__(self, name):
        return getattr(self._conn, name)


@pytest.fixture
def client():
    """
    Flask test client backed by a fresh in-memory SQLite DB per test.
    Patches get_db() so every route call returns the same live connection.
    conn.close() inside routes becomes a no-op so the DB stays open.
    """
    automl_app.app.config["TESTING"] = True
    automl_app.app.config["WTF_CSRF_ENABLED"] = False

    # Real in-memory connection — lives for the duration of this test
    real_conn = sqlite3.connect(":memory:")
    real_conn.row_factory = sqlite3.Row
    wrapped = _NoCloseConn(real_conn)

    # Create schema
    real_conn.executescript("""
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
    real_conn.commit()

    original_get_db = automl_app.get_db
    automl_app.get_db = lambda: wrapped

    with automl_app.app.test_client() as c:
        yield c

    automl_app.get_db = original_get_db
    real_conn.close()   # actually close now that the test is done


@pytest.fixture
def classification_csv():
    """Minimal binary-classification CSV as bytes."""
    df = pd.DataFrame({
        "age":      [25, 32, 41, 19, 55, 28, 63, 47, 36, 22,
                     30, 45, 51, 27, 38, 60, 23, 44, 31, 50],
        "salary":   [50000, 72000, 90000, 30000, 110000, 60000, 130000, 95000,
                     78000, 42000, 65000, 98000, 115000, 55000, 83000, 125000,
                     47000, 92000, 68000, 105000],
        "gender":   ["M","F","M","F","M","F","M","F","M","F",
                     "M","F","M","F","M","F","M","F","M","F"],
        "purchased":[0,1,1,0,1,0,1,1,1,0,0,1,1,0,1,1,0,1,1,1],
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


@pytest.fixture
def regression_csv():
    """Minimal regression CSV as bytes."""
    import numpy as np
    rng = np.random.default_rng(42)
    n   = 30
    x1  = rng.uniform(0, 10, n)
    x2  = rng.uniform(0, 5,  n)
    y   = 3 * x1 + 2 * x2 + rng.normal(0, 1, n)
    df  = pd.DataFrame({"x1": x1, "x2": x2, "target": y})
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


@pytest.fixture
def csv_with_missing():
    """CSV that has NaN values in feature columns."""
    import numpy as np
    df = pd.DataFrame({
        "a": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, np.nan, 8.0, 9.0, 10.0,
              1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "b": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
              10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "label": [0,1,0,1,0,1,0,1,0,1, 0,1,0,1,0,1,0,1,0,1],
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf