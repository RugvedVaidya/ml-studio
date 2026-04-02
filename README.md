# 🤖 AutoML Hyperparameter Optimization Platform

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Online-success)](https://ml-studio-lvy2.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3-lightgrey)](https://flask.palletsprojects.com)
[![Tests](https://img.shields.io/badge/Tests-41%20passing-brightgreen)](test_app.py)

A full-stack, browser-based AutoML platform built with Flask and scikit-learn. Upload any CSV dataset, explore it visually, train and compare multiple ML models with automated hyperparameter tuning, and get explainable AI insights — all without writing a single line of code.

> **Live progress streaming** — watch each model train in real time via Server-Sent Events instead of staring at a loading spinner.

---

## 📸 Feature Overview

| Feature | Details |
|---|---|
| **Task types** | Classification, Regression, Clustering |
| **Models** | 8 supervised + 3 clustering algorithms |
| **Hyperparameter tuning** | RandomizedSearchCV with 5-fold cross-validation |
| **Explainability** | SHAP bar + beeswarm plots, feature importance charts |
| **EDA** | Correlation heatmap, target distribution, column stats, data preview |
| **Live progress** | Server-Sent Events stream per-model results as they finish |
| **Auth + History** | Register/login, SQLite run history, re-use saved models |
| **Prediction page** | Enter feature values → get prediction + confidence score |
| **REST API** | 5 JSON endpoints + interactive Swagger UI at `/api/docs` |
| **Plots** | ROC-AUC, Precision-Recall, Confusion Matrix, Residuals, PCA scatter |

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/automl-platform.git
cd automl-platform
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## 📁 Project Structure

```
automl-platform/
│
├── app.py                  # Main Flask application (2100+ lines)
│
├── templates/
│   ├── index.html          # Upload + EDA page
│   ├── result.html         # Training results page
│   ├── cluster.html        # Clustering results page
│   ├── predict.html        # Live prediction form
│   ├── history.html        # Run history dashboard
│   ├── login.html          # Auth — login
│   ├── register.html       # Auth — register
│   └── swagger.html        # REST API documentation UI
│
├── uploads/                # Uploaded CSV files (auto-created)
├── models/                 # Saved model .pkl files (auto-created)
├── scalers/                # Saved scaler + imputer .pkl files (auto-created)
├── reports/                # Saved JSON reports (auto-created)
│
├── automl.db               # SQLite database — users + run history (auto-created)
├── requirements.txt
└── README.md
```

---

## 🎯 How It Works

### Step-by-step flow

```
Upload CSV  →  EDA (stats, heatmap, preview)
           →  Select target column  →  Target distribution chart
           →  Select models  →  Run (live SSE progress)
           →  Results page (metrics, SHAP, plots, feature importance)
           →  Make predictions  →  Download model / report
```

### Data preprocessing pipeline (automatic)

1. **Column cleaning** — removes special characters, strips whitespace
2. **Duplicate removal** — drops exact duplicate rows
3. **ID/timestamp detection** — word-boundary matching drops `PassengerId`, `user_id`, `timestamp` but keeps `Width`, `valid`, `period`
4. **Time-string detection** — drops columns containing `HH:MM AM/PM` patterns
5. **High-cardinality drop** — object columns with >50% unique values (e.g. `Name`, `Ticket`) are dropped to avoid dummy explosion
6. **One-hot encoding** — remaining object columns encoded with `pd.get_dummies`
7. **Median imputation** — `SimpleImputer(strategy="median")` fills all NaN values
8. **Standard scaling** — `StandardScaler` applied to all features
9. **SMOTE** — applied to training split only when class imbalance detected (classification)

All preprocessing artefacts (imputer + scaler) are saved as `.pkl` files so prediction uses the exact same pipeline.

---

## 🧠 Supported Models

### Classification (8 models)
| Model | Key |
|---|---|
| Random Forest | `random_forest` |
| XGBoost | `xgboost` |
| LightGBM | `lightgbm` |
| Logistic Regression | `logistic_regression` |
| K-Nearest Neighbours | `knn` |
| Support Vector Machine | `svm` |
| Decision Tree | `decision_tree` |
| Gradient Boosting | `gradient_boosting` |

**Metrics:** Accuracy, F1 Score (weighted), 5-fold CV mean ± std, ROC-AUC, Precision-Recall, Confusion Matrix

### Regression (7 models)
| Model | Key |
|---|---|
| Random Forest | `random_forest` |
| XGBoost | `xgboost` |
| LightGBM | `lightgbm` |
| K-Nearest Neighbours | `knn` |
| Support Vector Regressor | `svr` |
| Decision Tree | `decision_tree` |
| Gradient Boosting | `gradient_boosting` |

**Metrics:** R² Score, RMSE, 5-fold CV mean ± std, Residual plots, Predicted vs Actual

### Clustering (3 algorithms, no target needed)
| Algorithm | Strategy |
|---|---|
| KMeans | Optimal k chosen by silhouette score (tested k=2..8) |
| DBSCAN | `eps` auto-tuned via 90th percentile of 5-NN distances |
| Agglomerative | Ward linkage with same k as KMeans |

**Metrics:** Silhouette Score, Davies-Bouldin Score, Cluster sizes, PCA 2D scatter plots, Elbow curve

---

## 📊 SHAP Explainability

After training, SHAP values are computed for the best model:

- **Bar plot** — mean absolute SHAP value per feature (global importance)
- **Beeswarm plot** — each dot = one sample; colour = feature value (red=high, blue=low); x-axis = SHAP impact

The explainer is chosen automatically:
- `TreeExplainer` for tree-based models (fast, exact)
- `KernelExplainer` for SVM, KNN, Logistic Regression (approximation on 50-sample background)

---

## ⚡ Live Training Progress (SSE)

Instead of a blocking page load, training uses **Server-Sent Events**:

1. Browser POSTs file + settings to `/train_stream`
2. Server starts a background thread, returns `job_id` immediately
3. Browser opens `EventSource` to `/train_progress/<job_id>`
4. Server streams events as each model finishes:
   ```
   event: progress
   data: ✅ Random Forest — Accuracy 82.3%, F1 81.9% (9.4s)
   ```
5. On completion, an inline results table appears with a "View Full Results" button

Three event types:
| Event | Meaning |
|---|---|
| `message` | General log line (preprocessing, saving) |
| `progress` | Per-model result with metric + timing |
| `done` | Training complete — includes full results JSON |
| `error` | Something went wrong — message displayed in red |

---

## 🔐 Authentication & History

- **Register** at `/register` — username + password (SHA-256 hashed, stored in SQLite)
- **Login** at `/login` — creates a Flask session
- Every training run by a logged-in user is saved to the `runs` table
- **History page** at `/history` shows all past runs with:
  - Dataset name, target column, task type
  - Best model + score (colour-coded green/yellow/red)
  - Training date and number of models run
  - Links to re-download the report or open the prediction page
- The app works fully without an account — auth is optional

---

## 🔮 Prediction Page

After training, click **"Make Prediction"** on the results page:

- Dynamic form with one input per feature (post-encoding)
- Values are run through the saved imputer → scaler → model pipeline
- Returns predicted class or value
- For classifiers: shows a **confidence % bar** (green ≥90%, yellow ≥70%, red <70%)
- Last 5 predictions shown as a history list with timestamps

---

## 🌐 REST API

Interactive docs at **`http://127.0.0.1:5000/api/docs`** (Swagger UI).

### Endpoints

#### `GET /api/v1/models`
List available models, optionally filtered by task type.

```bash
curl "http://localhost:5000/api/v1/models?task=classification"
```

```json
{
  "task": "classification",
  "models": ["Random Forest", "XGBoost", "LightGBM", ...]
}
```

---

#### `POST /api/v1/predict`
Run a prediction using a saved model. Returns result immediately.

```bash
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "meta_path": "models/Random Forest_20250329_143000_meta.json",
    "features": {
      "Pclass": 3,
      "Age": 22,
      "SibSp": 1,
      "Parch": 0,
      "Fare": 7.25,
      "Sex_male": 1,
      "Embarked_Q": 0,
      "Embarked_S": 1
    }
  }'
```

```json
{
  "prediction": "0",
  "confidence_pct": 84.3,
  "is_classification": true,
  "model_name": "Random Forest"
}
```

---

#### `POST /api/v1/train`
Start an async training job. Returns a `job_id` immediately (202 Accepted).

```bash
curl -X POST http://localhost:5000/api/v1/train \
  -F "file=@titanic.csv" \
  -F "target_column=Survived" \
  -F "selected_models=random_forest,xgboost,logistic_regression"
```

```json
{
  "job_id": "a3f2e1b4...",
  "status": "running",
  "poll_url": "/api/v1/jobs/a3f2e1b4..."
}
```

---

#### `GET /api/v1/jobs/{job_id}`
Poll training job status.

```bash
curl http://localhost:5000/api/v1/jobs/a3f2e1b4...
```

```json
{
  "job_id": "a3f2e1b4...",
  "status": "done",
  "result": {
    "best_model": "XGBoost",
    "task_type": "classification",
    "meta_path": "models/XGBoost_20250329_143512_meta.json",
    "results": { ... }
  }
}
```

---

#### `GET /api/v1/runs`
List your saved runs (session authentication required — log in at `/login` first).

```bash
curl http://localhost:5000/api/v1/runs
```

---

## 📦 Requirements

```
Flask==2.3.3
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
matplotlib==3.8.4
seaborn==0.13.2
joblib==1.4.2
jinja2==3.1.3
Werkzeug==2.3.8
xgboost
lightgbm
imbalanced-learn
shap
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🐛 Known Limitations

- **Single-user SSE job store** — `_sse_jobs` and `_sse_queues` are in-memory Python dicts. They reset on server restart and are not shared across multiple workers. For production, replace with Redis.
- **Synchronous `/upload` route** — the classic form-submit path still blocks the request thread. Use the SSE path (`startTraining()`) for large datasets.
- **SHAP on KernelExplainer** — slow for large datasets with SVM/KNN. SHAP silently skips and logs a warning if it times out.
- **No GPU support** — XGBoost and LightGBM run on CPU only.
- **Debug mode** — `app.run(debug=True)` is fine for development but must be changed to a production WSGI server (gunicorn, waitress) before deploying.

---

## 🛠️ Configuration

| Environment variable | Default | Description |
|---|---|---|
| `FLASK_SECRET_KEY` | Random bytes | Session signing key — set a fixed value in production |

Folder paths (`uploads/`, `models/`, `scalers/`, `reports/`) are created automatically on first run.

---

## 🗺️ Route Reference

| Method | Route | Description |
|---|---|---|
| GET | `/` | Home — upload + EDA page |
| POST | `/get_columns` | Return column list + EDA stats for a CSV |
| POST | `/eda_target` | Return target distribution chart |
| POST | `/check_target_column` | Detect task type + return model list |
| POST | `/upload` | Classic blocking training (returns `result.html`) |
| POST | `/train_stream` | Start SSE training job, return `job_id` |
| GET | `/train_progress/<job_id>` | SSE stream — per-model progress events |
| POST | `/cluster` | Run clustering analysis |
| GET | `/predict_page` | Prediction form for a saved model |
| POST | `/predict` | Run a prediction, return JSON |
| GET | `/download_model` | Download best model `.pkl` |
| GET | `/download_report/<id>` | Download JSON training report |
| GET | `/register` | Registration form |
| POST | `/register` | Create account |
| GET | `/login` | Login form |
| POST | `/login` | Authenticate |
| GET | `/logout` | Clear session |
| GET | `/history` | Run history (login required) |
| GET | `/api/docs` | Swagger UI |
| GET | `/api/v1/openapi.json` | OpenAPI 3.0 spec |
| GET | `/api/v1/models` | List available models |
| POST | `/api/v1/predict` | JSON prediction endpoint |
| POST | `/api/v1/train` | Start async training job |
| GET | `/api/v1/jobs/<job_id>` | Poll job status |
| GET | `/api/v1/runs` | List saved runs (auth required) |

---

## 🤝 Contributing

Pull requests welcome. For major changes please open an issue first to discuss what you'd like to change.

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.