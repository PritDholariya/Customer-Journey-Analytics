# Customer Journey Analytics & Behavioral Segmentation Platform

An end-to-end analytics and machine learning system that transforms raw e-commerce clickstream events into actionable behavioral segments, pinpoints purchase funnel drop-off leakages, and serves real-time insights via a FastAPI REST service.

Built on the real-world **Retailrocket e-commerce dataset** (~2.76M clickstream events over ~4.5 months).

---

## Architecture Overview

```
                                      [ Raw Events Log ]
                                      (events.csv - Kaggle)
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │  Phase 1: Canonical Acquisition   │
                              │  (Unix ms → DateTime, Sampling)  │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │  Phase 2: Cleaning & Sessions    │
                              │  (Deduplication, 30m Inactivity) │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │  Phase 3: Feature Engineering    │
                              │  (24 behavioral & intent metrics)│
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │  Phase 4: Unsupervised ML        │
                              │  (StandardScaler, K-Means, DBSCAN│
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │  Phase 5: Funnel Drop-off Engine │
                              │  (view → addtocart → transaction)│
                              └────────┬───────────────┬─────────┘
                                       │               │
                 ┌─────────────────────┘               └─────────────────────┐
                 ▼                                                           ▼
┌─────────────────────────────────┐                         ┌─────────────────────────────────┐
│ Phase 6: Visualizations         │                         │ Phase 7: PostgreSQL Storage     │
│ (PCA scatter, distributions)    │                         │ (SQLAlchemy ORM & Relational DB)│
└─────────────────────────────────┘                         └────────────────┬────────────────┘
                                                                             │
                                                                             ▼
                                                            ┌─────────────────────────────────┐
                                                            │ Phase 8: FastAPI REST Service   │
                                                            │ (/segments, /funnels, /users)   │
                                                            └─────────────────────────────────┘
```

---

## Discovered Behavioral Segments

Unsupervised clustering ($K=5$ K-Means, validated with Silhouette score and Davies-Bouldin index) revealed 5 distinct customer personas:

| Segment ID | Segment Name | Share | Key Behavioral Pattern | Business Recommendation |
| :---: | :--- | :---: | :--- | :--- |
| **0** | **Casual Visitors** | ~72% | Minimal engagement (~2.5 views, 0 carts, 1.4 sessions). | Improve landing page relevance and search discovery. |
| **1** | **Active Browsers** | ~14% | High page views (~12 views), but very low cart intent ($1.3\%$). | Personalized recommendations, social proof, and discount nudges. |
| **2** | **Cart Abandoners** | ~6% | Extremely high cart intent ($64.8\%$), but zero purchases. | Target with exit-intent popups, cart recovery emails, and checkout friction audits. |
| **3** | **Loyal Window Shoppers** | ~5% | Repeat visits across ~49 days ($3.9$ sessions), but rarely cart or buy. | Retargeting campaigns, limited-time offers, and loyalty incentives. |
| **4** | **Converters** | ~3% | High closing efficiency ($94.1\%$ purchase rate from cart). | VIP loyalty rewards, cross-sell/upsell post-purchase sequences. |

---

## Purchase Funnel Analysis

The journey stages are evaluated sequentially:
$$\text{View (Product Discovery)} \longrightarrow \text{Add-to-Cart (Purchase Intent)} \longrightarrow \text{Transaction (Conversion)}$$

- **View $\rightarrow$ Cart Drop-off:** The dominant drop-off for casual visitors and window shoppers.
- **Cart $\rightarrow$ Transaction Drop-off:** The primary bottleneck for cart abandoners, representing high-intent lost revenue.

---

## Repository Structure

```
Customer-Journey-Analytics/
├── data/
│   ├── raw/                        # Kaggle raw CSVs (events.csv, category_tree.csv)
│   └── processed/                  # Cleaned parquet artifacts & figures
│       ├── events_canonical.parquet
│       ├── events_clean.parquet
│       ├── user_features.parquet
│       ├── user_segments.parquet
│       ├── segment_funnels.parquet
│       └── figures/                # Saved charts (PCA, distributions, funnels)
├── docs/
│   └── BUILD_PLAN.md               # Detailed architectural specification
├── src/
│   └── cja/
│       ├── __init__.py
│       ├── config.py               # Centralized configuration & environment variables
│       ├── acquisition.py          # Phase 1: Ingestion & schema standardization
│       ├── cleaning.py             # Phase 2: Deduplication & 30-min sessionization
│       ├── features.py             # Phase 3: Behavioral feature engineering (log-transformed)
│       ├── clustering.py           # Phase 4: K-Means (K=5) & DBSCAN clustering
│       ├── funnel.py               # Phase 5: Per-segment drop-off analysis
│       ├── visualization.py        # Phase 6: Matplotlib/Seaborn diagnostic visual plots
│       ├── database.py             # Phase 7: SQLAlchemy schema & PostgreSQL loading
│       └── api.py                  # Phase 8: FastAPI REST API service
├── tests/                          # 39 Unit tests across phases (pytest)
├── .env.example                    # Template for database credentials
├── pyproject.toml                  # Editable packaging configuration
└── requirements.txt                # Python dependencies
```

---

## Setup & Installation

### 1. Clone & Environment Setup

```bash
# Clone the repository
git clone https://github.com/PritDholariya/Customer-Journey-Analytics.git
cd Customer-Journey-Analytics

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate       # On Linux / macOS
# or: .\venv\Scripts\activate   # On Windows

# Install in editable mode
pip install -e .
```

### 2. Dataset Acquisition

Download the Retailrocket dataset from [Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset):
- Place `events.csv` inside `data/raw/`
- Or use the Kaggle CLI:
  ```bash
  kaggle datasets download -d retailrocket/ecommerce-dataset -p data/raw --unzip
  ```

### 3. Environment Configuration

Copy the example `.env` file and set your PostgreSQL credentials (optional for testing, required for Phase 7/8 production database connection):
```bash
cp .env.example .env
```

---

## Step-by-Step Execution Guide

You can run each phase independently to reproduce the pipeline from scratch:

```bash
# Phase 1: Ingest raw events and map to canonical schema
python src/cja/acquisition.py

# Phase 2: Deduplicate and sessionize events (30-minute inactivity threshold)
python src/cja/cleaning.py

# Phase 3: Engineer 24 user-level behavioral features
python src/cja/features.py

# Phase 4: Run K-Means & DBSCAN clustering and profile segments
python src/cja/clustering.py

# Phase 5: Calculate stage conversion and funnel drop-off rates
python src/cja/funnel.py

# Phase 6: Generate all diagnostic visualizations into data/processed/figures/
python src/cja/visualization.py

# Phase 7: Create PostgreSQL relational tables and load data
python src/cja/database.py

# Phase 8: Launch the FastAPI REST service
uvicorn cja.api:app --reload --host 127.0.0.1 --port 8000
```

---

## REST API Reference (FastAPI)

Once the API is running, visit interactive documentation at:
- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service health status and database connectivity check |
| `GET` | `/segments` | List all 5 behavioral segments with their aggregate feature profiles |
| `GET` | `/segments/{segment_name}` | Detailed behavioral metrics for a specific segment |
| `GET` | `/segments/{segment_name}/funnel` | Funnel conversion and drop-off metrics for a specific segment |
| `GET` | `/funnels` | Side-by-side funnel metrics across all segments |
| `GET` | `/users/{user_id}/segment` | Real-time lookup of segment classification for a given `user_id` |


---

## Testing & Quality Assurance

Unit tests are written with `pytest` and validate data integrity, sessionization logic, schema guarantees, and API endpoints using an isolated in-memory SQLite database:

```bash
# Run the test suite
pytest tests/ -v
```

All 39 unit tests cover:
- Timestamp conversions and canonical schema mappings (Phase 1)
- Session cut-offs and deduplication (Phase 2)
- Zero-division guards and log transforms on skewed counts (Phase 3)
- Feature scaling and cluster assignment validity (Phase 4)
- Monotonicity of funnel conversion rates (Phase 5)
- SQLAlchemy relational schema integrity and foreign keys (Phase 7)
- FastAPI endpoint responses and Pydantic validation (Phase 8)

---

## License

This project is licensed under the MIT License.
