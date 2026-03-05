# Capstone Project: Production Feature Store for E-Commerce ML

## Overview

This capstone project combines everything you learned across all 10 modules into a single, production-grade feature store implementation. You will build an end-to-end ML feature platform that ingests raw e-commerce data, computes features, serves them for both training and real-time inference, validates data quality, and runs with monitoring in Docker.

This is the project you will showcase to hiring managers.

---

## The Challenge

Build a complete feature store that supports **two ML models** -- a fraud detection system and a product recommendation engine -- sharing features from a single, well-governed feature platform.

Your solution must demonstrate:

1. **Feature Engineering** (Modules 01-03): Define entities, data sources, feature views, and feature services using Feast
2. **Dual Store Serving** (Modules 04-05): Offline store (PostgreSQL) for training data with point-in-time joins, online store (Redis) for sub-15ms inference
3. **Automated Pipelines** (Module 06): Feature pipeline that generates, computes, validates, and materializes features on a schedule
4. **Real-Time Features** (Modules 07, 09): On-demand feature views that compute risk signals at inference time, with data quality gates
5. **Feature Governance** (Module 08): Tagged, documented, and searchable feature definitions with team ownership
6. **Production Readiness** (Module 10): Docker deployment, health checks, monitoring, and a CI/CD workflow

---

## Architecture

```
+------------------------------------------------------------------+
|                     Capstone Architecture                         |
|                                                                    |
|  +------------------+                                              |
|  | Raw Data         |  Synthetic e-commerce transactions           |
|  | (generate stage) |  1000 customers, 200 products, 50K txns     |
|  +--------+---------+                                              |
|           |                                                        |
|           v                                                        |
|  +--------+---------+    +------------------+                      |
|  | Feature Pipeline |    | Data Quality     |                      |
|  | (compute stage)  |--->| (GE validation)  |                      |
|  | - Aggregations   |    | - Null checks    |                      |
|  | - Window funcs   |    | - Range checks   |                      |
|  | - Derived feats  |    | - Freshness      |                      |
|  +--------+---------+    +--------+---------+                      |
|           |                       |                                |
|           v                       v                                |
|  +------------------+    +------------------+                      |
|  | Offline Store    |    | Online Store     |                      |
|  | (PostgreSQL)     |    | (Redis)          |                      |
|  | - Parquet files  |    | - Materialized   |                      |
|  | - PIT joins      |    |   latest values  |                      |
|  +--------+---------+    +--------+---------+                      |
|           |                       |                                |
|           v                       v                                |
|  +------------------+    +------------------+    +---------------+ |
|  | Training Data    |    | Feature Server   |    | On-Demand     | |
|  | Generation       |    | (FastAPI)        |    | Features      | |
|  | - Entity DFs     |    | /features/cust   |    | - Z-scores    | |
|  | - Hist. features |    | /features/prod   |    | - Risk flags  | |
|  | - Model training |    | /features/fraud  |    | - Ratios      | |
|  +------------------+    +------------------+    +---------------+ |
|                                                                    |
|  Feature Services:                                                 |
|    fraud_detection      -> txn_features + profile + risk (on-demand)|
|    recommendation_engine -> profile + product_catalog              |
+------------------------------------------------------------------+
```

---

## Requirements

### Must Have

- [ ] **Feast feature repository** with `feature_store.yaml` connecting to PostgreSQL and Redis
- [ ] **At least 3 entities** defined (customer, product, and one more of your choice)
- [ ] **At least 3 feature views** with typed schemas, TTLs, and tags
- [ ] **At least 2 feature services** mapping to specific ML models
- [ ] **At least 1 on-demand feature view** that combines stored and request-time data
- [ ] **Feature pipeline** that generates raw data, computes features, and writes to the offline store
- [ ] **Materialization** to push features to Redis for online serving
- [ ] **FastAPI feature server** with endpoints for customer, product, and fraud features
- [ ] **Point-in-time correct** training data generation using `get_historical_features()`
- [ ] **Data quality validation** using Great Expectations before materialization
- [ ] **Docker Compose** deployment with PostgreSQL, Redis, app, and Jupyter
- [ ] **Health check endpoint** on the feature server
- [ ] **Documentation** for the entire system (this README, inline code comments)
- [ ] **Validation script** passes all checks

### Nice to Have

- [ ] CI/CD pipeline (GitHub Actions) for feature definition validation
- [ ] Prometheus metrics on the feature server
- [ ] Feature freshness monitoring dashboard
- [ ] Feature drift detection (PSI or similar)
- [ ] Feature search/discovery tool
- [ ] Performance benchmarks for online store latency
- [ ] Automated backfill capability
- [ ] Multiple environment support (dev/staging/prod configs)

---

## Getting Started

```bash
# 1. Clone and enter the project
git clone https://github.com/techlearn-center/feature-store-lab.git
cd feature-store-lab

# 2. Copy the environment file
cp .env.example .env

# 3. Start infrastructure
docker compose up -d postgres redis

# 4. Create a Python virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# 5. Generate data and compute features
python -m src.pipelines.feature_pipeline --stage generate
python -m src.pipelines.feature_pipeline --stage compute

# 6. Apply Feast definitions
cd feature_repo
feast apply
cd ..

# 7. Materialize features to Redis
python -m src.pipelines.feature_pipeline --stage materialize

# 8. Start the feature server
uvicorn src.serving.feature_server:app --host 0.0.0.0 --port 8000

# 9. Test it
curl http://localhost:8000/health
curl -X POST http://localhost:8000/features/customer \
  -H "Content-Type: application/json" \
  -d '{"customer_ids": ["C00001"]}'

# 10. Validate your work
bash capstone/validation/validate.sh
```

---

## Evaluation Criteria

| Criteria | Weight | What We Look For |
|---|---|---|
| **Functionality** | 30% | All endpoints work, training data generates correctly, pipeline runs end-to-end |
| **Architecture** | 20% | Clean separation (entities, views, services), proper use of online/offline stores |
| **Data Quality** | 15% | Validation gates prevent bad data from reaching the online store |
| **Production Readiness** | 15% | Docker deployment, health checks, monitoring, error handling |
| **Code Quality** | 10% | Clean code, meaningful comments, proper error handling, type hints |
| **Documentation** | 10% | Clear README, architecture diagram, inline docs explaining "why" not just "what" |

---

## Solution

The `solution/` directory contains a reference implementation. **Try to complete the capstone yourself first** -- that is what builds real skills and interview confidence.

Key solution files:
- `feature_repo/feature_definitions.py` -- All Feast definitions
- `feature_repo/feature_store.yaml` -- Feast configuration
- `src/pipelines/feature_pipeline.py` -- Full feature pipeline
- `src/serving/feature_server.py` -- FastAPI feature server
- `docker-compose.yml` -- Complete infrastructure setup

---

## Interview Talking Points

When presenting this capstone to hiring managers, be ready to discuss:

1. **"Why did you choose Redis for the online store?"**
   - Sub-millisecond reads, simple key-value model, persistence options, cluster scaling

2. **"How do you prevent training-serving skew?"**
   - Single feature definition used for both offline (training) and online (serving) retrieval

3. **"How do you handle data quality?"**
   - Great Expectations validation gates in the pipeline, blocking materialization on failure

4. **"How would you scale this to 100K requests per second?"**
   - Redis Cluster, multiple FastAPI workers behind a load balancer, horizontal scaling

5. **"What happens if the feature pipeline fails?"**
   - Online store continues serving last-known-good values (TTL-based expiry), alerting triggers investigation, manual backfill after fix

---

## Showcasing to Hiring Managers

1. **Fork this repo** to your personal GitHub
2. **Add your solution** with detailed commit messages
3. **Update this README** with your specific architecture decisions
4. **Record a demo video** showing the pipeline running and API responding (optional but impressive)
5. **Reference it on your resume** and LinkedIn profile
6. **Be ready to demo it** in technical interviews -- have Docker Compose ready to go

See [docs/portfolio-guide.md](../docs/portfolio-guide.md) for detailed guidance.
