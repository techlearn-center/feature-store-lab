# Module 10: Production Deployment -- Monitoring, Scaling, and CI/CD for Features

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Advanced |
| **Prerequisites** | Modules 01-09 completed |

---

## Learning Objectives

By the end of this module, you will be able to:

- Deploy a feature store with Docker Compose for production-like environments
- Implement monitoring for feature freshness, latency, and quality
- Scale the online store for high-throughput workloads
- Set up CI/CD pipelines for feature definitions
- Design a runbook for common production incidents

---

## Concepts

### Production Feature Store Architecture

```
                    Production Feature Store
+------------------------------------------------------------------+
|                                                                    |
|  CI/CD Pipeline                                                    |
|  +--------------------+                                            |
|  | PR: new feature    |---> Lint + Test ---> feast apply (staging) |
|  | definition         |---> Integration test ---> feast apply (prod)|
|  +--------------------+                                            |
|                                                                    |
|  Data Plane                                                        |
|  +------------------+    +------------------+    +---------------+ |
|  | Feature Pipeline |    | Offline Store    |    | Online Store  | |
|  | (Airflow/Prefect)|    | (PostgreSQL)     |    | (Redis        | |
|  | - Compute hourly |--->| - Training data  |--->|  Cluster)     | |
|  | - Validate data  |    | - Backfills      |    | - <5ms reads  | |
|  +------------------+    +------------------+    +-------+-------+ |
|                                                          |         |
|  Serving Layer                                           |         |
|  +------------------+                            +-------v-------+ |
|  | Load Balancer    |<-------------------------->| Feature       | |
|  | (nginx/ALB)      |                            | Server        | |
|  +------------------+                            | (FastAPI x N) | |
|                                                  +---------------+ |
|  Observability                                                     |
|  +------------------+    +------------------+    +---------------+ |
|  | Metrics          |    | Logs             |    | Alerts        | |
|  | (Prometheus)     |    | (ELK/CloudWatch) |    | (PagerDuty)   | |
|  +------------------+    +------------------+    +---------------+ |
+------------------------------------------------------------------+
```

### Key Production Concerns

| Concern | What to Monitor | Target |
|---|---|---|
| **Feature freshness** | Time since last materialization | < 1 hour |
| **Serving latency** | p50, p95, p99 of feature retrieval | p95 < 15ms |
| **Online store health** | Redis memory, connections, errors | < 80% memory |
| **Pipeline reliability** | Success rate, duration, data quality | > 99% success |
| **Feature coverage** | % of requests with non-null features | > 95% |

---

## Hands-On Lab

### Exercise 1: Deploy with Docker Compose

**Goal:** Stand up the complete feature store infrastructure.

```bash
# Copy environment file
cp .env.example .env

# Start all services
docker compose up -d

# Verify all services are healthy
docker compose ps

# Expected:
# feast-postgres   running (healthy)
# feast-redis      running (healthy)
# feast-app        running
# feast-jupyter    running

# Check logs
docker compose logs app --tail 20
docker compose logs postgres --tail 10
docker compose logs redis --tail 10

# Test the feature server
curl http://localhost:8000/health
```

### Exercise 2: Add Prometheus Metrics to the Feature Server

**Goal:** Instrument the feature server with latency, error, and throughput metrics.

```python
# modules/10-production-feature-store/lab/starter/monitored_server.py
"""
Add Prometheus metrics to the feature server.
"""
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Define metrics
REQUEST_COUNT = Counter(
    "feature_server_requests_total",
    "Total feature server requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "feature_server_request_duration_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

FEATURE_FETCH_LATENCY = Histogram(
    "feature_fetch_duration_seconds",
    "Time to fetch features from online store",
    ["feature_view"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
)

FEATURE_NULL_RATE = Gauge(
    "feature_null_rate",
    "Percentage of null values in feature responses",
    ["feature_view"],
)

ONLINE_STORE_ERRORS = Counter(
    "online_store_errors_total",
    "Errors when connecting to the online store",
    ["error_type"],
)


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
        ).inc()

        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(duration)

        return response


# Add to your FastAPI app:
# app.add_middleware(MetricsMiddleware)
#
# @app.get("/metrics")
# async def metrics():
#     return Response(
#         content=generate_latest(),
#         media_type="text/plain",
#     )
```

### Exercise 3: Build a Feature Freshness Dashboard

**Goal:** Create a monitoring script that checks feature freshness and alerts on staleness.

```python
# modules/10-production-feature-store/lab/starter/freshness_dashboard.py
import pandas as pd
from datetime import datetime, timedelta
from feast import FeatureStore
import json

class FeatureFreshnessDashboard:
    """Monitor feature freshness across all feature views."""

    def __init__(self, repo_path: str = "feature_repo"):
        self.store = FeatureStore(repo_path=repo_path)

    def check_all(self, max_age_hours: float = 2.0) -> dict:
        """Check freshness of all feature data files."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "max_age_hours": max_age_hours,
            "views": {},
            "overall_status": "HEALTHY",
        }

        # Check each data source
        checks = {
            "customer_transaction_features": "data/processed/customer_transactions.parquet",
            "product_catalog_features": "data/processed/product_features.parquet",
        }

        for view_name, path in checks.items():
            try:
                df = pd.read_parquet(path)
                latest_ts = df["event_timestamp"].max()
                age_hours = (datetime.now() - latest_ts).total_seconds() / 3600

                status = "FRESH" if age_hours <= max_age_hours else "STALE"
                if status == "STALE":
                    report["overall_status"] = "DEGRADED"

                report["views"][view_name] = {
                    "latest_timestamp": str(latest_ts),
                    "age_hours": round(age_hours, 2),
                    "status": status,
                    "row_count": len(df),
                }
            except Exception as e:
                report["views"][view_name] = {
                    "status": "ERROR",
                    "error": str(e),
                }
                report["overall_status"] = "DEGRADED"

        return report

    def check_online_store(self, sample_entities: list[dict]) -> dict:
        """Verify features are available in the online store."""
        report = {"entities_checked": len(sample_entities), "results": []}

        for entity in sample_entities:
            try:
                result = self.store.get_online_features(
                    features=["customer_transaction_features:total_spend_30d"],
                    entity_rows=[entity],
                ).to_dict()

                has_data = result["total_spend_30d"][0] is not None
                report["results"].append({
                    "entity": entity,
                    "has_data": has_data,
                })
            except Exception as e:
                report["results"].append({
                    "entity": entity,
                    "has_data": False,
                    "error": str(e),
                })

        coverage = sum(1 for r in report["results"] if r["has_data"]) / len(report["results"])
        report["coverage_pct"] = round(coverage * 100, 1)
        report["status"] = "HEALTHY" if coverage > 0.95 else "DEGRADED"

        return report

    def print_report(self):
        """Print a formatted freshness report."""
        offline = self.check_all()
        sample = [{"customer_id": f"C{i:05d}"} for i in range(10)]
        online = self.check_online_store(sample)

        print("=" * 60)
        print(f"FEATURE STORE HEALTH REPORT - {offline['timestamp']}")
        print("=" * 60)
        print(f"\nOverall Status: {offline['overall_status']}")

        print("\n--- Offline Store ---")
        for name, info in offline["views"].items():
            status_icon = "OK" if info["status"] == "FRESH" else "STALE"
            print(f"  [{status_icon}] {name}: age={info.get('age_hours', 'N/A')}h, "
                  f"rows={info.get('row_count', 'N/A')}")

        print(f"\n--- Online Store ---")
        print(f"  Coverage: {online['coverage_pct']}%")
        print(f"  Status: {online['status']}")


if __name__ == "__main__":
    dashboard = FeatureFreshnessDashboard()
    dashboard.print_report()
```

### Exercise 4: Set Up CI/CD for Feature Definitions

**Goal:** Automate testing and deployment of feature definition changes.

```yaml
# .github/workflows/feature-store-ci.yml
name: Feature Store CI/CD

on:
  pull_request:
    paths:
      - 'feature_repo/**'
      - 'src/pipelines/**'
  push:
    branches: [main]
    paths:
      - 'feature_repo/**'
      - 'src/pipelines/**'

jobs:
  validate:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_DB: feast_offline
          POSTGRES_USER: feast
          POSTGRES_PASSWORD: feast_password
        ports:
          - 5432:5432
      redis:
        image: redis:7
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Lint feature definitions
        run: |
          python -c "
          import ast
          ast.parse(open('feature_repo/feature_definitions.py').read())
          print('Feature definitions: syntax OK')
          "

      - name: Apply feature definitions (dry run)
        run: |
          cd feature_repo
          feast apply
        env:
          POSTGRES_HOST: localhost
          REDIS_HOST: localhost

      - name: Verify registry
        run: |
          cd feature_repo
          feast entities list
          feast feature-views list
          feast feature-services list

      - name: Run feature pipeline (generate + compute)
        run: |
          python -m src.pipelines.feature_pipeline --stage generate --n-customers 50 --n-products 20 --n-transactions 500
          python -m src.pipelines.feature_pipeline --stage compute

      - name: Validate feature data quality
        run: |
          python -c "
          import pandas as pd
          df = pd.read_parquet('data/processed/customer_transactions.parquet')
          assert len(df) > 0, 'No customer features generated'
          assert df['total_spend_30d'].min() >= 0, 'Negative spend detected'
          print(f'Validated {len(df)} customer feature rows')
          "

  deploy:
    needs: validate
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Deploy feature definitions
        run: |
          echo "In production, this would:"
          echo "  1. feast apply to production registry"
          echo "  2. Trigger materialization job"
          echo "  3. Verify online store health"
```

### Exercise 5: Create a Production Runbook

**Goal:** Document procedures for common production incidents.

```python
# modules/10-production-feature-store/lab/starter/runbook.py
"""
Feature Store Production Runbook
================================

Incident 1: Features are stale (not updating)
----------------------------------------------
Symptoms:
  - Freshness dashboard shows STALE status
  - Model performance degrading

Diagnosis:
  1. Check pipeline status: is the scheduler running?
  2. Check pipeline logs for errors
  3. Check offline store: are new Parquet files being written?
  4. Check materialization: did the last feast materialize succeed?

Resolution:
  - If scheduler stopped: restart the scheduler service
  - If pipeline error: check data source connectivity
  - If materialization failed: run manual materialize
"""

def diagnose_stale_features():
    """Run diagnostic checks for stale features."""
    import os
    from datetime import datetime

    print("=== Diagnosing Stale Features ===\n")

    # Check 1: Data files exist and are recent
    data_files = [
        "data/processed/customer_transactions.parquet",
        "data/processed/product_features.parquet",
    ]
    for path in data_files:
        if os.path.exists(path):
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            age_hours = (datetime.now() - mtime).total_seconds() / 3600
            status = "OK" if age_hours < 2 else "STALE"
            print(f"  [{status}] {path}: modified {age_hours:.1f}h ago")
        else:
            print(f"  [MISSING] {path}")

    # Check 2: Redis connectivity
    try:
        import redis
        r = redis.Redis(host="localhost", port=6379)
        r.ping()
        key_count = r.dbsize()
        print(f"\n  [OK] Redis: connected, {key_count} keys")
    except Exception as e:
        print(f"\n  [ERROR] Redis: {e}")

    # Check 3: PostgreSQL connectivity
    try:
        import psycopg2
        conn = psycopg2.connect(
            host="localhost", port=5432,
            database="feast_offline", user="feast", password="feast_password"
        )
        conn.close()
        print(f"  [OK] PostgreSQL: connected")
    except Exception as e:
        print(f"  [ERROR] PostgreSQL: {e}")

"""
Incident 2: High feature serving latency
-----------------------------------------
Symptoms:
  - p95 latency > 50ms
  - Timeouts in ML model inference

Diagnosis:
  1. Check Redis: INFO stats, memory usage, slow log
  2. Check network: latency between app and Redis
  3. Check batch size: are requests too large?

Resolution:
  - If Redis memory high: increase maxmemory or evict old keys
  - If network latency: co-locate app and Redis
  - If batch too large: split into smaller requests
"""

def diagnose_high_latency():
    """Run diagnostic checks for high latency."""
    import redis
    import time

    r = redis.Redis(host="localhost", port=6379)
    info = r.info()

    print("=== Diagnosing High Latency ===\n")
    print(f"  Redis memory used: {info['used_memory_human']}")
    print(f"  Redis memory peak: {info['used_memory_peak_human']}")
    print(f"  Connected clients: {info['connected_clients']}")
    print(f"  Evicted keys: {info.get('evicted_keys', 0)}")

    # Check Redis latency
    latencies = []
    for _ in range(100):
        start = time.time()
        r.ping()
        latencies.append((time.time() - start) * 1000)

    p50 = sorted(latencies)[49]
    p99 = sorted(latencies)[98]
    print(f"\n  Redis ping p50: {p50:.2f}ms")
    print(f"  Redis ping p99: {p99:.2f}ms")

    if p99 > 5:
        print("  WARNING: Redis latency is high. Check network or Redis load.")

"""
Incident 3: Null features in production
----------------------------------------
Symptoms:
  - Feature responses contain null values
  - Model receiving incomplete inputs

Diagnosis:
  1. Was the entity recently created? (no features yet)
  2. Is materialization running? (features not pushed to online store)
  3. Has the TTL expired? (features too old)
  4. Was the feature view schema changed? (key mismatch)

Resolution:
  - If new entity: expected behavior, handle nulls in model
  - If materialization stopped: trigger manual materialize
  - If TTL expired: extend TTL or increase materialization frequency
"""

if __name__ == "__main__":
    diagnose_stale_features()
    print()
    diagnose_high_latency()
```

---

## Scaling Considerations

| Component | Scale Strategy | When |
|---|---|---|
| **Redis** | Redis Cluster (sharding) | >10GB features or >50K QPS |
| **Feature Server** | Horizontal scaling (multiple FastAPI instances behind LB) | >1K QPS |
| **Offline Store** | PostgreSQL read replicas or migrate to BigQuery/Snowflake | >100GB historical data |
| **Feature Pipeline** | Spark/Flink for distributed computation | >10M entities |
| **Registry** | SQL registry with connection pooling | >10 concurrent users |

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| No monitoring | Issues discovered by end users | Add freshness + latency + error rate metrics from day one |
| No CI/CD for feature definitions | Breaking changes reach production | Validate and test on every PR |
| Single Redis instance | SPOF, cannot scale | Use Redis Cluster or Redis Sentinel |
| No runbook | Long incident resolution times | Document diagnosis and resolution for common failures |
| Scaling feature server without scaling Redis | Redis becomes bottleneck | Monitor Redis before adding more app instances |

---

## Self-Check Questions

1. What are the three most important metrics to monitor for a production feature store?
2. How would you handle a Redis failover without dropping feature requests?
3. What should a CI/CD pipeline for feature definitions test?
4. How would you roll back a bad feature definition change?
5. What is the difference between horizontal and vertical scaling for the online store?

---

## You Know You Have Completed This Module When...

- [ ] You can deploy the full stack with `docker compose up`
- [ ] You have monitoring for feature freshness and serving latency
- [ ] You have a CI/CD pipeline that validates feature definitions
- [ ] You have a production runbook for common incidents
- [ ] Validation script passes: `bash modules/10-production-feature-store/validation/validate.sh`

---

## Troubleshooting

**Issue: Docker Compose services fail to start**
```bash
docker compose down -v   # Remove volumes
docker compose up -d     # Restart fresh
docker compose logs      # Check for errors
```

**Issue: Feature server crashes under load**
```bash
# Increase uvicorn workers
uvicorn src.serving.feature_server:app --workers 4 --host 0.0.0.0 --port 8000
```

**Issue: Redis out of memory**
```bash
# Check current usage
docker exec feast-redis redis-cli INFO memory
# Increase limit in docker-compose.yml maxmemory setting
```

---

**Congratulations! You have completed all 10 modules. Proceed to the [Capstone Project -->](../../capstone/)**
