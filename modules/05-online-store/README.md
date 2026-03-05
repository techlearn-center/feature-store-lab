# Module 05: Online Feature Serving -- Redis and Low-Latency Reads

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Intermediate |
| **Prerequisites** | Module 04 completed, Redis running |

---

## Learning Objectives

By the end of this module, you will be able to:

- Materialize features from the offline store into Redis
- Retrieve features from the online store with sub-millisecond latency
- Build a FastAPI endpoint that serves features for real-time inference
- Benchmark online store performance
- Understand when to use online vs. offline retrieval

---

## Concepts

### The Online Store: Why Redis?

During inference, ML models need features **fast**. A recommendation model serving a web page has less than 50ms for the entire request. Reading from a data warehouse would take seconds. That is why we use an online store.

```
Production Inference Flow:

 User Request                     ML Model
      |                              ^
      v                              |
 [Feature Server]  -- 2-5ms -->  [Features]
      |
      v
 [Redis Online Store]
  - Pre-computed values
  - Key-value lookup by entity ID
  - Sub-millisecond response
```

**Redis** is the most popular choice for online feature stores because:
- **Speed**: Sub-millisecond reads for key-value lookups
- **Simplicity**: No complex query language needed
- **Scalability**: Redis Cluster for horizontal scaling
- **Persistence**: AOF/RDB for durability

### How Materialization Works

Materialization copies the latest feature values from the offline store into the online store.

```
Offline Store (PostgreSQL/Parquet)          Online Store (Redis)
+------------------------------------------+   +---------------------------+
| customer_id | spend_30d | timestamp      |   | C001 -> {spend_30d: 1500} |
|-------------|-----------|----------------|   | C002 -> {spend_30d: 3200} |
| C001        | 1200      | Jan 1          |   | C003 -> {spend_30d: 800}  |
| C001        | 1500      | Feb 1   -------|-->|                           |
| C002        | 3000      | Jan 1          |   +---------------------------+
| C002        | 3200      | Feb 1   -------|-->  Only the LATEST value
| C001        | 1800      | Mar 1   -------|-->  per entity is stored
+------------------------------------------+
```

---

## Hands-On Lab

### Prerequisites Check

```bash
# Ensure Redis is running
docker exec feast-redis redis-cli ping
# PONG

# Ensure feature data exists
ls data/processed/customer_transactions.parquet

# If not, generate it
python -m src.pipelines.feature_pipeline --stage generate
python -m src.pipelines.feature_pipeline --stage compute
```

### Exercise 1: Materialize Features to Redis

**Goal:** Push computed features from Parquet files into the Redis online store.

```bash
cd feature_repo

feast apply

# Materialize features
feast materialize 2024-01-01T00:00:00 $(date -u +%Y-%m-%dT%H:%M:%S)
```

```python
# Or materialize programmatically
from feast import FeatureStore
from datetime import datetime, timedelta

store = FeatureStore(repo_path="feature_repo")

store.materialize(
    start_date=datetime.now() - timedelta(days=2),
    end_date=datetime.now(),
)
print("Materialization complete!")
```

### Exercise 2: Retrieve Features from the Online Store

**Goal:** Fetch features by entity ID with low latency.

```python
# modules/05-online-store/lab/starter/online_retrieval.py
import time
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo")

# Single entity lookup
start = time.time()
features = store.get_online_features(
    features=[
        "customer_transaction_features:total_transactions_30d",
        "customer_transaction_features:total_spend_30d",
        "customer_transaction_features:avg_transaction_amount_30d",
        "customer_profile_features:customer_segment",
        "customer_profile_features:lifetime_value",
    ],
    entity_rows=[{"customer_id": "C00001"}],
).to_dict()
elapsed_ms = (time.time() - start) * 1000

print(f"Latency: {elapsed_ms:.2f}ms")
print(f"Features: {features}")

# Batch lookup
start = time.time()
batch_features = store.get_online_features(
    features=[
        "customer_transaction_features:total_spend_30d",
        "customer_transaction_features:avg_transaction_amount_30d",
    ],
    entity_rows=[
        {"customer_id": "C00001"},
        {"customer_id": "C00002"},
        {"customer_id": "C00003"},
        {"customer_id": "C00010"},
        {"customer_id": "C00050"},
    ],
).to_dict()
elapsed_ms = (time.time() - start) * 1000

print(f"\nBatch latency (5 entities): {elapsed_ms:.2f}ms")
for i, cid in enumerate(batch_features["customer_id"]):
    spend = batch_features["total_spend_30d"][i]
    print(f"  {cid}: total_spend_30d = {spend}")
```

### Exercise 3: Build and Test the Feature Serving API

**Goal:** Use the FastAPI feature server to serve features over REST.

```bash
# Start the server
uvicorn src.serving.feature_server:app --host 0.0.0.0 --port 8000

# In another terminal, test the endpoints:

# Health check
curl http://localhost:8000/health

# Get customer features
curl -X POST http://localhost:8000/features/customer \
  -H "Content-Type: application/json" \
  -d '{"customer_ids": ["C00001", "C00002"]}'

# Get product features
curl -X POST http://localhost:8000/features/product \
  -H "Content-Type: application/json" \
  -d '{"product_ids": ["P00001", "P00010"]}'

# Get fraud detection features (includes on-demand computation)
curl -X POST http://localhost:8000/features/fraud \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "C00001",
    "transaction_amount": 500.00,
    "merchant_category": "electronics"
  }'
```

### Exercise 4: Benchmark Online Store Performance

**Goal:** Measure latency characteristics of your online store.

```python
# modules/05-online-store/lab/starter/benchmark.py
import time
import statistics
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo")

features_to_fetch = [
    "customer_transaction_features:total_transactions_30d",
    "customer_transaction_features:total_spend_30d",
    "customer_transaction_features:avg_transaction_amount_30d",
]

# Warm up
for _ in range(5):
    store.get_online_features(
        features=features_to_fetch,
        entity_rows=[{"customer_id": "C00001"}],
    )

# Benchmark single entity lookups
latencies = []
for i in range(100):
    customer_id = f"C{i:05d}"
    start = time.time()
    store.get_online_features(
        features=features_to_fetch,
        entity_rows=[{"customer_id": customer_id}],
    )
    latencies.append((time.time() - start) * 1000)

print("=== Single Entity Lookup Benchmark (100 iterations) ===")
print(f"  p50: {statistics.median(latencies):.2f}ms")
print(f"  p95: {sorted(latencies)[94]:.2f}ms")
print(f"  p99: {sorted(latencies)[98]:.2f}ms")
print(f"  Mean: {statistics.mean(latencies):.2f}ms")

# Benchmark batch lookups
batch_sizes = [1, 5, 10, 50, 100]
for batch_size in batch_sizes:
    entity_rows = [{"customer_id": f"C{i:05d}"} for i in range(batch_size)]
    batch_latencies = []
    for _ in range(20):
        start = time.time()
        store.get_online_features(
            features=features_to_fetch,
            entity_rows=entity_rows,
        )
        batch_latencies.append((time.time() - start) * 1000)
    p50 = statistics.median(batch_latencies)
    print(f"  Batch size {batch_size:3d}: p50={p50:.2f}ms")
```

### Exercise 5: Inspect Redis Directly

**Goal:** Understand how Feast stores data in Redis.

```bash
# Connect to Redis CLI
docker exec -it feast-redis redis-cli

# Count all keys
DBSIZE

# Check memory usage
INFO memory

# Monitor real-time commands (then trigger a lookup in another terminal)
MONITOR
```

---

## Latency Budget Reference

| Component | Typical Latency | Notes |
|---|---|---|
| Redis key lookup | 0.1 - 1ms | Single entity, local network |
| Feast SDK overhead | 1 - 5ms | Serialization, connection pool |
| Network (same region) | 0.5 - 2ms | Between app and Redis |
| On-demand transform | 1 - 10ms | Depends on computation complexity |
| **Total feature fetch** | **3 - 15ms** | End-to-end with Feast |

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Forgetting to materialize | Online features return `null` | Run `feast materialize` after computing features |
| Redis memory full | Materialization fails with OOM | Increase `maxmemory` or reduce feature cardinality |
| Materializing too infrequently | Stale features served | Schedule at least as often as data updates |
| Not warming up connections | First request very slow | Pre-connect in server startup event |

---

## Self-Check Questions

1. What is the difference between `get_online_features()` and `get_historical_features()`?
2. Why does the online store only keep the latest value per entity?
3. How would you handle a feature that needs minute-level updates?
4. What happens when you request features for an entity not in the online store?
5. How would you scale Redis for 100,000 QPS?

---

## You Know You Have Completed This Module When...

- [ ] Features are materialized in Redis
- [ ] You can retrieve features via `get_online_features()` with <15ms latency
- [ ] The FastAPI feature server responds to all endpoints
- [ ] You have benchmarked single and batch lookups
- [ ] Validation script passes: `bash modules/05-online-store/validation/validate.sh`

---

## Troubleshooting

**Issue: All features return `null` after materialization**
```python
# Verify data timestamps are within the materialization window
# Check feast materialize output for row counts
```

**Issue: Redis connection refused**
```bash
docker compose up -d redis
docker exec feast-redis redis-cli ping
```

---

**Next: [Module 06 - Feature Pipelines -->](../06-feature-pipelines/)**
