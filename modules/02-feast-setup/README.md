# Module 02: Feast Setup and Configuration

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Beginner |
| **Prerequisites** | Module 01 completed, Docker running |

---

## Learning Objectives

By the end of this module, you will be able to:

- Install and configure Feast for local development
- Understand the `feature_store.yaml` configuration file
- Connect Feast to PostgreSQL (offline store) and Redis (online store)
- Initialize a feature repository and apply definitions
- Use the Feast CLI to inspect your feature store

---

## Concepts

### What Is Feast?

Feast (Feature Store) is an open-source feature store that bridges the gap between data engineering and machine learning. It provides:

- **A unified serving layer** for both training and inference
- **Point-in-time correct** feature retrieval for training data
- **Low-latency online serving** via Redis, DynamoDB, or other key-value stores
- **A feature registry** that tracks schemas, ownership, and lineage

### Feast Architecture

```
+-------------------------------------------------------------------+
|                        Feast Feature Store                         |
|                                                                    |
|  +------------------+    +------------------+    +---------------+ |
|  |  feature_store   |    |  Feature         |    |  CLI / SDK    | |
|  |  .yaml           |    |  Registry        |    |  feast apply  | |
|  |  (configuration) |    |  (metadata DB)   |    |  feast mat..  | |
|  +--------+---------+    +--------+---------+    +-------+-------+ |
|           |                       |                      |         |
|  +--------v---------+    +--------v---------+    +-------v-------+ |
|  |  Offline Store   |    |  Online Store    |    |  Feature      | |
|  |  (PostgreSQL)    |    |  (Redis)         |    |  Server       | |
|  |  - Historical    |    |  - Latest values |    |  (FastAPI)    | |
|  |  - Training data |    |  - Low latency   |    |  - REST API   | |
|  +------------------+    +------------------+    +---------------+ |
+-------------------------------------------------------------------+
```

### The `feature_store.yaml` File

This file is the central configuration for your Feast project. It tells Feast where to store metadata, where to read historical data, and where to write online features.

```yaml
# feature_repo/feature_store.yaml
project: feature_store_lab       # Unique project name
provider: local                  # local, gcp, or aws

registry:
  registry_type: sql
  path: postgresql://feast:feast_password@localhost:5432/feast_offline
  cache_ttl_seconds: 60         # How long to cache registry metadata

offline_store:
  type: postgres                 # Could also be: file, bigquery, snowflake, redshift
  host: localhost
  port: 5432
  database: feast_offline
  db_schema: public
  user: feast
  password: feast_password

online_store:
  type: redis                    # Could also be: sqlite, dynamodb, datastore
  connection_string: localhost:6379

entity_key_serialization_version: 2
```

### Provider Options Comparison

| Provider | Offline Store | Online Store | Registry | Best For |
|---|---|---|---|---|
| **Local** | File / Postgres | SQLite / Redis | File / SQL | Development, small teams |
| **GCP** | BigQuery | Datastore / Bigtable | GCS / SQL | Google Cloud workloads |
| **AWS** | Redshift / S3 | DynamoDB | S3 / SQL | Amazon Web Services |

---

## Hands-On Lab

### Prerequisites Check

```bash
# Start the infrastructure
docker compose up -d postgres redis

# Verify services are healthy
docker compose ps

# Expected output:
# feast-postgres   running (healthy)
# feast-redis      running (healthy)

# Test PostgreSQL connection
docker exec feast-postgres pg_isready -U feast
# /var/run/postgresql:5432 - accepting connections

# Test Redis connection
docker exec feast-redis redis-cli ping
# PONG
```

### Exercise 1: Install Feast and Initialize the Repository

**Goal:** Get a working Feast installation connected to your infrastructure.

**Step 1:** Install dependencies.

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install all dependencies
pip install -r requirements.txt

# Verify Feast is installed
feast version
```

**Step 2:** Examine the feature store configuration.

```bash
# Look at the config file
cat feature_repo/feature_store.yaml
```

Key things to understand:
- `project` -- namespace that prevents collisions if multiple teams share infrastructure
- `registry` -- where Feast stores metadata about your features
- `offline_store` -- where historical feature data lives
- `online_store` -- where the latest feature values are cached for fast serving

**Step 3:** Initialize the feature store.

```bash
cd feature_repo

# Apply the feature definitions to the registry
feast apply

# Expected output:
# Created entity customer
# Created entity product
# Created feature view customer_transaction_features
# Created feature view customer_profile_features
# Created feature view product_catalog_features
# Created on demand feature view transaction_risk_features
# Created feature service fraud_detection
# Created feature service recommendation_engine
```

### Exercise 2: Explore the Feast CLI

**Goal:** Learn the CLI commands you will use daily.

```bash
# List all registered entities
feast entities list

# Output:
# NAME       DESCRIPTION                              JOIN KEYS
# customer   A customer who interacts with our platform  [customer_id]
# product    A product available in the catalog          [product_id]

# List all feature views
feast feature-views list

# Output:
# NAME                              ENTITIES    TYPE
# customer_transaction_features     [customer]  FeatureView
# customer_profile_features         [customer]  FeatureView
# product_catalog_features          [product]   FeatureView
# transaction_risk_features         []          OnDemandFeatureView

# List feature services
feast feature-services list

# Output:
# NAME                    FEATURES
# fraud_detection         [customer_transaction_features, ...]
# recommendation_engine   [customer_profile_features, ...]

# Get details about a specific feature view
feast feature-views describe customer_transaction_features
```

### Exercise 3: Verify Store Connectivity

**Goal:** Confirm that Feast can talk to both PostgreSQL and Redis.

```python
# modules/02-feast-setup/lab/starter/verify_stores.py
from feast import FeatureStore
import redis
import psycopg2

# 1. Verify Feast can connect to the registry
store = FeatureStore(repo_path="feature_repo")
print(f"Project: {store.project}")
print(f"Feature views: {[fv.name for fv in store.list_feature_views()]}")
print(f"Entities: {[e.name for e in store.list_entities()]}")

# 2. Verify Redis connectivity
r = redis.Redis(host="localhost", port=6379)
assert r.ping(), "Redis is not responding"
print(f"Redis connected: {r.ping()}")

# 3. Verify PostgreSQL connectivity
conn = psycopg2.connect(
    host="localhost", port=5432,
    database="feast_offline", user="feast", password="feast_password"
)
cur = conn.cursor()
cur.execute("SELECT version();")
print(f"PostgreSQL: {cur.fetchone()[0][:30]}...")
conn.close()

print("\nAll stores connected successfully!")
```

### Exercise 4: Understanding Configuration Options

**Goal:** Modify the configuration and observe the effects.

**Step 1:** Try changing the online store to SQLite (for comparison):

```yaml
# Temporarily modify feature_store.yaml
online_store:
  type: sqlite
  path: data/online_store.db
```

```bash
feast apply
# Notice: Feast creates a local SQLite database file
ls data/online_store.db
```

**Step 2:** Revert back to Redis:

```yaml
online_store:
  type: redis
  connection_string: localhost:6379
```

```bash
feast apply
```

**Why Redis over SQLite?** SQLite works for development, but Redis provides the sub-millisecond latency needed for production inference. In this lab, we use Redis from the start to match real-world setups.

---

## Key Configuration Reference

| Parameter | Description | Example |
|---|---|---|
| `project` | Unique namespace for your feature store | `feature_store_lab` |
| `provider` | Infrastructure provider | `local`, `gcp`, `aws` |
| `registry.registry_type` | Where to store metadata | `sql`, `file` |
| `registry.cache_ttl_seconds` | How long to cache registry reads | `60` |
| `offline_store.type` | Backend for historical data | `postgres`, `file`, `bigquery` |
| `online_store.type` | Backend for low-latency serving | `redis`, `sqlite`, `dynamodb` |
| `entity_key_serialization_version` | Key encoding format | `2` (recommended) |

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Docker services not running | `feast apply` fails with connection errors | Run `docker compose up -d postgres redis` first |
| Wrong connection string in YAML | Registry errors on apply | Double-check host, port, user, password |
| Forgetting to run `feast apply` | Feature views not found | Always apply after changing definitions |
| Using file registry in multi-user setup | Registry conflicts | Use SQL registry for team environments |

---

## Self-Check Questions

1. What does `feast apply` do under the hood?
2. Why do we use a SQL registry instead of a file registry?
3. What is the role of `entity_key_serialization_version`?
4. How would you switch the offline store from PostgreSQL to BigQuery?
5. What happens if the Redis online store is down -- can you still train models?

---

## You Know You Have Completed This Module When...

- [ ] Feast is installed and `feast version` works
- [ ] `feast apply` completes without errors
- [ ] You can list entities, feature views, and feature services via the CLI
- [ ] PostgreSQL and Redis connections are verified
- [ ] Validation script passes: `bash modules/02-feast-setup/validation/validate.sh`

---

## Troubleshooting

**Issue: `feast apply` fails with "connection refused"**
```bash
# Make sure Docker services are running
docker compose up -d postgres redis
docker compose ps  # Check health status
```

**Issue: `ModuleNotFoundError: No module named 'feast'`**
```bash
# Activate your virtual environment
source .venv/bin/activate
pip install -r requirements.txt
```

**Issue: PostgreSQL authentication failed**
```bash
# Check your .env matches docker-compose.yml
cat .env | grep POSTGRES
docker compose logs postgres | tail -5
```

---

**Next: [Module 03 - Feature Definitions -->](../03-feature-definitions/)**
