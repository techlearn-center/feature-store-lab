# Module 01: Feature Store Fundamentals

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Beginner |
| **Prerequisites** | Docker installed, Python 3.10+, basic ML concepts |

---

## Learning Objectives

By the end of this module, you will be able to:

- Explain why feature stores exist and what problems they solve
- Distinguish between online and offline feature stores
- Identify the training-serving skew problem and how feature stores prevent it
- Describe the core components of a feature store architecture
- Run a minimal feature engineering workflow from raw data to model-ready features

---

## Concepts

### Why Do Feature Stores Exist?

In a typical ML system without a feature store, teams face three recurring problems:

1. **Training-serving skew** -- The features computed during training are different from those computed during inference, causing silent model degradation.
2. **Duplicated work** -- Every team rebuilds the same features (e.g., "user's average spend in the last 30 days") from scratch.
3. **Inconsistent data** -- Feature logic lives in scattered notebooks, pipelines, and microservices with no single source of truth.

A feature store is a centralized system that **computes, stores, and serves** ML features consistently across training and inference.

```
+-------------------+        +------------------+        +------------------+
|   Raw Data        |  --->  |  Feature Store   |  --->  |  ML Model        |
|  (events, logs,   |        |  - Compute       |        |  - Training      |
|   transactions)   |        |  - Store         |        |  - Inference     |
+-------------------+        |  - Serve         |        +------------------+
                             +------------------+
                              |               |
                         Offline Store   Online Store
                         (PostgreSQL)      (Redis)
                         Historical        Low-latency
                         training data     serving
```

### Online vs. Offline Stores

| Aspect | Offline Store | Online Store |
|---|---|---|
| **Purpose** | Generate training datasets | Serve features at inference time |
| **Latency** | Seconds to minutes | Sub-millisecond to milliseconds |
| **Storage** | Data warehouse (PostgreSQL, BigQuery, Snowflake) | Key-value store (Redis, DynamoDB) |
| **Access pattern** | Batch queries with point-in-time joins | Key-based lookups by entity ID |
| **Data volume** | Full history (months/years) | Latest values only |
| **Typical user** | Data scientist building training sets | ML model in production |

### Core Architecture Components

```
                          +---------------------+
                          |   Feature Registry  |
                          |  (metadata, schemas,|
                          |   lineage, owners)  |
                          +----------+----------+
                                     |
   +------------------+    +---------+---------+    +------------------+
   | Data Sources     |--->| Feature Pipeline  |--->| Feature Views    |
   | (DB, streams,    |    | (compute, validate|    | (logical groups  |
   |  files, APIs)    |    |  transform)       |    |  of features)    |
   +------------------+    +-------------------+    +--------+---------+
                                                             |
                                              +--------------+--------------+
                                              |                             |
                                     +--------+--------+          +---------+--------+
                                     |  Offline Store  |          |  Online Store    |
                                     |  (historical)   |          |  (low-latency)   |
                                     +-----------------+          +------------------+
```

**Entity** -- The primary key for feature lookups (e.g., `customer_id`, `driver_id`).

**Feature View** -- A named group of related features from a single data source with a defined schema.

**Feature Service** -- A bundle of feature views consumed by a specific ML model.

**Materialization** -- The process of computing features and pushing them from the offline store into the online store.

### The Training-Serving Skew Problem

Consider a fraud detection model. During training, a data scientist computes `avg_transaction_amount_30d` in a Jupyter notebook using pandas. During inference, a backend engineer recomputes it using a SQL query. Subtle differences (timezone handling, null treatment, window boundaries) cause the model to see different feature values than it was trained on.

```python
# Training (notebook) - uses pandas, inclusive window
avg_30d = df[df['date'] >= cutoff].groupby('user_id')['amount'].mean()

# Serving (API) - uses SQL, different boundary logic
# SELECT AVG(amount) FROM transactions WHERE date > cutoff GROUP BY user_id
#                                                     ^^ exclusive vs inclusive
```

A feature store eliminates this by computing features **once** and serving the same values to both training and inference.

---

## Hands-On Lab

### Prerequisites Check

```bash
# Verify Python
python --version  # 3.10+

# Verify Docker
docker --version
docker compose version

# Verify project structure
ls feature_repo/
ls src/pipelines/
```

### Exercise 1: Feature Engineering Without a Feature Store

**Goal:** Experience the pain points that feature stores solve.

**Step 1:** Create a raw dataset manually.

```python
# modules/01-feature-engineering-basics/lab/starter/raw_features.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
n_users = 100
n_transactions = 5000

transactions = pd.DataFrame({
    "user_id": np.random.choice([f"U{i:03d}" for i in range(n_users)], n_transactions),
    "amount": np.round(np.random.lognormal(3, 1, n_transactions), 2),
    "timestamp": [datetime.now() - timedelta(hours=np.random.exponential(200))
                  for _ in range(n_transactions)],
    "category": np.random.choice(["food", "electronics", "clothing", "travel"], n_transactions),
})

print(f"Generated {len(transactions)} transactions for {n_users} users")
print(transactions.head())
```

**Step 2:** Compute features manually for training.

```python
# Feature engineering in a notebook (the "old way")
from datetime import datetime, timedelta

cutoff = datetime.now() - timedelta(days=30)
recent = transactions[transactions["timestamp"] >= cutoff]

training_features = recent.groupby("user_id").agg(
    txn_count_30d=("amount", "count"),
    avg_amount_30d=("amount", "mean"),
    max_amount_30d=("amount", "max"),
    unique_categories=("category", "nunique"),
).reset_index()

print(f"Training features shape: {training_features.shape}")
print(training_features.head())
```

**Step 3:** Now imagine you need to serve these same features in an API. Write a different version.

```python
# Simulating a "serving" implementation (different code, same intent)
import sqlite3

def get_user_features_for_serving(user_id: str) -> dict:
    """This function has DIFFERENT logic than the training code above."""
    conn = sqlite3.connect(":memory:")
    # ... completely different implementation
    # This is where training-serving skew comes from
    pass
```

**Notice the problem?** Two different implementations for the same features.

### Exercise 2: Understanding Online vs. Offline Access Patterns

**Goal:** Understand when you need each store type.

```python
# modules/01-feature-engineering-basics/lab/starter/access_patterns.py
"""
Scenario: E-commerce fraud detection system

For each scenario below, decide: Online store or Offline store?

1. A data scientist needs 6 months of user features to train a new model.
   Answer: OFFLINE -- batch retrieval, point-in-time correctness needed.

2. The fraud API needs a user's avg_spend to score a transaction in <10ms.
   Answer: ONLINE -- single key lookup, low latency required.

3. A weekly report needs feature distributions for model monitoring.
   Answer: OFFLINE -- batch scan, latency not critical.

4. A recommendation model needs product features during page load.
   Answer: ONLINE -- per-request lookup, <50ms budget.
"""

# Simulate the performance difference
import time

# Offline pattern: scan + aggregate (slow, but powerful)
def offline_query(data, user_id):
    """Simulates an offline store query -- full table scan with joins."""
    start = time.time()
    result = data[data["user_id"] == user_id].agg({
        "amount": ["count", "mean", "max"],
        "category": "nunique",
    })
    elapsed = time.time() - start
    print(f"Offline query: {elapsed*1000:.1f}ms")
    return result

# Online pattern: key lookup (fast, pre-computed)
def online_query(cache, user_id):
    """Simulates an online store query -- direct key lookup."""
    start = time.time()
    result = cache.get(user_id, {})
    elapsed = time.time() - start
    print(f"Online query: {elapsed*1000:.4f}ms")
    return result
```

### Exercise 3: Map the Feature Store Architecture

**Goal:** Draw the architecture for your lab's feature store.

Create a text diagram (or use a tool like draw.io) that shows:

1. Where raw data comes from (PostgreSQL transactions table)
2. How features are computed (Python feature pipeline)
3. Where features are stored offline (PostgreSQL / Parquet)
4. Where features are stored online (Redis)
5. How models consume features (FastAPI feature server)
6. Where metadata lives (Feast registry)

```
Your architecture should look something like:

   [PostgreSQL: raw transactions]
            |
            v
   [Feature Pipeline (Python)]
            |
     +------+------+
     |             |
     v             v
 [Parquet]    [Redis]
 (offline)    (online)
     |             |
     v             v
 [Training]   [FastAPI Server]
 (batch)      (real-time)
```

---

## Starter Files

Check `lab/starter/` for:
- `raw_features.py` -- Skeleton for manual feature engineering
- `access_patterns.py` -- Online vs. offline exercise

## Solution Files

Check `lab/solution/` for:
- Complete implementations with comments
- Expected output for each exercise

> **Important:** Try the exercises yourself first. The learning happens when you struggle with the trade-offs.

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Computing features differently for training vs. serving | Model accuracy drops in production | Use a feature store to serve the same values everywhere |
| Storing features without timestamps | Cannot do point-in-time joins | Always include `event_timestamp` in feature data |
| Using the online store for training data generation | Slow, incomplete data | Use the offline store for batch/historical queries |
| Ignoring feature freshness | Model uses stale data | Set appropriate TTLs and materialization schedules |

---

## Self-Check Questions

1. What is training-serving skew and why is it dangerous?
2. Name three benefits of using a feature store over ad-hoc feature computation.
3. When would you use the offline store vs. the online store?
4. What is an "entity" in feature store terminology?
5. Why does materialization exist -- why not just read from the offline store at serving time?

---

## You Know You Have Completed This Module When...

- [ ] You can explain the training-serving skew problem to a colleague
- [ ] You understand the difference between online and offline stores
- [ ] You can list the core components of a feature store (entity, feature view, registry, etc.)
- [ ] You completed all three exercises
- [ ] Validation script passes: `bash modules/01-feature-engineering-basics/validation/validate.sh`

---

## Troubleshooting

**Issue: Python version too old**
```bash
python --version  # Need 3.10+
# Install with pyenv or download from python.org
```

**Issue: Cannot import pandas/numpy**
```bash
pip install -r requirements.txt
```

---

**Next: [Module 02 - Feast Setup and Configuration -->](../02-feast-setup/)**
