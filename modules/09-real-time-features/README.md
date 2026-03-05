# Module 09: On-Demand Features -- Real-Time Computed Features

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Advanced |
| **Prerequisites** | Module 08 completed |

---

## Learning Objectives

By the end of this module, you will be able to:

- Explain when on-demand features are needed vs. pre-computed features
- Define on-demand feature views in Feast
- Combine stored features with request-time data
- Use push sources for streaming feature updates
- Benchmark on-demand feature computation latency

---

## Concepts

### Pre-Computed vs. On-Demand Features

| Aspect | Pre-Computed | On-Demand |
|---|---|---|
| **When computed** | During pipeline (batch) | At request time (real-time) |
| **Storage** | Stored in online/offline store | Not stored, computed per request |
| **Latency** | Fast lookup (sub-ms) | Compute time added (1-10ms) |
| **Use case** | Features that don't change with context | Features that depend on the current request |
| **Example** | Customer avg spend over 30 days | Z-score of current transaction vs. customer avg |

### Why On-Demand Features?

Some features **cannot be pre-computed** because they depend on information only available at inference time.

```
Pre-computed (stored in Redis):
  - avg_transaction_amount_30d = $45.00    (does not change per request)
  - total_transactions_30d = 23            (does not change per request)

On-demand (computed at inference time):
  - transaction_amount_zscore              (depends on current transaction!)
    = (current_amount - avg_amount) / std_amount
    = ($500 - $45) / $30 = 15.17    <-- this changes every request

  - is_high_value_transaction              (depends on current transaction!)
    = 1 if current_amount > avg + 2*std else 0
```

### On-Demand Feature View Architecture

```
Inference Request
  {customer_id: "C001", amount: 500.00, category: "electronics"}
         |
         v
  +-------------------+
  | Feature Server    |
  |                   |
  | 1. Fetch stored   |<---- Redis: avg_amount=45, std_amount=30
  |    features       |
  |                   |
  | 2. Combine with   |<---- Request: amount=500, category=electronics
  |    request data   |
  |                   |
  | 3. Compute        |----> zscore = (500-45)/30 = 15.17
  |    on-demand      |----> is_high_value = 1
  |    features       |----> ratio = 500/45 = 11.11
  |                   |
  | 4. Return all     |----> {avg_amount: 45, zscore: 15.17, ...}
  +-------------------+
```

---

## Hands-On Lab

### Exercise 1: Define an On-Demand Feature View

**Goal:** Create on-demand features that combine stored data with request-time data.

```python
# modules/09-real-time-features/lab/starter/on_demand_features.py
import pandas as pd
from feast import Field, RequestSource
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float64, Int64, String

# Request source: data the caller provides at inference time
transaction_request = RequestSource(
    name="transaction_request",
    schema=[
        Field(name="transaction_amount", dtype=Float64),
        Field(name="merchant_category", dtype=String),
    ],
)

# On-demand feature view: combines stored + request-time data
@on_demand_feature_view(
    sources=[
        customer_transaction_features,  # From the online store
        transaction_request,             # From the API request
    ],
    schema=[
        Field(name="transaction_amount_zscore", dtype=Float64),
        Field(name="is_high_value_transaction", dtype=Int64),
        Field(name="amount_to_avg_ratio", dtype=Float64),
        Field(name="is_above_spend_pattern", dtype=Int64),
    ],
)
def transaction_risk_features(inputs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute real-time risk signals from the transaction context.

    inputs DataFrame contains columns from BOTH sources:
      - From online store: avg_transaction_amount_30d, std_transaction_amount_30d, ...
      - From request: transaction_amount, merchant_category
    """
    df = pd.DataFrame()

    avg = inputs["avg_transaction_amount_30d"]
    std = inputs["std_transaction_amount_30d"].clip(lower=1.0)
    amount = inputs["transaction_amount"]

    # Z-score: how unusual is this transaction for this customer?
    df["transaction_amount_zscore"] = (amount - avg) / std

    # High value flag: is this above the customer's 95th percentile?
    threshold = avg + 2 * std
    df["is_high_value_transaction"] = (amount > threshold).astype(int)

    # Ratio to average: how many times the customer's average?
    df["amount_to_avg_ratio"] = amount / avg.clip(lower=0.01)

    # Above historical max?
    df["is_above_spend_pattern"] = (
        amount > inputs["max_transaction_amount_30d"]
    ).astype(int)

    return df
```

### Exercise 2: Fetch On-Demand Features via the Online Store

**Goal:** Retrieve on-demand features alongside stored features.

```python
# modules/09-real-time-features/lab/starter/fetch_on_demand.py
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo")

# The entity row includes BOTH the entity key AND the request-time data
entity_rows = [
    {
        "customer_id": "C00001",
        "transaction_amount": 500.00,      # Request-time
        "merchant_category": "electronics", # Request-time
    }
]

# Feast automatically:
# 1. Fetches stored features from Redis (avg, std, max, etc.)
# 2. Passes request-time data to the on-demand transform
# 3. Returns all features together
result = store.get_online_features(
    features=[
        # Stored features (from Redis)
        "customer_transaction_features:avg_transaction_amount_30d",
        "customer_transaction_features:std_transaction_amount_30d",
        "customer_transaction_features:max_transaction_amount_30d",
        "customer_transaction_features:total_transactions_30d",
        # On-demand features (computed now)
        "transaction_risk_features:transaction_amount_zscore",
        "transaction_risk_features:is_high_value_transaction",
        "transaction_risk_features:amount_to_avg_ratio",
        "transaction_risk_features:is_above_spend_pattern",
    ],
    entity_rows=entity_rows,
).to_dict()

print("=== Feature Results ===")
for key, values in result.items():
    print(f"  {key}: {values[0]}")

# Example output:
# customer_id: C00001
# avg_transaction_amount_30d: 45.50
# std_transaction_amount_30d: 30.20
# max_transaction_amount_30d: 250.00
# total_transactions_30d: 23
# transaction_amount_zscore: 15.05    <-- computed on-demand
# is_high_value_transaction: 1        <-- computed on-demand
# amount_to_avg_ratio: 10.99          <-- computed on-demand
# is_above_spend_pattern: 1           <-- computed on-demand
```

### Exercise 3: Use Push Sources for Streaming Updates

**Goal:** Push real-time data into the feature store from a streaming source.

```python
# modules/09-real-time-features/lab/starter/push_features.py
"""
Push sources allow you to write features to both the online and offline
store from a streaming pipeline (e.g., Kafka consumer, Flink job).
"""
import pandas as pd
from datetime import datetime
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo")

# Simulate a real-time event arriving from a stream processor
event_df = pd.DataFrame({
    "customer_id": ["C00001"],
    "total_transactions_30d": [24],       # Updated count
    "total_spend_30d": [1550.00],         # Updated spend
    "avg_transaction_amount_30d": [64.58], # Updated average
    "max_transaction_amount_30d": [500.0], # Updated max
    "min_transaction_amount_30d": [5.0],
    "std_transaction_amount_30d": [95.0],
    "unique_merchants_30d": [8],
    "days_since_last_transaction": [0],    # Just transacted
    "transaction_frequency_7d": [4.0],
    "spend_trend_7d_vs_30d": [0.35],
    "event_timestamp": [datetime.now()],
    "created_timestamp": [datetime.now()],
})

# Push to online store (for real-time serving)
store.push("customer_activity_push", event_df, to=PushMode.ONLINE)
print("Pushed to online store")

# Push to both online and offline (for consistency)
store.push("customer_activity_push", event_df, to=PushMode.ONLINE_AND_OFFLINE)
print("Pushed to online and offline stores")
```

### Exercise 4: Benchmark On-Demand Feature Latency

**Goal:** Measure how much latency on-demand computation adds.

```python
# modules/09-real-time-features/lab/starter/on_demand_benchmark.py
import time
import statistics
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo")

stored_only = [
    "customer_transaction_features:total_transactions_30d",
    "customer_transaction_features:avg_transaction_amount_30d",
    "customer_transaction_features:std_transaction_amount_30d",
]

with_on_demand = stored_only + [
    "transaction_risk_features:transaction_amount_zscore",
    "transaction_risk_features:is_high_value_transaction",
    "transaction_risk_features:amount_to_avg_ratio",
]

entity_row_stored = [{"customer_id": "C00001"}]
entity_row_on_demand = [{
    "customer_id": "C00001",
    "transaction_amount": 500.0,
    "merchant_category": "electronics",
}]

# Warm up
for _ in range(5):
    store.get_online_features(features=stored_only, entity_rows=entity_row_stored)
    store.get_online_features(features=with_on_demand, entity_rows=entity_row_on_demand)

# Benchmark stored-only
stored_latencies = []
for _ in range(100):
    start = time.time()
    store.get_online_features(features=stored_only, entity_rows=entity_row_stored)
    stored_latencies.append((time.time() - start) * 1000)

# Benchmark with on-demand
od_latencies = []
for _ in range(100):
    start = time.time()
    store.get_online_features(features=with_on_demand, entity_rows=entity_row_on_demand)
    od_latencies.append((time.time() - start) * 1000)

print("=== Stored Features Only ===")
print(f"  p50: {statistics.median(stored_latencies):.2f}ms")
print(f"  p95: {sorted(stored_latencies)[94]:.2f}ms")

print("\n=== With On-Demand Features ===")
print(f"  p50: {statistics.median(od_latencies):.2f}ms")
print(f"  p95: {sorted(od_latencies)[94]:.2f}ms")

overhead = statistics.median(od_latencies) - statistics.median(stored_latencies)
print(f"\n  On-demand overhead (p50): {overhead:.2f}ms")
```

---

## When to Use On-Demand vs. Pre-Computed Features

| Scenario | Approach | Reason |
|---|---|---|
| Customer's average spend over 30 days | Pre-computed | Does not change per request |
| Z-score of current transaction amount | On-demand | Depends on current amount |
| Product's average rating | Pre-computed | Slowly changing |
| Time since customer's last login | On-demand | Changes every second |
| User-item affinity score | Pre-computed | Can be batched |
| Price difference from competitor | On-demand | External data at request time |

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Heavy computation in on-demand views | High serving latency | Move complex logic to batch pipeline |
| Missing request source fields in entity_rows | `KeyError` at inference | Include all RequestSource fields in entity_rows |
| Using on-demand for features that could be pre-computed | Unnecessary latency | Pre-compute anything that does not depend on request context |
| Not testing on-demand views with edge cases | NaN or infinity in results | Test with zero, negative, and extreme values |

---

## Self-Check Questions

1. What makes a feature on-demand vs. pre-computable?
2. How does an on-demand feature view get its input data?
3. What is a RequestSource and when do you need one?
4. What is the latency trade-off of on-demand features?
5. How do push sources differ from batch materialization?

---

## You Know You Have Completed This Module When...

- [ ] You have defined an on-demand feature view with request-time inputs
- [ ] You can fetch on-demand features via `get_online_features()`
- [ ] You understand push sources for streaming updates
- [ ] You have benchmarked the latency overhead of on-demand computation
- [ ] Validation script passes: `bash modules/09-real-time-features/validation/validate.sh`

---

## Troubleshooting

**Issue: On-demand feature returns NaN**
```python
# Check for division by zero in your transform
# Use .clip(lower=1.0) on denominators
# Test with: inputs["std_transaction_amount_30d"] = 0
```

**Issue: KeyError for request source field**
```python
# Ensure your entity_rows include ALL fields from the RequestSource
# Check: transaction_amount AND merchant_category must be present
```

---

**Next: [Module 10 - Production Deployment -->](../10-production-feature-store/)**
