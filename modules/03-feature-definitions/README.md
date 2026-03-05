# Module 03: Feature Definitions -- Entities, Feature Views, and Data Sources

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Intermediate |
| **Prerequisites** | Module 02 completed, Feast installed |

---

## Learning Objectives

By the end of this module, you will be able to:

- Define Feast entities with appropriate join keys
- Create data sources that point to your feature data
- Build feature views with typed schemas and TTLs
- Group feature views into feature services for ML models
- Understand how Feast resolves feature lookups at query time

---

## Concepts

### Entities: The Primary Keys of Your Feature Store

An **entity** is the real-world object that features describe. Every feature lookup starts with an entity key.

```python
from feast import Entity

# A customer entity -- features will be looked up by customer_id
customer = Entity(
    name="customer",
    join_keys=["customer_id"],       # Column name used for joins
    description="A platform customer",
)

# A driver entity -- for a ride-sharing use case
driver = Entity(
    name="driver",
    join_keys=["driver_id"],
    description="A driver in the fleet",
)
```

**Design rule:** Entities should map to the grain of your ML model's input. If your model predicts per-user, your entity is `user`. If it predicts per-user-per-item, you might need a composite key.

### Data Sources: Where Features Come From

Data sources tell Feast where to read raw feature data. Common types:

```python
from feast import FileSource, PostgreSQLSource, PushSource, RequestSource
from feast.types import Float64, String
from feast import Field

# File source -- reads from Parquet files (good for dev/testing)
file_source = FileSource(
    name="transactions_file",
    path="data/processed/customer_transactions.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Push source -- for real-time streaming data
push_source = PushSource(
    name="realtime_events",
    batch_source=file_source,  # Fallback for historical queries
)

# Request source -- data provided at inference time, not stored
request_source = RequestSource(
    name="inference_context",
    schema=[
        Field(name="current_amount", dtype=Float64),
        Field(name="device_type", dtype=String),
    ],
)
```

### Feature Views: Logical Groups of Features

A **feature view** maps a set of features to a data source. Think of it as a "table" in your feature store.

```python
from datetime import timedelta
from feast import FeatureView, Field
from feast.types import Float64, Int64, String

customer_features = FeatureView(
    name="customer_transaction_features",
    entities=[customer],                    # Which entity this view describes
    ttl=timedelta(days=1),                  # How long features stay valid
    schema=[
        Field(name="total_transactions_30d", dtype=Int64),
        Field(name="total_spend_30d", dtype=Float64),
        Field(name="avg_transaction_amount_30d", dtype=Float64),
    ],
    source=file_source,                     # Where data comes from
    online=True,                            # Materialize to online store?
    tags={"team": "ml-platform", "tier": "critical"},
    description="Customer transaction aggregates over 30-day windows.",
)
```

**Key parameters explained:**

| Parameter | Purpose | Guidance |
|---|---|---|
| `entities` | Links view to entity join keys | Must match entity `join_keys` columns in your data |
| `ttl` | Time-to-live for online features | Set based on how frequently features change |
| `schema` | Typed column definitions | Must match the columns in your data source exactly |
| `source` | Where Feast reads data | Must have `timestamp_field` for point-in-time joins |
| `online` | Whether to materialize to online store | Set `False` for training-only features |
| `tags` | Arbitrary key-value metadata | Use for ownership, tier, team, documentation |

### Feature Services: Bundles for ML Models

A **feature service** groups multiple feature views into a single unit consumed by one ML model. This creates a clear contract between the feature store and the model.

```python
from feast import FeatureService

fraud_detection_service = FeatureService(
    name="fraud_detection",
    features=[
        customer_features,          # Transaction aggregates
        customer_profile_features,  # Profile/segmentation
        transaction_risk_features,  # On-demand risk signals
    ],
    tags={"model": "fraud_detection_v2"},
    description="All features needed by the fraud detection model.",
)
```

**Why feature services matter:**
- They version-control which features a model uses
- They enable independent feature view evolution without breaking consumers
- They make it easy to reproduce a model's exact feature set

---

## Hands-On Lab

### Exercise 1: Define Entities for an E-Commerce Platform

**Goal:** Create entity definitions for customers, products, and merchants.

```python
# modules/03-feature-definitions/lab/starter/entities.py
from feast import Entity

# 1. Customer entity
customer = Entity(
    name="customer",
    join_keys=["customer_id"],
    description="A customer who makes purchases on the platform.",
)

# 2. Product entity
product = Entity(
    name="product",
    join_keys=["product_id"],
    description="A product in the catalog.",
)

# 3. Merchant entity (your addition)
merchant = Entity(
    name="merchant",
    join_keys=["merchant_id"],
    description="A merchant selling products on the platform.",
)

print(f"Defined entities: customer ({customer.join_keys}), "
      f"product ({product.join_keys}), merchant ({merchant.join_keys})")
```

### Exercise 2: Create Feature Views with Proper Schemas

**Goal:** Build feature views with typed schemas, TTLs, and tags.

```python
# modules/03-feature-definitions/lab/starter/feature_views.py
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Float64, Int64, String

# Data source
customer_source = FileSource(
    name="customer_transactions",
    path="data/processed/customer_transactions.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

customer = Entity(name="customer", join_keys=["customer_id"])

# Create a feature view with at least 5 features
customer_transaction_features = FeatureView(
    name="customer_transaction_features",
    entities=[customer],
    ttl=timedelta(days=1),
    schema=[
        Field(name="total_transactions_30d", dtype=Int64),
        Field(name="total_spend_30d", dtype=Float64),
        Field(name="avg_transaction_amount_30d", dtype=Float64),
        Field(name="max_transaction_amount_30d", dtype=Float64),
        Field(name="unique_merchants_30d", dtype=Int64),
    ],
    source=customer_source,
    online=True,
    tags={
        "team": "ml-platform",
        "owner": "data-engineering",
        "tier": "critical",
    },
)

# Inspect the feature view
print(f"Feature view: {customer_transaction_features.name}")
print(f"  Features: {[f.name for f in customer_transaction_features.schema]}")
print(f"  TTL: {customer_transaction_features.ttl}")
print(f"  Online: {customer_transaction_features.online}")
```

### Exercise 3: Build a Feature Service

**Goal:** Bundle feature views into a service for a specific ML model.

```python
# modules/03-feature-definitions/lab/starter/feature_services.py
from feast import FeatureService

# Feature service for fraud detection model
fraud_service = FeatureService(
    name="fraud_detection",
    features=[
        customer_transaction_features,
        customer_profile_features,
    ],
    tags={"model": "fraud_v2", "team": "ml-platform"},
)

# Feature service for recommendation model
recommendation_service = FeatureService(
    name="recommendations",
    features=[
        customer_profile_features,
        product_catalog_features,
    ],
    tags={"model": "rec_v1", "team": "ml-platform"},
)

print(f"Fraud service: {fraud_service.name}")
print(f"Recommendation service: {recommendation_service.name}")
```

### Exercise 4: Apply and Verify Definitions

**Goal:** Register your definitions with Feast and verify them.

```bash
cd feature_repo

# Apply all definitions
feast apply

# Verify entities
feast entities list

# Verify feature views
feast feature-views list

# Describe a specific feature view in detail
feast feature-views describe customer_transaction_features

# List feature services
feast feature-services list
```

```python
# Programmatic verification
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo")

for entity in store.list_entities():
    print(f"Entity: {entity.name}, keys: {entity.join_keys}")

for fv in store.list_feature_views():
    print(f"FeatureView: {fv.name}, features: {[f.name for f in fv.schema]}")

for fs in store.list_feature_services():
    print(f"FeatureService: {fs.name}")
```

---

## Feature Type Reference

| Python Type | Feast Type | Use Case |
|---|---|---|
| `int` | `Int64` | Counts, IDs, flags |
| `float` | `Float64` | Amounts, ratios, scores |
| `str` | `String` | Categories, names, labels |
| `bool` | `Bool` | Binary flags |
| `datetime` | `UnixTimestamp` | Timestamps |
| `bytes` | `Bytes` | Embeddings, serialized objects |
| `list[float]` | `Array(Float64)` | Feature vectors, embeddings |

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Schema column names don't match data source | `KeyError` during materialization | Ensure column names in `schema` exactly match your Parquet/DB columns |
| Missing `timestamp_field` in data source | Point-in-time joins fail | Every data source needs an event timestamp column |
| TTL too short | Features show as `null` in online store | Set TTL longer than your materialization interval |
| TTL too long | Stale features served to models | Match TTL to your data freshness requirements |
| Entity join key mismatch | Empty results on feature retrieval | The `join_keys` name must match the column in your data |

---

## Self-Check Questions

1. What is the relationship between an entity and a feature view?
2. Why do feature views require a `ttl` parameter?
3. When would you set `online=False` on a feature view?
4. What is the purpose of a feature service vs. requesting features individually?
5. How does Feast use the `timestamp_field` in data sources?

---

## You Know You Have Completed This Module When...

- [ ] You have defined at least 2 entities with meaningful join keys
- [ ] You have created at least 2 feature views with typed schemas
- [ ] You have built at least 1 feature service
- [ ] `feast apply` completes successfully
- [ ] `feast feature-views list` shows your definitions
- [ ] Validation script passes: `bash modules/03-feature-definitions/validation/validate.sh`

---

## Troubleshooting

**Issue: `feast apply` says "schema mismatch"**
```bash
# Check your Parquet file columns match the schema
python -c "import pandas as pd; print(pd.read_parquet('data/processed/customer_transactions.parquet').columns.tolist())"
```

**Issue: Entity not found during feature retrieval**
```bash
feast entities list
# Check your join_keys match the column name in your data
```

---

**Next: [Module 04 - Offline Feature Retrieval -->](../04-offline-store/)**
