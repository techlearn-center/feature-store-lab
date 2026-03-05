# Module 04: Offline Feature Retrieval -- Point-in-Time Joins and Training Data

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Intermediate |
| **Prerequisites** | Module 03 completed, feature definitions applied |

---

## Learning Objectives

By the end of this module, you will be able to:

- Explain what point-in-time joins are and why they prevent data leakage
- Retrieve historical features for training dataset generation
- Build entity DataFrames with proper timestamps
- Use `get_historical_features()` to create training data
- Debug common issues with offline feature retrieval

---

## Concepts

### Why Point-in-Time Joins Matter

When building training datasets, the most dangerous mistake is **data leakage** -- using information from the future to predict the past. Point-in-time joins ensure that for each training example at time T, you only get feature values that existed **at or before** time T.

```
Timeline:    Jan 1    Jan 15    Feb 1    Feb 15    Mar 1
             |        |         |        |         |
Features:    v=10     v=25      v=30     v=45      v=50
             |        |         |        |         |
Event:       -------- -------- [Label]   |         |
                                  ^      |         |
                                  |      |         |
                           PIT join returns v=30   |
                           (NOT v=45 or v=50)      |
```

Without point-in-time correctness, a naive join would grab the latest feature value (v=50), leaking future information into the training data.

### How Feast Handles Point-in-Time Joins

```python
# You provide an "entity DataFrame" with timestamps for each training example
entity_df = pd.DataFrame({
    "customer_id": ["C001", "C002", "C001"],
    "event_timestamp": [
        datetime(2024, 2, 1),   # Get C001's features as of Feb 1
        datetime(2024, 2, 15),  # Get C002's features as of Feb 15
        datetime(2024, 3, 1),   # Get C001's features as of Mar 1
    ],
    "label": [0, 1, 0],
})

# Feast joins features as they existed at each event_timestamp
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "customer_transaction_features:total_spend_30d",
        "customer_transaction_features:avg_transaction_amount_30d",
    ],
).to_df()
```

**Under the hood**, Feast executes a temporal join:
1. For each row in `entity_df`, find all feature rows for that entity
2. Filter to rows where `feature_timestamp <= event_timestamp`
3. Pick the most recent one (closest to but not after the event)
4. If the most recent value is older than the TTL, return `null`

### Entity DataFrame Design

The entity DataFrame is the "spine" of your training data:

```
Entity DataFrame (spine):
+-------------+------------------+-------+
| customer_id | event_timestamp  | label |
+-------------+------------------+-------+
| C001        | 2024-01-15 10:00 |   0   |  <-- get features as of this moment
| C002        | 2024-01-20 14:30 |   1   |
| C001        | 2024-02-01 09:00 |   0   |  <-- same customer, different time
+-------------+------------------+-------+

After PIT join:
+-------------+------------------+-------+----------------+------------------+
| customer_id | event_timestamp  | label | total_spend_30d| avg_txn_amount   |
+-------------+------------------+-------+----------------+------------------+
| C001        | 2024-01-15 10:00 |   0   | 1500.00        | 45.50            |
| C002        | 2024-01-20 14:30 |   1   | 3200.00        | 120.75           |
| C001        | 2024-02-01 09:00 |   0   | 1800.00        | 52.30            |
+-------------+------------------+-------+----------------+------------------+
```

---

## Hands-On Lab

### Prerequisites Check

```bash
# Ensure data exists
ls data/processed/customer_transactions.parquet
ls data/processed/product_features.parquet

# If not, run the feature pipeline first
python -m src.pipelines.feature_pipeline --stage generate
python -m src.pipelines.feature_pipeline --stage compute
```

### Exercise 1: Build an Entity DataFrame

**Goal:** Create a properly structured entity DataFrame for training data generation.

```python
# modules/04-offline-store/lab/starter/entity_dataframe.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load raw transactions to get realistic entity-timestamp pairs
transactions = pd.read_parquet("data/raw/transactions.parquet")

# Sample transactions as "label events"
sample = transactions.sample(n=2000, random_state=42)

# Build the entity DataFrame
entity_df = pd.DataFrame({
    "customer_id": sample["customer_id"].values,
    "event_timestamp": sample["timestamp"].values,
})

# Add a synthetic label
entity_df["is_fraud"] = np.random.choice(
    [0, 1], size=len(entity_df), p=[0.95, 0.05]
)

print(f"Entity DataFrame shape: {entity_df.shape}")
print(f"Unique customers: {entity_df['customer_id'].nunique()}")
print(f"Time range: {entity_df['event_timestamp'].min()} to "
      f"{entity_df['event_timestamp'].max()}")
print(entity_df.head(10))
```

### Exercise 2: Retrieve Historical Features

**Goal:** Use Feast to perform point-in-time correct feature retrieval.

```python
# modules/04-offline-store/lab/starter/historical_retrieval.py
from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path="feature_repo")
entity_df = pd.read_parquet("data/processed/entity_df.parquet")

# Retrieve historical features -- Feast handles the PIT join
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "customer_transaction_features:total_transactions_30d",
        "customer_transaction_features:total_spend_30d",
        "customer_transaction_features:avg_transaction_amount_30d",
        "customer_transaction_features:max_transaction_amount_30d",
        "customer_transaction_features:std_transaction_amount_30d",
        "customer_transaction_features:unique_merchants_30d",
        "customer_profile_features:customer_segment",
        "customer_profile_features:lifetime_value",
        "customer_profile_features:account_age_days",
    ],
).to_df()

print(f"Training DataFrame shape: {training_df.shape}")
print(f"Columns: {training_df.columns.tolist()}")
print(f"\nNull counts:\n{training_df.isnull().sum()}")
print(f"\nSample rows:")
print(training_df.head())

training_df.to_parquet("data/processed/training_data.parquet", index=False)
```

### Exercise 3: Using Feature Services for Retrieval

**Goal:** Retrieve features using a named feature service.

```python
# modules/04-offline-store/lab/starter/service_retrieval.py
from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path="feature_repo")
entity_df = pd.read_parquet("data/processed/entity_df.parquet")

# Retrieve using a feature service -- cleaner and version-controlled
fraud_service = store.get_feature_service("fraud_detection")

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=fraud_service,
).to_df()

print(f"Features from fraud_detection service: {training_df.columns.tolist()}")
print(f"Shape: {training_df.shape}")
```

### Exercise 4: Train a Simple Model with Feature Store Data

**Goal:** Complete the end-to-end workflow by training a model.

```python
# modules/04-offline-store/lab/starter/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

df = pd.read_parquet("data/processed/training_data.parquet")

feature_cols = [
    "total_transactions_30d", "total_spend_30d",
    "avg_transaction_amount_30d", "max_transaction_amount_30d",
    "std_transaction_amount_30d", "unique_merchants_30d",
    "lifetime_value", "account_age_days",
]

le = LabelEncoder()
df["customer_segment_encoded"] = le.fit_transform(
    df["customer_segment"].fillna("unknown")
)
feature_cols.append("customer_segment_encoded")

X = df[feature_cols].fillna(0)
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

importances = pd.Series(model.feature_importances_, index=feature_cols)
print("\nFeature Importance:")
print(importances.sort_values(ascending=False))
```

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Entity DataFrame missing `event_timestamp` | `KeyError` or wrong join results | Always include an `event_timestamp` column |
| Timestamp timezone mismatch | Features return as `null` | Ensure all timestamps are UTC or timezone-naive consistently |
| Requesting features from wrong entity | Empty or null results | Check that entity keys in your DataFrame match the feature view's entity |
| Not running `feast apply` before retrieval | Feature view not found | Always `feast apply` after changing definitions |
| Label leakage via feature timestamps | Suspiciously high accuracy | Verify features are from before the label event |

---

## Self-Check Questions

1. What would happen if you used a regular SQL join instead of a point-in-time join?
2. Why does the entity DataFrame need an `event_timestamp` column?
3. What happens when a feature's timestamp is older than the TTL?
4. How does using a feature service differ from listing features individually?
5. How would you verify that no data leakage occurred in your training dataset?

---

## You Know You Have Completed This Module When...

- [ ] You can explain point-in-time joins and why they prevent data leakage
- [ ] You have built an entity DataFrame with proper timestamps
- [ ] You have retrieved historical features using `get_historical_features()`
- [ ] You have trained a model using Feast-generated training data
- [ ] Validation script passes: `bash modules/04-offline-store/validation/validate.sh`

---

## Troubleshooting

**Issue: `get_historical_features()` returns all nulls**
```python
# Check that your data timestamps overlap with the entity DataFrame
features_df = pd.read_parquet("data/processed/customer_transactions.parquet")
print(f"Feature data range: {features_df['event_timestamp'].min()} to "
      f"{features_df['event_timestamp'].max()}")
print(f"Entity DF range: {entity_df['event_timestamp'].min()} to "
      f"{entity_df['event_timestamp'].max()}")
```

**Issue: Very slow historical retrieval**
```bash
# For large datasets, use PostgreSQL offline store instead of file-based
# Update offline_store section in feature_store.yaml
```

---

**Next: [Module 05 - Online Feature Serving -->](../05-online-store/)**
