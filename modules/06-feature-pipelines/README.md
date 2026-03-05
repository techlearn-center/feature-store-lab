# Module 06: Feature Pipelines -- Materialization, Scheduling, and Orchestration

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Intermediate |
| **Prerequisites** | Module 05 completed, features materialized |

---

## Learning Objectives

By the end of this module, you will be able to:

- Build end-to-end feature engineering pipelines
- Understand materialization strategies (full vs. incremental)
- Schedule feature pipelines for automated execution
- Handle late-arriving data and backfills
- Monitor pipeline health and freshness

---

## Concepts

### What Is a Feature Pipeline?

A feature pipeline is an automated workflow that:
1. **Reads** raw data from source systems (databases, event streams, files)
2. **Transforms** raw data into ML features (aggregations, ratios, encodings)
3. **Writes** features to the offline store (for training data generation)
4. **Materializes** features to the online store (for real-time serving)

```
+----------+     +-----------+     +-------------+     +--------------+
|  Raw     | --> | Transform | --> | Write to    | --> | Materialize  |
|  Data    |     | & Compute |     | Offline     |     | to Online    |
|  Sources |     | Features  |     | Store       |     | Store        |
+----------+     +-----------+     +-------------+     +--------------+
     ^                                                        |
     |                                                        v
  Scheduled                                             Redis (serving)
  (cron / Airflow / Prefect)
```

### Materialization Strategies

| Strategy | How It Works | When to Use |
|---|---|---|
| **Full** | Recomputes all features from scratch | Initial load, schema changes, backfills |
| **Incremental** | Only processes new data since last run | Daily/hourly updates, cost efficiency |
| **feast materialize** | Pushes latest offline values to online store | After offline store is updated |
| **feast materialize-incremental** | Only materializes new data | Regular scheduled updates |

```bash
# Full materialization (specify explicit time range)
feast materialize 2024-01-01T00:00:00 2024-12-31T23:59:59

# Incremental materialization (from last checkpoint)
feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)
```

### Pipeline Architecture

```
                    Feature Pipeline Architecture

  +------------------+     +------------------+     +------------------+
  | Stage 1:         |     | Stage 2:         |     | Stage 3:         |
  | Data Ingestion   |     | Feature Compute  |     | Store & Serve    |
  |                  |     |                  |     |                  |
  | - Read from DB   |     | - Aggregations   |     | - Write Parquet  |
  | - Read from S3   | --> | - Window funcs   | --> | - Materialize    |
  | - Read streams   |     | - Joins          |     | - Validate       |
  | - Validate raw   |     | - Encode cats    |     | - Alert on fail  |
  +------------------+     +------------------+     +------------------+
         |                        |                        |
         v                        v                        v
    [Data Quality]         [Feature Quality]         [Freshness Check]
```

---

## Hands-On Lab

### Exercise 1: Run the Feature Pipeline End-to-End

**Goal:** Execute the full feature pipeline from data generation to materialization.

```bash
# Run all stages sequentially
python -m src.pipelines.feature_pipeline --stage all

# Or run stages individually:

# Stage 1: Generate synthetic raw data
python -m src.pipelines.feature_pipeline --stage generate

# Stage 2: Compute features from raw data
python -m src.pipelines.feature_pipeline --stage compute

# Stage 3: Materialize to online store
python -m src.pipelines.feature_pipeline --stage materialize
```

```python
# Verify the pipeline output
import pandas as pd

# Check offline store data
customers = pd.read_parquet("data/processed/customer_transactions.parquet")
products = pd.read_parquet("data/processed/product_features.parquet")

print(f"Customer features: {customers.shape} ({customers['customer_id'].nunique()} unique)")
print(f"Product features: {products.shape} ({products['product_id'].nunique()} unique)")
print(f"\nCustomer feature columns: {customers.columns.tolist()}")
print(f"\nSample customer features:")
print(customers.head())
```

### Exercise 2: Build an Incremental Pipeline

**Goal:** Create a pipeline that only processes new data.

```python
# modules/06-feature-pipelines/lab/starter/incremental_pipeline.py
import pandas as pd
import os
from datetime import datetime

CHECKPOINT_FILE = "data/.pipeline_checkpoint"

def get_last_checkpoint() -> datetime:
    """Read the last successful pipeline run timestamp."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return datetime.fromisoformat(f.read().strip())
    return datetime.min

def save_checkpoint(ts: datetime) -> None:
    """Save the current pipeline run timestamp."""
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(ts.isoformat())

def run_incremental_pipeline():
    last_run = get_last_checkpoint()
    current_run = datetime.now()
    print(f"Processing data from {last_run} to {current_run}")

    # Read only new raw data since last checkpoint
    transactions = pd.read_parquet("data/raw/transactions.parquet")
    new_data = transactions[transactions["timestamp"] > last_run]

    if len(new_data) == 0:
        print("No new data to process. Skipping.")
        return

    print(f"Found {len(new_data)} new transactions to process")

    # Compute features for affected entities only
    affected_customers = new_data["customer_id"].unique()
    print(f"Recomputing features for {len(affected_customers)} customers")

    # ... (recompute features for affected customers)

    save_checkpoint(current_run)
    print(f"Checkpoint saved: {current_run}")

if __name__ == "__main__":
    run_incremental_pipeline()
```

### Exercise 3: Schedule Pipeline Execution

**Goal:** Set up automated scheduling for the feature pipeline.

```python
# modules/06-feature-pipelines/lab/starter/scheduled_pipeline.py
"""
Option 1: Simple cron-based scheduling (Linux/Mac)
Add to crontab: crontab -e

# Run feature pipeline every hour
0 * * * * cd /path/to/feature-store-lab && python -m src.pipelines.feature_pipeline --stage all >> /var/log/feature_pipeline.log 2>&1

# Run materialization every 15 minutes
*/15 * * * * cd /path/to/feature-store-lab && python -m src.pipelines.feature_pipeline --stage materialize >> /var/log/materialization.log 2>&1
"""

"""
Option 2: Python-based scheduler using APScheduler
"""
from apscheduler.schedulers.blocking import BlockingScheduler
from src.pipelines.feature_pipeline import (
    generate_raw_data,
    compute_customer_features,
    compute_product_features,
    write_features_to_store,
    materialize_features,
)

scheduler = BlockingScheduler()

@scheduler.scheduled_job("interval", hours=1, id="feature_compute")
def hourly_feature_compute():
    """Recompute features from raw data every hour."""
    customer_features = compute_customer_features()
    product_features = compute_product_features()
    write_features_to_store(customer_features, product_features)
    print("Feature computation complete")

@scheduler.scheduled_job("interval", minutes=15, id="materialize")
def frequent_materialize():
    """Push latest features to online store every 15 minutes."""
    materialize_features(days_back=1)
    print("Materialization complete")

if __name__ == "__main__":
    print("Starting feature pipeline scheduler...")
    scheduler.start()
```

```python
"""
Option 3: Airflow DAG (production-grade)
"""
# dags/feature_pipeline_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "ml-platform",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "feature_pipeline",
    default_args=default_args,
    schedule_interval="0 * * * *",  # Every hour
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["feature-store", "ml"],
) as dag:

    generate = PythonOperator(
        task_id="generate_raw_data",
        python_callable=generate_raw_data,
    )

    compute = PythonOperator(
        task_id="compute_features",
        python_callable=lambda: write_features_to_store(
            compute_customer_features(), compute_product_features()
        ),
    )

    materialize = PythonOperator(
        task_id="materialize_to_online",
        python_callable=materialize_features,
    )

    generate >> compute >> materialize
```

### Exercise 4: Handle Backfills

**Goal:** Reprocess historical data when feature logic changes.

```python
# modules/06-feature-pipelines/lab/starter/backfill.py
from feast import FeatureStore
from datetime import datetime, timedelta

def backfill_features(days_back: int = 90):
    """
    Backfill features when:
    - A new feature is added to an existing view
    - Feature computation logic changes
    - Data quality issues are discovered and fixed
    """
    store = FeatureStore(repo_path="feature_repo")

    end = datetime.now()
    start = end - timedelta(days=days_back)

    print(f"Backfilling features from {start} to {end}")
    print("This will reprocess ALL feature views for the time range.")

    # Step 1: Recompute features (this would run with the new logic)
    # python -m src.pipelines.feature_pipeline --stage compute

    # Step 2: Full materialization over the backfill window
    store.materialize(
        start_date=start,
        end_date=end,
    )

    print("Backfill complete!")

if __name__ == "__main__":
    backfill_features(days_back=90)
```

### Exercise 5: Monitor Pipeline Freshness

**Goal:** Track when features were last updated and alert on staleness.

```python
# modules/06-feature-pipelines/lab/starter/freshness_monitor.py
import pandas as pd
from datetime import datetime, timedelta

def check_feature_freshness(max_age_hours: int = 2):
    """Check that features are not stale."""
    issues = []

    # Check customer features
    df = pd.read_parquet("data/processed/customer_transactions.parquet")
    latest = df["event_timestamp"].max()
    age = datetime.now() - latest
    age_hours = age.total_seconds() / 3600

    print(f"Customer features - Last updated: {latest} (age: {age_hours:.1f}h)")
    if age_hours > max_age_hours:
        issues.append(f"Customer features are {age_hours:.1f}h old (max: {max_age_hours}h)")

    # Check product features
    df = pd.read_parquet("data/processed/product_features.parquet")
    latest = df["event_timestamp"].max()
    age = datetime.now() - latest
    age_hours = age.total_seconds() / 3600

    print(f"Product features  - Last updated: {latest} (age: {age_hours:.1f}h)")
    if age_hours > max_age_hours:
        issues.append(f"Product features are {age_hours:.1f}h old (max: {max_age_hours}h)")

    if issues:
        print(f"\nSTALE FEATURES DETECTED:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print("\nAll features are fresh.")
    return True

if __name__ == "__main__":
    check_feature_freshness()
```

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Running full materialization when incremental suffices | Slow, expensive pipeline runs | Use `materialize-incremental` for regular updates |
| No checkpoint tracking | Duplicate or missed data processing | Implement checkpoint files or use orchestrator state |
| No freshness monitoring | Stale features silently degrade model | Set up alerts on feature age |
| Not handling late-arriving data | Missing features for recent events | Use a buffer window in your processing cutoff |
| Backfilling without testing | Corrupted features in production | Test backfill on a staging environment first |

---

## Self-Check Questions

1. What is the difference between `materialize` and `materialize-incremental`?
2. Why would you need to backfill features?
3. How would you handle a feature pipeline that fails midway?
4. What is the trade-off between pipeline frequency and cost?
5. How would you schedule different feature views at different frequencies?

---

## You Know You Have Completed This Module When...

- [ ] You can run the feature pipeline end-to-end
- [ ] You understand full vs. incremental materialization
- [ ] You have built or configured a scheduling mechanism
- [ ] You have implemented a freshness monitoring check
- [ ] Validation script passes: `bash modules/06-feature-pipelines/validation/validate.sh`

---

## Troubleshooting

**Issue: Pipeline runs but features are not updated in Redis**
```bash
# Check that feast apply was run after any definition changes
cd feature_repo && feast apply
# Then re-run materialization
feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)
```

**Issue: Incremental pipeline misses data**
```python
# Add a buffer to your checkpoint to catch late-arriving data
buffer = timedelta(hours=1)
effective_start = last_checkpoint - buffer
```

---

**Next: [Module 07 - Data Quality -->](../07-data-quality/)**
