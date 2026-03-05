# Module 07: Data Quality -- Great Expectations Integration and Feature Validation

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Advanced |
| **Prerequisites** | Module 06 completed, Great Expectations installed |

---

## Learning Objectives

By the end of this module, you will be able to:

- Integrate Great Expectations with your feature pipeline
- Define expectations (validation rules) for feature data
- Build automated data quality checks that run before materialization
- Detect feature drift and schema violations
- Set up alerting for data quality failures

---

## Concepts

### Why Data Quality Matters for Features

Bad features produce bad models. Common data quality issues:

| Issue | Impact | Example |
|---|---|---|
| **Null values** | Model receives missing inputs | Customer has no transactions |
| **Schema changes** | Pipeline breaks silently | Column renamed in upstream table |
| **Value drift** | Model accuracy degrades | Average spend shifts due to pricing change |
| **Duplicates** | Inflated aggregations | Same transaction counted twice |
| **Out-of-range values** | Garbage predictions | Negative transaction amounts |

### Great Expectations Overview

Great Expectations (GE) is a Python library for data validation. It lets you define **expectations** (rules) about your data and automatically verify them.

```
Feature Pipeline with Data Quality:

  Raw Data --> [Validate Raw] --> Transform --> [Validate Features] --> Store
                   |                                  |
                   v                                  v
              Pass? Continue                    Pass? Materialize
              Fail? Alert + Stop                Fail? Alert + Rollback
```

### Core GE Concepts

| Concept | Description |
|---|---|
| **Expectation** | A single validation rule (e.g., "column X should not be null") |
| **Expectation Suite** | A collection of expectations for a dataset |
| **Validator** | Runs expectations against actual data |
| **Checkpoint** | An automated validation job |
| **Data Docs** | Auto-generated HTML reports of validation results |

---

## Hands-On Lab

### Prerequisites Check

```bash
pip install great-expectations
great_expectations --version

# Ensure feature data exists
ls data/processed/customer_transactions.parquet
```

### Exercise 1: Define Expectations for Raw Data

**Goal:** Create validation rules for the raw transaction data that arrives before feature computation.

```python
# modules/07-data-quality/lab/starter/raw_data_validation.py
import great_expectations as gx
import pandas as pd

# Load raw data
transactions = pd.read_parquet("data/raw/transactions.parquet")

# Create a GE context and data source
context = gx.get_context()

# Create a validator from the DataFrame
validator = context.sources.pandas_default.read_dataframe(transactions)

# Define expectations for raw transaction data

# 1. Schema expectations
validator.expect_column_to_exist("transaction_id")
validator.expect_column_to_exist("customer_id")
validator.expect_column_to_exist("amount")
validator.expect_column_to_exist("timestamp")

# 2. Null checks
validator.expect_column_values_to_not_be_null("transaction_id")
validator.expect_column_values_to_not_be_null("customer_id")
validator.expect_column_values_to_not_be_null("amount")

# 3. Value range checks
validator.expect_column_values_to_be_between(
    "amount", min_value=0.01, max_value=100000.0
)

# 4. Uniqueness checks
validator.expect_column_values_to_be_unique("transaction_id")

# 5. Pattern checks
validator.expect_column_values_to_match_regex(
    "customer_id", regex=r"^C\d{5}$"
)

# 6. Cardinality checks
validator.expect_column_distinct_values_to_be_in_set(
    "merchant_category",
    value_set=["electronics", "clothing", "food", "home", "sports", "books"],
)

# Run validation
results = validator.validate()
print(f"Validation success: {results.success}")
print(f"Results: {results.statistics}")

# Print failed expectations
for result in results.results:
    if not result.success:
        print(f"FAILED: {result.expectation_config.expectation_type}")
        print(f"  Details: {result.result}")
```

### Exercise 2: Validate Computed Features

**Goal:** Create validation rules for features after computation.

```python
# modules/07-data-quality/lab/starter/feature_validation.py
import great_expectations as gx
import pandas as pd

# Load computed features
features = pd.read_parquet("data/processed/customer_transactions.parquet")

context = gx.get_context()
validator = context.sources.pandas_default.read_dataframe(features)

# Feature-specific expectations

# 1. No negative counts
validator.expect_column_values_to_be_between(
    "total_transactions_30d", min_value=0
)

# 2. Spend should be non-negative
validator.expect_column_values_to_be_between(
    "total_spend_30d", min_value=0.0
)

# 3. Average should be between min and max
# (logical consistency check)
validator.expect_column_pair_values_a_to_be_greater_than_b(
    "max_transaction_amount_30d",
    "avg_transaction_amount_30d",
    or_equal=True,
)

# 4. Standard deviation should be non-negative
validator.expect_column_values_to_be_between(
    "std_transaction_amount_30d", min_value=0.0
)

# 5. Null rate should be below threshold
validator.expect_column_values_to_not_be_null(
    "customer_id"
)
validator.expect_column_values_to_not_be_null(
    "event_timestamp"
)

# 6. Feature freshness check
validator.expect_column_max_to_be_between(
    "event_timestamp",
    min_value=pd.Timestamp.now() - pd.Timedelta(hours=24),
    max_value=pd.Timestamp.now() + pd.Timedelta(hours=1),
)

# 7. Distribution checks (detect drift)
validator.expect_column_mean_to_be_between(
    "avg_transaction_amount_30d",
    min_value=1.0,
    max_value=10000.0,
)

validator.expect_column_stdev_to_be_between(
    "total_spend_30d",
    min_value=0.0,
    max_value=100000.0,
)

results = validator.validate()
print(f"\nFeature Validation: {'PASSED' if results.success else 'FAILED'}")
print(f"Statistics: {results.statistics}")
```

### Exercise 3: Integrate Validation into the Feature Pipeline

**Goal:** Add quality gates to the feature pipeline so bad data never reaches the online store.

```python
# modules/07-data-quality/lab/starter/validated_pipeline.py
import pandas as pd
import great_expectations as gx
import logging

logger = logging.getLogger("validated_pipeline")

class FeatureQualityGate:
    """Validates feature data before it is written to the store."""

    def __init__(self):
        self.context = gx.get_context()

    def validate_customer_features(self, df: pd.DataFrame) -> bool:
        """Validate customer feature DataFrame. Returns True if valid."""
        validator = self.context.sources.pandas_default.read_dataframe(df)

        # Critical checks (pipeline stops on failure)
        validator.expect_column_values_to_not_be_null("customer_id")
        validator.expect_column_values_to_not_be_null("event_timestamp")
        validator.expect_column_values_to_be_between(
            "total_transactions_30d", min_value=0
        )
        validator.expect_column_values_to_be_between(
            "total_spend_30d", min_value=0.0
        )

        results = validator.validate()

        if not results.success:
            logger.error("Feature validation FAILED!")
            for r in results.results:
                if not r.success:
                    logger.error(
                        f"  {r.expectation_config.expectation_type}: "
                        f"{r.result.get('unexpected_percent', 'N/A')}% unexpected"
                    )
            return False

        logger.info("Feature validation PASSED")
        return True

    def validate_product_features(self, df: pd.DataFrame) -> bool:
        """Validate product feature DataFrame."""
        validator = self.context.sources.pandas_default.read_dataframe(df)

        validator.expect_column_values_to_not_be_null("product_id")
        validator.expect_column_values_to_be_between("price", min_value=0.0)
        validator.expect_column_values_to_be_between(
            "avg_rating", min_value=0.0, max_value=5.0
        )

        results = validator.validate()
        return results.success


def run_validated_pipeline():
    """Feature pipeline with quality gates."""
    from src.pipelines.feature_pipeline import (
        compute_customer_features,
        compute_product_features,
        write_features_to_store,
        materialize_features,
    )

    gate = FeatureQualityGate()

    # Compute features
    customer_features = compute_customer_features()
    product_features = compute_product_features()

    # Quality gate: validate before writing
    customer_ok = gate.validate_customer_features(customer_features)
    product_ok = gate.validate_product_features(product_features)

    if not customer_ok or not product_ok:
        logger.error("PIPELINE HALTED: Data quality checks failed.")
        logger.error("Features NOT written to store. Investigate and fix.")
        return False

    # Write to offline store
    write_features_to_store(customer_features, product_features)

    # Materialize to online store
    materialize_features()

    logger.info("Validated pipeline completed successfully.")
    return True

if __name__ == "__main__":
    run_validated_pipeline()
```

### Exercise 4: Detect Feature Drift

**Goal:** Compare current feature distributions against a baseline to detect drift.

```python
# modules/07-data-quality/lab/starter/drift_detection.py
import pandas as pd
import numpy as np

def compute_drift_report(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    numeric_cols: list[str],
    drift_threshold: float = 0.3,
) -> dict:
    """
    Simple drift detection using Population Stability Index (PSI).

    PSI < 0.1: No significant drift
    PSI 0.1 - 0.25: Moderate drift (monitor)
    PSI > 0.25: Significant drift (investigate)
    """
    report = {}

    for col in numeric_cols:
        base_vals = baseline[col].dropna()
        curr_vals = current[col].dropna()

        # Create bins from baseline distribution
        bins = np.histogram_bin_edges(base_vals, bins=10)
        base_hist, _ = np.histogram(base_vals, bins=bins, density=True)
        curr_hist, _ = np.histogram(curr_vals, bins=bins, density=True)

        # Avoid division by zero
        base_hist = np.clip(base_hist, 1e-6, None)
        curr_hist = np.clip(curr_hist, 1e-6, None)

        # Calculate PSI
        psi = np.sum((curr_hist - base_hist) * np.log(curr_hist / base_hist))

        status = "OK" if psi < 0.1 else "MONITOR" if psi < 0.25 else "ALERT"
        report[col] = {"psi": round(psi, 4), "status": status}

        if psi > drift_threshold:
            print(f"DRIFT ALERT: {col} PSI={psi:.4f} (threshold={drift_threshold})")

    return report

# Example usage
baseline = pd.read_parquet("data/processed/customer_transactions.parquet")
current = baseline.copy()  # In practice, this would be the latest data

# Simulate drift in one feature
current["total_spend_30d"] = current["total_spend_30d"] * 1.5 + 100

drift_report = compute_drift_report(
    baseline, current,
    numeric_cols=["total_spend_30d", "avg_transaction_amount_30d", "total_transactions_30d"],
)

print("\nDrift Report:")
for col, info in drift_report.items():
    print(f"  {col}: PSI={info['psi']}, Status={info['status']}")
```

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Only validating at the end | Bad data corrupts features | Validate raw data AND computed features |
| Too strict expectations | Pipeline fails on acceptable variation | Use statistical thresholds, not exact values |
| No baseline for drift detection | Cannot detect gradual shifts | Save a baseline snapshot when pipeline is known good |
| Validation without alerting | Failures go unnoticed | Integrate with Slack, PagerDuty, or email |
| Blocking on warnings | Pipeline never completes | Separate critical (blocking) from warning (logging) checks |

---

## Self-Check Questions

1. What is the difference between a data quality check and a drift detection check?
2. Why should validation happen before materialization rather than after?
3. What is the Population Stability Index (PSI) and how do you interpret it?
4. How would you handle a feature that is expected to have some null values?
5. What is the trade-off between strict validation and pipeline reliability?

---

## You Know You Have Completed This Module When...

- [ ] You have defined expectations for both raw data and computed features
- [ ] Validation runs as part of the feature pipeline
- [ ] You can detect feature drift using PSI or a similar metric
- [ ] You understand the difference between blocking and warning checks
- [ ] Validation script passes: `bash modules/07-data-quality/validation/validate.sh`

---

## Troubleshooting

**Issue: Great Expectations version compatibility**
```bash
pip install "great-expectations>=0.18.0,<1.0.0"
```

**Issue: Validation passes but model performance drops**
```python
# Your expectations may not cover the right dimensions
# Add distribution-level checks (mean, stdev, percentiles)
# Add cross-column consistency checks
```

---

**Next: [Module 08 - Feature Discovery and Reuse -->](../08-feature-sharing/)**
