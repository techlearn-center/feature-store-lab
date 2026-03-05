# Module 08: Feature Discovery and Reuse -- Registry, Team Workflows, and Governance

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Advanced |
| **Prerequisites** | Module 07 completed |

---

## Learning Objectives

By the end of this module, you will be able to:

- Use the Feast registry as a feature catalog
- Tag and document features for discoverability
- Implement feature reuse patterns across multiple ML models
- Design team workflows for feature governance
- Build a simple feature search and discovery tool

---

## Concepts

### The Feature Reuse Problem

Without a feature registry, ML teams repeatedly build the same features:

```
Team A (Fraud Detection):
  "avg_transaction_amount_30d" -- built in notebook, pandas

Team B (Credit Scoring):
  "mean_txn_amt_last_month" -- built in Spark job

Team C (Recommendations):
  "user_avg_spend_30days" -- built in SQL view

All three compute the SAME thing with different:
  - Names, schemas, computation logic, freshness, quality checks
```

A feature registry eliminates this waste by making features **discoverable** and **reusable**.

### Feature Registry as a Catalog

The Feast registry stores metadata about every feature definition:

```
+---------------------------------------------------------------------+
|                       Feature Registry                               |
|                                                                      |
|  Entity: customer                                                    |
|    join_keys: [customer_id]                                          |
|    description: "A platform customer"                                |
|                                                                      |
|  FeatureView: customer_transaction_features                          |
|    owner: data-engineering                                           |
|    tier: critical                                                    |
|    features:                                                         |
|      - total_transactions_30d (Int64)                                |
|      - total_spend_30d (Float64)                                     |
|      - avg_transaction_amount_30d (Float64)                          |
|    source: customer_transactions.parquet                             |
|    ttl: 1 day                                                        |
|    last_updated: 2024-03-01T10:00:00                                |
|                                                                      |
|  FeatureService: fraud_detection                                     |
|    model: fraud_detection_v2                                         |
|    views: [customer_transaction_features, customer_profile_features] |
+---------------------------------------------------------------------+
```

### Feature Governance Workflow

```
         Feature Lifecycle

  [Propose] --> [Review] --> [Approve] --> [Deploy] --> [Monitor] --> [Deprecate]
     |             |             |             |             |             |
  PR with       Code review    Merge to     feast apply   Freshness    Remove from
  feature def   + data review  main branch  + materialize  checks       registry
```

---

## Hands-On Lab

### Exercise 1: Explore the Feature Registry

**Goal:** Use the Feast SDK and CLI to discover available features.

```bash
# CLI-based discovery
cd feature_repo

# List all entities
feast entities list

# List all feature views with details
feast feature-views list

# Describe a feature view in detail
feast feature-views describe customer_transaction_features

# List all feature services
feast feature-services list
```

```python
# modules/08-feature-sharing/lab/starter/explore_registry.py
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo")

print("=" * 60)
print("FEATURE STORE CATALOG")
print("=" * 60)

# List all entities
print("\n--- ENTITIES ---")
for entity in store.list_entities():
    print(f"  {entity.name}")
    print(f"    Join keys: {entity.join_keys}")
    print(f"    Description: {entity.description}")

# List all feature views with metadata
print("\n--- FEATURE VIEWS ---")
for fv in store.list_feature_views():
    print(f"\n  {fv.name}")
    print(f"    Entities: {[e.name for e in fv.entity_columns]}")
    print(f"    Features ({len(fv.schema)}):")
    for field in fv.schema:
        print(f"      - {field.name} ({field.dtype})")
    print(f"    TTL: {fv.ttl}")
    print(f"    Online: {fv.online}")
    print(f"    Tags: {dict(fv.tags)}")

# List on-demand feature views
print("\n--- ON-DEMAND FEATURE VIEWS ---")
for odfv in store.list_on_demand_feature_views():
    print(f"  {odfv.name}")
    print(f"    Features: {[f.name for f in odfv.schema]}")

# List feature services
print("\n--- FEATURE SERVICES ---")
for fs in store.list_feature_services():
    print(f"\n  {fs.name}")
    print(f"    Tags: {dict(fs.tags)}")
    print(f"    Description: {fs.description}")
```

### Exercise 2: Build a Feature Search Tool

**Goal:** Create a utility that helps data scientists find features by keyword.

```python
# modules/08-feature-sharing/lab/starter/feature_search.py
from feast import FeatureStore
from typing import Optional

class FeatureCatalog:
    """A searchable catalog of all features in the store."""

    def __init__(self, repo_path: str = "feature_repo"):
        self.store = FeatureStore(repo_path=repo_path)
        self._index = self._build_index()

    def _build_index(self) -> list[dict]:
        """Build a searchable index of all features."""
        index = []

        for fv in self.store.list_feature_views():
            for field in fv.schema:
                index.append({
                    "feature_name": field.name,
                    "feature_view": fv.name,
                    "dtype": str(field.dtype),
                    "entity": fv.entities[0] if fv.entities else "N/A",
                    "ttl": str(fv.ttl),
                    "online": fv.online,
                    "tags": dict(fv.tags),
                    "full_name": f"{fv.name}:{field.name}",
                })

        for odfv in self.store.list_on_demand_feature_views():
            for field in odfv.schema:
                index.append({
                    "feature_name": field.name,
                    "feature_view": odfv.name,
                    "dtype": str(field.dtype),
                    "entity": "on-demand",
                    "ttl": "N/A",
                    "online": True,
                    "tags": {},
                    "full_name": f"{odfv.name}:{field.name}",
                })

        return index

    def search(self, keyword: str) -> list[dict]:
        """Search features by keyword (matches name, view, or tags)."""
        keyword = keyword.lower()
        results = []
        for entry in self._index:
            searchable = (
                entry["feature_name"].lower()
                + " " + entry["feature_view"].lower()
                + " " + " ".join(entry["tags"].values())
            )
            if keyword in searchable:
                results.append(entry)
        return results

    def list_by_entity(self, entity_name: str) -> list[dict]:
        """List all features for a given entity."""
        return [e for e in self._index if e["entity"] == entity_name]

    def list_by_team(self, team: str) -> list[dict]:
        """List all features owned by a team."""
        return [
            e for e in self._index
            if e["tags"].get("team", "") == team
        ]

    def get_feature_reference(self, feature_name: str) -> Optional[str]:
        """Get the full Feast reference string for a feature name."""
        for entry in self._index:
            if entry["feature_name"] == feature_name:
                return entry["full_name"]
        return None


# Demo usage
catalog = FeatureCatalog()

print("--- Search: 'transaction' ---")
for f in catalog.search("transaction"):
    print(f"  {f['full_name']} ({f['dtype']})")

print("\n--- Search: 'spend' ---")
for f in catalog.search("spend"):
    print(f"  {f['full_name']} ({f['dtype']})")

print("\n--- Features for entity: 'customer' ---")
for f in catalog.list_by_entity("customer"):
    print(f"  {f['full_name']}")

print("\n--- Features by team: 'ml-platform' ---")
for f in catalog.list_by_team("ml-platform"):
    print(f"  {f['full_name']} (owner: {f['tags'].get('owner', 'N/A')})")
```

### Exercise 3: Implement Feature Reuse Across Models

**Goal:** Show how two different models can share the same features.

```python
# modules/08-feature-sharing/lab/starter/feature_reuse.py
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo")

# Model 1: Fraud Detection -- uses transaction + profile + risk features
fraud_features = store.get_feature_service("fraud_detection")
print(f"Fraud model uses: {fraud_features.name}")

# Model 2: Recommendation -- uses profile + product features
rec_features = store.get_feature_service("recommendation_engine")
print(f"Recommendation model uses: {rec_features.name}")

# Both models share customer_profile_features!
# The feature view is computed ONCE and served to BOTH models.
# No duplication of computation or storage.

# To add a new model that reuses existing features:
# Simply create a new FeatureService in feature_definitions.py
"""
churn_prediction_service = FeatureService(
    name="churn_prediction",
    features=[
        customer_transaction_features,  # reused from fraud model
        customer_profile_features,      # reused from both models
    ],
    tags={"model": "churn_v1", "team": "growth"},
)
"""

print("\nBenefits of feature reuse:")
print("  1. customer_profile_features computed ONCE, used by 2+ models")
print("  2. All models get the same feature values (no skew)")
print("  3. New models get instant access to existing features")
print("  4. Changes to feature logic propagate to all consumers")
```

### Exercise 4: Feature Tagging and Documentation Standards

**Goal:** Establish conventions for feature documentation using tags.

```python
# modules/08-feature-sharing/lab/starter/tagging_conventions.py
"""
Recommended Feature Tagging Convention:

Tags make features discoverable and governable. Standardize these across your org.
"""

REQUIRED_TAGS = {
    "team": "Team that owns and maintains this feature view",
    "owner": "Individual or group responsible for data quality",
    "tier": "critical | standard | experimental",
}

OPTIONAL_TAGS = {
    "description": "Human-readable description of the feature view",
    "data_source": "Where the raw data comes from",
    "refresh_frequency": "How often features are recomputed",
    "pii": "yes | no -- whether this contains personal data",
    "deprecated": "Set to 'true' when planning to remove",
}

# Example of a well-tagged feature view:
"""
customer_transaction_features = FeatureView(
    name="customer_transaction_features",
    ...
    tags={
        # Required
        "team": "ml-platform",
        "owner": "data-engineering",
        "tier": "critical",
        # Optional
        "description": "Rolling 30-day transaction aggregates per customer",
        "data_source": "postgres.public.transactions",
        "refresh_frequency": "hourly",
        "pii": "no",
    },
)
"""

# Validate that all feature views have required tags
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo")

print("Feature View Tag Audit:")
for fv in store.list_feature_views():
    tags = dict(fv.tags)
    missing = [tag for tag in REQUIRED_TAGS if tag not in tags]
    status = "PASS" if not missing else "FAIL"
    print(f"  [{status}] {fv.name}")
    if missing:
        print(f"    Missing required tags: {missing}")
    else:
        print(f"    Team: {tags['team']}, Owner: {tags['owner']}, Tier: {tags['tier']}")
```

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| No feature documentation | Teams rebuild existing features | Enforce required tags on all feature views |
| No naming conventions | Confusion about what features represent | Adopt `<entity>_<description>_<window>` naming |
| No ownership tracking | Nobody maintains stale features | Require `team` and `owner` tags |
| Building features for one model only | Wasted compute, inconsistent values | Design features for reuse via feature services |
| No deprecation process | Old features accumulate | Add `deprecated` tag and migration timeline |

---

## Self-Check Questions

1. How does a feature registry prevent duplicate feature computation across teams?
2. What metadata should every feature view include for discoverability?
3. How would you handle a breaking change to a feature that multiple models use?
4. What is the difference between a feature view and a feature service in terms of reuse?
5. How would you deprecate a feature that is used by multiple downstream models?

---

## You Know You Have Completed This Module When...

- [ ] You can explore the feature registry via CLI and SDK
- [ ] You have built a feature search/discovery tool
- [ ] You understand how feature services enable reuse across models
- [ ] You have established tagging conventions for your feature views
- [ ] Validation script passes: `bash modules/08-feature-sharing/validation/validate.sh`

---

## Troubleshooting

**Issue: Registry out of sync**
```bash
cd feature_repo
feast apply  # Re-sync definitions to registry
```

**Issue: Cannot find a feature you know exists**
```python
# Check that you are pointing to the right repo_path
store = FeatureStore(repo_path="feature_repo")
print(store.project)  # Should match your feature_store.yaml project name
```

---

**Next: [Module 09 - On-Demand Features -->](../09-real-time-features/)**
