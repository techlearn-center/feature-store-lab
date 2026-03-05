"""
Feature Store Lab - Feast Feature Definitions

This module defines all entities, data sources, feature views, and on-demand
feature transformations for the feature store. These definitions serve as the
central contract between data producers and ML model consumers.

Architecture:
    Raw Data (PostgreSQL) -> Feature Views -> Online Store (Redis)
                                          -> Offline Store (PostgreSQL)
                                          -> On-Demand Transforms
"""

from datetime import timedelta

import pandas as pd
from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    PushSource,
    RequestSource,
)
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Float64, Int64, String, UnixTimestamp


# =============================================================================
# ENTITIES
# Entities are the primary keys used to look up features. Each entity
# represents a real-world object (user, driver, product) that features
# describe.
# =============================================================================

customer_entity = Entity(
    name="customer",
    join_keys=["customer_id"],
    description="A customer who interacts with our platform.",
)

product_entity = Entity(
    name="product",
    join_keys=["product_id"],
    description="A product available in the catalog.",
)


# =============================================================================
# DATA SOURCES
# Data sources tell Feast where to read raw feature data from. In production,
# these typically point to data warehouse tables, streaming topics, or files.
# =============================================================================

# Batch source: historical customer transaction data stored as Parquet
customer_transactions_source = FileSource(
    name="customer_transactions",
    path="data/processed/customer_transactions.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
    description="Customer transaction aggregates computed by the feature pipeline.",
)

# Batch source: product catalog features
product_features_source = FileSource(
    name="product_features",
    path="data/processed/product_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
    description="Product-level features including price stats and category info.",
)

# Push source: real-time customer activity events
customer_activity_push_source = PushSource(
    name="customer_activity_push",
    batch_source=customer_transactions_source,
    description="Real-time push source for streaming customer activity updates.",
)

# Request source: data provided at inference time (not stored in any store)
transaction_request_source = RequestSource(
    name="transaction_request",
    schema=[
        Field(name="transaction_amount", dtype=Float64),
        Field(name="merchant_category", dtype=String),
    ],
)


# =============================================================================
# FEATURE VIEWS
# Feature views define a set of features, their data source, and how long
# they stay valid (TTL). Each view maps to a logical group of features.
# =============================================================================

customer_transaction_features = FeatureView(
    name="customer_transaction_features",
    entities=[customer_entity],
    ttl=timedelta(days=1),
    schema=[
        Field(name="total_transactions_30d", dtype=Int64),
        Field(name="total_spend_30d", dtype=Float64),
        Field(name="avg_transaction_amount_30d", dtype=Float64),
        Field(name="max_transaction_amount_30d", dtype=Float64),
        Field(name="min_transaction_amount_30d", dtype=Float64),
        Field(name="std_transaction_amount_30d", dtype=Float64),
        Field(name="unique_merchants_30d", dtype=Int64),
        Field(name="days_since_last_transaction", dtype=Int64),
        Field(name="transaction_frequency_7d", dtype=Float64),
        Field(name="spend_trend_7d_vs_30d", dtype=Float64),
    ],
    source=customer_transactions_source,
    online=True,
    tags={
        "team": "ml-platform",
        "owner": "data-engineering",
        "tier": "critical",
    },
    description="Aggregated transaction features for each customer over rolling windows.",
)

customer_profile_features = FeatureView(
    name="customer_profile_features",
    entities=[customer_entity],
    ttl=timedelta(days=7),
    schema=[
        Field(name="account_age_days", dtype=Int64),
        Field(name="customer_segment", dtype=String),
        Field(name="lifetime_value", dtype=Float64),
        Field(name="total_orders", dtype=Int64),
        Field(name="preferred_category", dtype=String),
        Field(name="avg_order_value", dtype=Float64),
    ],
    source=customer_transactions_source,
    online=True,
    tags={
        "team": "ml-platform",
        "owner": "data-engineering",
        "tier": "standard",
    },
    description="Slowly changing customer profile and segmentation features.",
)

product_catalog_features = FeatureView(
    name="product_catalog_features",
    entities=[product_entity],
    ttl=timedelta(days=1),
    schema=[
        Field(name="product_name", dtype=String),
        Field(name="category", dtype=String),
        Field(name="price", dtype=Float64),
        Field(name="avg_rating", dtype=Float64),
        Field(name="total_reviews", dtype=Int64),
        Field(name="days_since_launch", dtype=Int64),
        Field(name="stock_level", dtype=Int64),
        Field(name="price_percentile", dtype=Float64),
    ],
    source=product_features_source,
    online=True,
    tags={
        "team": "ml-platform",
        "owner": "product-team",
        "tier": "standard",
    },
    description="Product catalog features including price, ratings, and inventory.",
)


# =============================================================================
# ON-DEMAND FEATURE VIEWS
# On-demand features are computed at request time using data from other
# feature views combined with request-time data. These are useful for
# features that depend on the current transaction context.
# =============================================================================

@on_demand_feature_view(
    sources=[customer_transaction_features, transaction_request_source],
    schema=[
        Field(name="transaction_amount_zscore", dtype=Float64),
        Field(name="is_high_value_transaction", dtype=Int64),
        Field(name="amount_to_avg_ratio", dtype=Float64),
        Field(name="is_above_spend_pattern", dtype=Int64),
    ],
    description="Real-time fraud signals computed from transaction context and history.",
)
def transaction_risk_features(inputs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute risk-related features at inference time by combining the current
    transaction details with the customer's historical spending patterns.

    These features cannot be pre-computed because they depend on the
    specific transaction being evaluated.
    """
    df = pd.DataFrame()

    avg = inputs["avg_transaction_amount_30d"]
    std = inputs["std_transaction_amount_30d"]
    amount = inputs["transaction_amount"]

    # Z-score: how many standard deviations from the customer's mean
    df["transaction_amount_zscore"] = (amount - avg) / std.clip(lower=1.0)

    # Binary flag: is this transaction above the 95th percentile?
    threshold = avg + 2 * std.clip(lower=1.0)
    df["is_high_value_transaction"] = (amount > threshold).astype(int)

    # Ratio of current amount to the customer's average
    df["amount_to_avg_ratio"] = amount / avg.clip(lower=0.01)

    # Is the customer spending more than their recent pattern?
    df["is_above_spend_pattern"] = (
        amount > inputs["max_transaction_amount_30d"]
    ).astype(int)

    return df


# =============================================================================
# FEATURE SERVICES
# Feature services group feature views into logical units that ML models
# consume. Each model should have its own feature service definition.
# =============================================================================

fraud_detection_service = FeatureService(
    name="fraud_detection",
    features=[
        customer_transaction_features,
        customer_profile_features,
        transaction_risk_features,
    ],
    tags={"model": "fraud_detection_v2", "team": "ml-platform"},
    description="Features for the real-time fraud detection model.",
)

recommendation_service = FeatureService(
    name="recommendation_engine",
    features=[
        customer_profile_features,
        product_catalog_features,
    ],
    tags={"model": "recommendation_v1", "team": "ml-platform"},
    description="Features for the product recommendation model.",
)
