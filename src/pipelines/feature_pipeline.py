"""
Feature Store Lab - Feature Engineering Pipeline

This pipeline reads raw transaction data, computes aggregated features,
writes them to the offline store (Parquet files), and materializes
them to the online store (Redis) for real-time serving.

Usage:
    # Run the full pipeline
    python -m src.pipelines.feature_pipeline

    # Run individual stages
    python -m src.pipelines.feature_pipeline --stage generate
    python -m src.pipelines.feature_pipeline --stage compute
    python -m src.pipelines.feature_pipeline --stage materialize
"""

import argparse
import logging
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from feast import FeatureStore

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("feature_pipeline")

# Paths
DATA_RAW_DIR = os.getenv("DATA_RAW_DIR", "data/raw")
DATA_PROCESSED_DIR = os.getenv("DATA_PROCESSED_DIR", "data/processed")
FEAST_REPO_PATH = os.getenv("FEAST_REPO_PATH", "feature_repo")


# =============================================================================
# Stage 1: Generate Synthetic Raw Data
# In production, this would be replaced by reads from your data warehouse,
# event stream, or batch ETL output.
# =============================================================================

def generate_raw_data(
    n_customers: int = 1000,
    n_products: int = 200,
    n_transactions: int = 50000,
    seed: int = 42,
) -> None:
    """Generate synthetic e-commerce transaction data."""
    logger.info("Generating synthetic raw data...")
    np.random.seed(seed)

    os.makedirs(DATA_RAW_DIR, exist_ok=True)

    # --- Customer data ---
    customers = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n_customers)],
        "signup_date": pd.date_range(
            end=datetime.now(), periods=n_customers, freq="h"
        ).to_list(),
        "segment": np.random.choice(
            ["premium", "standard", "budget"], size=n_customers, p=[0.15, 0.55, 0.30]
        ),
    })
    customers.to_parquet(os.path.join(DATA_RAW_DIR, "customers.parquet"), index=False)
    logger.info(f"  Created {n_customers} customer records.")

    # --- Product data ---
    categories = ["electronics", "clothing", "food", "home", "sports", "books"]
    products = pd.DataFrame({
        "product_id": [f"P{i:05d}" for i in range(n_products)],
        "product_name": [f"Product_{i}" for i in range(n_products)],
        "category": np.random.choice(categories, size=n_products),
        "base_price": np.round(np.random.lognormal(3.5, 1.0, size=n_products), 2),
        "launch_date": pd.date_range(
            end=datetime.now(), periods=n_products, freq="3h"
        ).to_list(),
        "avg_rating": np.round(np.random.uniform(1.0, 5.0, size=n_products), 1),
        "total_reviews": np.random.poisson(50, size=n_products),
    })
    products.to_parquet(os.path.join(DATA_RAW_DIR, "products.parquet"), index=False)
    logger.info(f"  Created {n_products} product records.")

    # --- Transaction data ---
    transactions = pd.DataFrame({
        "transaction_id": [f"T{i:07d}" for i in range(n_transactions)],
        "customer_id": np.random.choice(customers["customer_id"], size=n_transactions),
        "product_id": np.random.choice(products["product_id"], size=n_transactions),
        "amount": np.round(np.random.lognormal(3.0, 1.2, size=n_transactions), 2),
        "timestamp": [
            datetime.now() - timedelta(days=np.random.exponential(15))
            for _ in range(n_transactions)
        ],
        "merchant_category": np.random.choice(categories, size=n_transactions),
    })
    transactions.to_parquet(
        os.path.join(DATA_RAW_DIR, "transactions.parquet"), index=False
    )
    logger.info(f"  Created {n_transactions} transaction records.")
    logger.info("Raw data generation complete.")


# =============================================================================
# Stage 2: Compute Features
# Transform raw data into feature vectors. This is where feature engineering
# logic lives -- windowed aggregations, ratios, flags, etc.
# =============================================================================

def compute_customer_features() -> pd.DataFrame:
    """Compute aggregated customer transaction features over rolling windows."""
    logger.info("Computing customer transaction features...")

    transactions = pd.read_parquet(os.path.join(DATA_RAW_DIR, "transactions.parquet"))
    customers = pd.read_parquet(os.path.join(DATA_RAW_DIR, "customers.parquet"))

    now = datetime.now()
    cutoff_30d = now - timedelta(days=30)
    cutoff_7d = now - timedelta(days=7)

    # 30-day window aggregations
    recent_txn = transactions[transactions["timestamp"] >= cutoff_30d]
    agg_30d = recent_txn.groupby("customer_id").agg(
        total_transactions_30d=("amount", "count"),
        total_spend_30d=("amount", "sum"),
        avg_transaction_amount_30d=("amount", "mean"),
        max_transaction_amount_30d=("amount", "max"),
        min_transaction_amount_30d=("amount", "min"),
        std_transaction_amount_30d=("amount", "std"),
        unique_merchants_30d=("merchant_category", "nunique"),
        last_transaction_date=("timestamp", "max"),
    ).reset_index()

    # 7-day window for trend features
    recent_7d = transactions[transactions["timestamp"] >= cutoff_7d]
    agg_7d = recent_7d.groupby("customer_id").agg(
        transaction_frequency_7d=("amount", "count"),
        total_spend_7d=("amount", "sum"),
    ).reset_index()

    # Merge windows
    features = agg_30d.merge(agg_7d, on="customer_id", how="left").fillna(0)

    # Derived features
    features["days_since_last_transaction"] = (
        now - features["last_transaction_date"]
    ).dt.days.fillna(999).astype(int)
    features["transaction_frequency_7d"] = features["transaction_frequency_7d"] / 7.0
    features["spend_trend_7d_vs_30d"] = (
        features["total_spend_7d"] / features["total_spend_30d"].clip(lower=0.01)
    )
    features["std_transaction_amount_30d"] = features[
        "std_transaction_amount_30d"
    ].fillna(0.0)

    # Add customer profile features
    features = features.merge(customers, on="customer_id", how="left")
    features["account_age_days"] = (now - features["signup_date"]).dt.days
    features["customer_segment"] = features["segment"]

    # Lifetime value estimate (simplified)
    all_agg = transactions.groupby("customer_id").agg(
        lifetime_value=("amount", "sum"),
        total_orders=("amount", "count"),
    ).reset_index()
    all_agg["avg_order_value"] = all_agg["lifetime_value"] / all_agg["total_orders"]

    # Find preferred category per customer
    pref_cat = (
        transactions.groupby(["customer_id", "merchant_category"])
        .size()
        .reset_index(name="cat_count")
        .sort_values("cat_count", ascending=False)
        .drop_duplicates("customer_id")
        .rename(columns={"merchant_category": "preferred_category"})[
            ["customer_id", "preferred_category"]
        ]
    )

    features = features.merge(all_agg, on="customer_id", how="left")
    features = features.merge(pref_cat, on="customer_id", how="left")

    # Add required timestamp columns
    features["event_timestamp"] = now
    features["created_timestamp"] = now

    # Select final columns
    customer_txn_cols = [
        "customer_id", "event_timestamp", "created_timestamp",
        "total_transactions_30d", "total_spend_30d", "avg_transaction_amount_30d",
        "max_transaction_amount_30d", "min_transaction_amount_30d",
        "std_transaction_amount_30d", "unique_merchants_30d",
        "days_since_last_transaction", "transaction_frequency_7d",
        "spend_trend_7d_vs_30d", "account_age_days", "customer_segment",
        "lifetime_value", "total_orders", "preferred_category", "avg_order_value",
    ]

    features = features[customer_txn_cols].round(4)
    logger.info(f"  Computed features for {len(features)} customers.")
    return features


def compute_product_features() -> pd.DataFrame:
    """Compute product-level features from catalog and transaction data."""
    logger.info("Computing product catalog features...")

    products = pd.read_parquet(os.path.join(DATA_RAW_DIR, "products.parquet"))
    transactions = pd.read_parquet(os.path.join(DATA_RAW_DIR, "transactions.parquet"))

    now = datetime.now()

    # Sales statistics per product
    product_stats = transactions.groupby("product_id").agg(
        total_sales=("amount", "sum"),
        num_purchases=("amount", "count"),
    ).reset_index()

    features = products.merge(product_stats, on="product_id", how="left").fillna(0)

    features["days_since_launch"] = (now - features["launch_date"]).dt.days
    features["price"] = features["base_price"]
    features["stock_level"] = np.random.randint(0, 500, size=len(features))
    features["price_percentile"] = features["price"].rank(pct=True).round(4)
    features["event_timestamp"] = now
    features["created_timestamp"] = now

    product_cols = [
        "product_id", "event_timestamp", "created_timestamp",
        "product_name", "category", "price", "avg_rating", "total_reviews",
        "days_since_launch", "stock_level", "price_percentile",
    ]
    features = features[product_cols]
    logger.info(f"  Computed features for {len(features)} products.")
    return features


def write_features_to_store(
    customer_features: pd.DataFrame,
    product_features: pd.DataFrame,
) -> None:
    """Write computed features to the offline store (Parquet files)."""
    logger.info("Writing features to offline store...")
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    customer_features.to_parquet(
        os.path.join(DATA_PROCESSED_DIR, "customer_transactions.parquet"), index=False
    )
    product_features.to_parquet(
        os.path.join(DATA_PROCESSED_DIR, "product_features.parquet"), index=False
    )
    logger.info("  Feature files written to disk.")


# =============================================================================
# Stage 3: Materialize Features to Online Store
# Push the latest feature values from offline store (Parquet) into the
# online store (Redis) so they can be served at low latency.
# =============================================================================

def materialize_features(days_back: int = 2) -> None:
    """Materialize features from the offline store to the online store (Redis)."""
    logger.info("Materializing features to online store...")
    store = FeatureStore(repo_path=FEAST_REPO_PATH)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    # Apply any pending feature definition changes
    store.apply(
        [
            # Import objects from feature_definitions module
        ]
    )

    store.materialize(
        start_date=start_date,
        end_date=end_date,
    )
    logger.info("  Materialization complete.")


# =============================================================================
# Stage 4: Generate Training Dataset
# Demonstrate point-in-time correct feature retrieval for model training.
# =============================================================================

def generate_training_data() -> pd.DataFrame:
    """
    Build a training dataset using point-in-time joins.
    This ensures no data leakage -- each row gets feature values as they
    existed at the time of the label event.
    """
    logger.info("Generating training dataset with point-in-time joins...")
    store = FeatureStore(repo_path=FEAST_REPO_PATH)

    # Simulated entity DataFrame with timestamps (label events)
    transactions = pd.read_parquet(os.path.join(DATA_RAW_DIR, "transactions.parquet"))
    entity_df = transactions[["customer_id", "timestamp"]].rename(
        columns={"timestamp": "event_timestamp"}
    ).head(5000)

    # Point-in-time join: get features as they existed at each event_timestamp
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "customer_transaction_features:total_transactions_30d",
            "customer_transaction_features:total_spend_30d",
            "customer_transaction_features:avg_transaction_amount_30d",
            "customer_profile_features:customer_segment",
            "customer_profile_features:lifetime_value",
        ],
    ).to_df()

    output_path = os.path.join(DATA_PROCESSED_DIR, "training_data.parquet")
    training_df.to_parquet(output_path, index=False)
    logger.info(f"  Training dataset saved: {output_path} ({len(training_df)} rows)")
    return training_df


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline")
    parser.add_argument(
        "--stage",
        choices=["generate", "compute", "materialize", "training", "all"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--n-customers", type=int, default=1000, help="Number of synthetic customers"
    )
    parser.add_argument(
        "--n-products", type=int, default=200, help="Number of synthetic products"
    )
    parser.add_argument(
        "--n-transactions", type=int, default=50000, help="Number of transactions"
    )
    args = parser.parse_args()

    stages = {
        "generate": lambda: generate_raw_data(
            args.n_customers, args.n_products, args.n_transactions
        ),
        "compute": lambda: write_features_to_store(
            compute_customer_features(), compute_product_features()
        ),
        "materialize": materialize_features,
        "training": generate_training_data,
    }

    if args.stage == "all":
        for name, fn in stages.items():
            logger.info(f"=== Running stage: {name} ===")
            fn()
    else:
        stages[args.stage]()

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
