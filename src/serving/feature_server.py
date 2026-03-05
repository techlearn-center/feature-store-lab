"""
Feature Store Lab - Real-Time Feature Serving API

A FastAPI application that serves features from the Feast online store
for real-time ML inference. This is the bridge between your feature store
and your ML models running in production.

Usage:
    uvicorn src.serving.feature_server:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health              - Health check
    POST /features/customer   - Get customer features by ID
    POST /features/product    - Get product features by ID
    POST /features/fraud      - Get fraud detection features (with on-demand)
    POST /features/batch      - Batch feature retrieval for multiple entities
    GET  /features/metadata   - List available feature services and views
"""

import logging
import os
import time
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("feature_server")

# ---- Lazy-load the Feast store to avoid import-time failures ----
_store = None
FEAST_REPO_PATH = os.getenv("FEAST_REPO_PATH", "feature_repo")


def get_store():
    """Lazily initialize the Feast FeatureStore connection."""
    global _store
    if _store is None:
        from feast import FeatureStore
        _store = FeatureStore(repo_path=FEAST_REPO_PATH)
        logger.info(f"Feast store initialized from {FEAST_REPO_PATH}")
    return _store


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Feature Store Lab - Feature Server",
    description="Real-time feature serving API backed by Feast + Redis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Request / Response Models
# =============================================================================

class CustomerFeatureRequest(BaseModel):
    customer_ids: list[str] = Field(
        ..., description="List of customer IDs to fetch features for", min_length=1
    )

class ProductFeatureRequest(BaseModel):
    product_ids: list[str] = Field(
        ..., description="List of product IDs to fetch features for", min_length=1
    )

class FraudFeatureRequest(BaseModel):
    customer_id: str = Field(..., description="Customer performing the transaction")
    transaction_amount: float = Field(..., description="Transaction amount in dollars")
    merchant_category: str = Field(..., description="Merchant category code")

class BatchFeatureRequest(BaseModel):
    feature_service: str = Field(
        ..., description="Name of the feature service to use"
    )
    entities: dict[str, list[Any]] = Field(
        ..., description="Entity key-value pairs, e.g. {'customer_id': ['C001', 'C002']}"
    )

class FeatureResponse(BaseModel):
    features: list[dict[str, Any]]
    metadata: dict[str, Any]


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health")
async def health_check():
    """Verify the feature server and its dependencies are operational."""
    try:
        store = get_store()
        return {
            "status": "healthy",
            "feast_project": store.project,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unhealthy: {str(e)}")


@app.post("/features/customer", response_model=FeatureResponse)
async def get_customer_features(request: CustomerFeatureRequest):
    """
    Retrieve customer features from the online store.

    Returns transaction aggregates and profile features for the requested
    customer IDs. Latency target: < 10 ms for single lookups.
    """
    start = time.time()
    store = get_store()

    try:
        entity_rows = [{"customer_id": cid} for cid in request.customer_ids]

        features = store.get_online_features(
            features=[
                "customer_transaction_features:total_transactions_30d",
                "customer_transaction_features:total_spend_30d",
                "customer_transaction_features:avg_transaction_amount_30d",
                "customer_transaction_features:unique_merchants_30d",
                "customer_transaction_features:days_since_last_transaction",
                "customer_profile_features:customer_segment",
                "customer_profile_features:lifetime_value",
                "customer_profile_features:total_orders",
            ],
            entity_rows=entity_rows,
        ).to_dict()

        result = _online_dict_to_rows(features)
        latency_ms = (time.time() - start) * 1000

        return FeatureResponse(
            features=result,
            metadata={
                "latency_ms": round(latency_ms, 2),
                "num_entities": len(request.customer_ids),
                "feature_view": "customer_transaction_features + customer_profile_features",
            },
        )
    except Exception as e:
        logger.error(f"Error fetching customer features: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/features/product", response_model=FeatureResponse)
async def get_product_features(request: ProductFeatureRequest):
    """Retrieve product catalog features from the online store."""
    start = time.time()
    store = get_store()

    try:
        entity_rows = [{"product_id": pid} for pid in request.product_ids]

        features = store.get_online_features(
            features=[
                "product_catalog_features:product_name",
                "product_catalog_features:category",
                "product_catalog_features:price",
                "product_catalog_features:avg_rating",
                "product_catalog_features:total_reviews",
                "product_catalog_features:stock_level",
                "product_catalog_features:price_percentile",
            ],
            entity_rows=entity_rows,
        ).to_dict()

        result = _online_dict_to_rows(features)
        latency_ms = (time.time() - start) * 1000

        return FeatureResponse(
            features=result,
            metadata={
                "latency_ms": round(latency_ms, 2),
                "num_entities": len(request.product_ids),
                "feature_view": "product_catalog_features",
            },
        )
    except Exception as e:
        logger.error(f"Error fetching product features: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/features/fraud", response_model=FeatureResponse)
async def get_fraud_features(request: FraudFeatureRequest):
    """
    Retrieve fraud detection features including on-demand computed features.

    This endpoint demonstrates how on-demand feature views work:
    1. Historical features are fetched from Redis (customer_transaction_features).
    2. Request-time data (transaction_amount, merchant_category) is combined.
    3. On-demand transforms (z-score, risk flags) are computed in real-time.
    """
    start = time.time()
    store = get_store()

    try:
        entity_rows = [
            {
                "customer_id": request.customer_id,
                "transaction_amount": request.transaction_amount,
                "merchant_category": request.merchant_category,
            }
        ]

        features = store.get_online_features(
            features=[
                "customer_transaction_features:total_transactions_30d",
                "customer_transaction_features:avg_transaction_amount_30d",
                "customer_transaction_features:std_transaction_amount_30d",
                "customer_transaction_features:max_transaction_amount_30d",
                "transaction_risk_features:transaction_amount_zscore",
                "transaction_risk_features:is_high_value_transaction",
                "transaction_risk_features:amount_to_avg_ratio",
                "transaction_risk_features:is_above_spend_pattern",
                "customer_profile_features:customer_segment",
            ],
            entity_rows=entity_rows,
        ).to_dict()

        result = _online_dict_to_rows(features)
        latency_ms = (time.time() - start) * 1000

        return FeatureResponse(
            features=result,
            metadata={
                "latency_ms": round(latency_ms, 2),
                "num_entities": 1,
                "feature_service": "fraud_detection",
                "includes_on_demand": True,
            },
        )
    except Exception as e:
        logger.error(f"Error fetching fraud features: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/features/batch", response_model=FeatureResponse)
async def get_batch_features(request: BatchFeatureRequest):
    """
    Batch feature retrieval using a named feature service.

    Feature services bundle multiple feature views into a single logical
    group that a model consumes. This endpoint lets you request all
    features for a given service in one call.
    """
    start = time.time()
    store = get_store()

    try:
        entity_rows = _entities_to_rows(request.entities)

        features = store.get_online_features(
            features=store.get_feature_service(request.feature_service),
            entity_rows=entity_rows,
        ).to_dict()

        result = _online_dict_to_rows(features)
        latency_ms = (time.time() - start) * 1000

        return FeatureResponse(
            features=result,
            metadata={
                "latency_ms": round(latency_ms, 2),
                "num_entities": len(entity_rows),
                "feature_service": request.feature_service,
            },
        )
    except Exception as e:
        logger.error(f"Error fetching batch features: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features/metadata")
async def get_metadata():
    """
    List all registered feature views, entities, and feature services.
    Useful for feature discovery and debugging.
    """
    store = get_store()

    try:
        feature_views = [
            {
                "name": fv.name,
                "entities": [e.name for e in fv.entity_columns],
                "features": [f.name for f in fv.schema],
                "ttl_seconds": int(fv.ttl.total_seconds()) if fv.ttl else None,
                "online": fv.online,
                "tags": dict(fv.tags),
            }
            for fv in store.list_feature_views()
        ]

        entities = [
            {"name": e.name, "join_keys": e.join_keys, "description": e.description}
            for e in store.list_entities()
        ]

        feature_services = [
            {"name": fs.name, "tags": dict(fs.tags), "description": fs.description}
            for fs in store.list_feature_services()
        ]

        return {
            "project": store.project,
            "feature_views": feature_views,
            "entities": entities,
            "feature_services": feature_services,
        }
    except Exception as e:
        logger.error(f"Error fetching metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Helpers
# =============================================================================

def _online_dict_to_rows(features_dict: dict) -> list[dict]:
    """Convert Feast online feature dict format to a list of row dicts."""
    if not features_dict:
        return []
    keys = list(features_dict.keys())
    n_rows = len(features_dict[keys[0]])
    return [
        {k: features_dict[k][i] for k in keys}
        for i in range(n_rows)
    ]


def _entities_to_rows(entities: dict[str, list]) -> list[dict]:
    """Convert {'customer_id': ['C1', 'C2']} to [{'customer_id': 'C1'}, ...]."""
    keys = list(entities.keys())
    n_rows = len(entities[keys[0]])
    return [
        {k: entities[k][i] for k in keys}
        for i in range(n_rows)
    ]
