-- Feature Store Lab - PostgreSQL Initialization
-- This script runs automatically when the postgres container starts for the first time.

-- Create schema for Feast registry and offline store tables
CREATE SCHEMA IF NOT EXISTS feast;

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA feast TO feast;
GRANT ALL PRIVILEGES ON SCHEMA public TO feast;

-- Create a table for raw transactions (used by the offline store)
CREATE TABLE IF NOT EXISTS public.transactions (
    transaction_id VARCHAR(20) PRIMARY KEY,
    customer_id VARCHAR(10) NOT NULL,
    product_id VARCHAR(10) NOT NULL,
    amount DOUBLE PRECISION NOT NULL,
    merchant_category VARCHAR(50),
    event_timestamp TIMESTAMP NOT NULL,
    created_timestamp TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_transactions_customer ON public.transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON public.transactions(event_timestamp);
