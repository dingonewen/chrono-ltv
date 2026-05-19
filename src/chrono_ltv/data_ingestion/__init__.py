"""Multi-source e-commerce data ingestion for ChronoLTV.

Provides schemas, aggregation, and validation for three heterogeneous data
streams that feed the feature engineering pipeline:
- Shopify webhook payloads
- Amazon Seller Central order reports
- 3PL / logistics provider invoices
"""

from chrono_ltv.data_ingestion.aggregator import MultiSourceAggregator
from chrono_ltv.data_ingestion.schemas import (
    AmazonOrderReport,
    Logistics3PLInvoice,
    ShopifyWebhookPayload,
)
from chrono_ltv.data_ingestion.validators import IngestionValidator

__all__ = [
    "AmazonOrderReport",
    "Logistics3PLInvoice",
    "MultiSourceAggregator",
    "ShopifyWebhookPayload",
    "IngestionValidator",
]
