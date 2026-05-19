"""Feature Engineering Pipeline for ChronoLTV.

Usage
-----
>>> from chrono_ltv.features.pipeline import FeaturePipeline
>>> pipe = FeaturePipeline()
>>> X, y = pipe.fit_transform(customers, transactions, clickstream, tickets, labels)

``X``  — float64 DataFrame indexed by customer_id, NaNs imputed.
``y``  — structured numpy array with dtype ``[("event", bool), ("duration", float)]``
         compatible with scikit-survival estimators.

Requires: pip install -e ".[features]"   (scikit-learn >= 1.4)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from chrono_ltv.features.encoders import (
    BehavioralFeatureExtractor,
    CustomerFeatureExtractor,
    RFMFeatureExtractor,
    TicketFeatureExtractor,
)

if TYPE_CHECKING:
    import numpy.typing as npt


# dtype used by scikit-survival's fit() / score() methods
SURVIVAL_DTYPE = np.dtype([("event", bool), ("duration", np.float64)])


class FeaturePipeline:
    """End-to-end feature engineering: extract → merge → impute → scale.

    Parameters
    ----------
    scale : bool
        Whether to apply ``StandardScaler`` to numeric columns after imputation.
        Default True.
    reference_date : pd.Timestamp | None
        Passed to ``RFMFeatureExtractor``.  None → max transaction date.
    """

    def __init__(
        self,
        scale: bool = True,
        reference_date: pd.Timestamp | None = None,
    ) -> None:
        self.scale = scale
        self.reference_date = reference_date

        self._rfm = RFMFeatureExtractor(reference_date=reference_date)
        self._behavioral = BehavioralFeatureExtractor()
        self._ticket = TicketFeatureExtractor()
        self._customer = CustomerFeatureExtractor()

        self._preprocessor: Any = None  # sklearn ColumnTransformer, set in fit
        self._feature_names: list[str] = []

    # ── public ───────────────────────────────────────────────────────────

    def fit(
        self,
        customers: pd.DataFrame,
        transactions: pd.DataFrame,
        clickstream: pd.DataFrame,
        tickets: pd.DataFrame,
        labels: pd.DataFrame,
    ) -> FeaturePipeline:
        self._rfm.fit(transactions)
        raw = self._extract_raw(customers, transactions, clickstream, tickets, labels)
        X = raw.drop(columns=["customer_id"])
        self._preprocessor = self._build_preprocessor(X)
        self._preprocessor.fit(X)
        self._feature_names = list(self._preprocessor.get_feature_names_out())
        return self

    def transform(
        self,
        customers: pd.DataFrame,
        transactions: pd.DataFrame,
        clickstream: pd.DataFrame,
        tickets: pd.DataFrame,
        labels: pd.DataFrame,
    ) -> tuple[pd.DataFrame, npt.NDArray[Any]]:
        if self._preprocessor is None:
            raise RuntimeError("Call fit() before transform().")
        raw = self._extract_raw(customers, transactions, clickstream, tickets, labels)
        customer_ids = raw["customer_id"].values
        X_raw = raw.drop(columns=["customer_id"])
        X_arr = self._preprocessor.transform(X_raw)
        X = pd.DataFrame(X_arr, columns=self._feature_names, index=customer_ids)
        X.index.name = "customer_id"
        y = self._build_targets(labels, customer_ids)
        return X, y

    def fit_transform(
        self,
        customers: pd.DataFrame,
        transactions: pd.DataFrame,
        clickstream: pd.DataFrame,
        tickets: pd.DataFrame,
        labels: pd.DataFrame,
    ) -> tuple[pd.DataFrame, npt.NDArray[Any]]:
        self.fit(customers, transactions, clickstream, tickets, labels)
        return self.transform(customers, transactions, clickstream, tickets, labels)

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    # ── private ──────────────────────────────────────────────────────────

    def _extract_raw(
        self,
        customers: pd.DataFrame,
        transactions: pd.DataFrame,
        clickstream: pd.DataFrame,
        tickets: pd.DataFrame,
        labels: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge all per-customer feature tables into a single wide DataFrame."""
        rfm = self._rfm.transform(transactions)
        behavioral = self._behavioral.transform(clickstream)
        ticket_feats = self._ticket.transform(tickets)
        customer_feats = self._customer.transform(customers)

        # Start from the label index so every customer has a row
        base = labels[["customer_id"]].copy()
        merged = (
            base.merge(customer_feats, on="customer_id", how="left")
            .merge(rfm, on="customer_id", how="left")
            .merge(behavioral, on="customer_id", how="left")
            .merge(ticket_feats, on="customer_id", how="left")
        )
        # Customers without tickets get 0 for count and NaN for rates
        if "n_tickets" in merged.columns:
            merged["n_tickets"] = merged["n_tickets"].fillna(0.0)
        return merged

    def _build_preprocessor(self, X: pd.DataFrame) -> Any:
        """Build a ColumnTransformer with median imputation + optional scaling."""
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        bool_cols = X.select_dtypes(include=[bool]).columns.tolist()

        # Ensure bool columns are treated as numeric (0/1)
        all_numeric = numeric_cols + bool_cols

        imputer = SimpleImputer(strategy="median")
        steps: list[Any] = [("impute", imputer)]
        if self.scale:
            steps.append(("scale", StandardScaler()))
        num_pipe = Pipeline(steps)

        return ColumnTransformer(
            transformers=[("num", num_pipe, all_numeric)],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    @staticmethod
    def _build_targets(
        labels: pd.DataFrame,
        customer_ids: npt.NDArray[Any],
    ) -> npt.NDArray[Any]:
        """Build structured array ``(event: bool, duration: float)`` aligned to *customer_ids*."""
        label_indexed = labels.set_index("customer_id")
        events = label_indexed.loc[customer_ids, "event_observed"].astype(bool).values
        durations = label_indexed.loc[customer_ids, "duration_days"].astype(float).values
        y: npt.NDArray[Any] = np.empty(len(customer_ids), dtype=SURVIVAL_DTYPE)
        y["event"] = events
        y["duration"] = durations
        return y
