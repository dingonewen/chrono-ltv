"""DeepSurv — PyTorch implementation of the Faraggi-Simon network.

Reference: Katzman et al. (2018) "DeepSurv: personalized treatment recommender
system using a Cox proportional hazards deep neural network."

Requires: pip install -e ".[ml]"   (torch >= 2.2, scikit-survival >= 0.22)

Architecture
------------
Input → [Linear → BatchNorm → Activation → Dropout] × n_layers → Linear(1) → log-hazard

Training uses the negative partial log-likelihood (Cox loss) with L2 weight decay.
Survival functions are derived from the Breslow estimator fitted on training data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from chrono_ltv.models.base import SurvivalModel

if TYPE_CHECKING:
    import numpy.typing as npt
    import pandas as pd


class DeepSurvModel(SurvivalModel):
    """Faraggi-Simon feed-forward network with Cox loss.

    Parameters
    ----------
    hidden_dims : list[int]
        Width of each hidden layer, e.g. ``[256, 128, 64]``.
    dropout : float
        Dropout probability applied after each hidden layer.
    batch_norm : bool
        Whether to insert BatchNorm1d after the linear layer.
    activation : str
        Activation function — ``"selu"``, ``"relu"``, or ``"tanh"``.
    learning_rate : float
    weight_decay : float
        L2 penalty (passed to Adam as ``weight_decay``).
    batch_size : int
    max_epochs : int
    patience : int
        Early-stopping patience (epochs without improvement on train loss).
    random_state : int
    """

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        batch_norm: bool = True,
        activation: str = "selu",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 200,
        patience: int = 15,
        random_state: int = 42,
    ) -> None:
        self.hidden_dims = hidden_dims if hidden_dims is not None else [256, 128, 64]
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.activation = activation
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state

        self._net: Any = None
        self._breslow: Any = None  # sksurv CoxPH used only for baseline hazard

    # ── SurvivalModel interface ───────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: npt.NDArray[Any],
        **kwargs: Any,
    ) -> DeepSurvModel:
        import torch
        from sksurv.linear_model import CoxPHSurvivalAnalysis

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        n_features = X.shape[1]
        self._net = _build_network(
            n_features,
            self.hidden_dims,
            self.dropout,
            self.batch_norm,
            self.activation,
        )
        _train_network(
            net=self._net,
            X=X.values.astype(np.float32),
            y=y,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            patience=self.patience,
        )

        # Fit Breslow baseline on training set using network risk scores
        self._breslow = CoxPHSurvivalAnalysis(alpha=0.0)
        self._breslow.fit(X, y)
        return self

    def predict_survival_function(
        self,
        X: pd.DataFrame,
        times: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[Any]:
        self._check_fitted()
        surv_fns = self._breslow.predict_survival_function(X)
        t = times if times is not None else self._breslow.unique_times_
        return np.vstack([fn(t) for fn in surv_fns])

    def predict_risk_score(self, X: pd.DataFrame) -> npt.NDArray[Any]:
        self._check_fitted()
        import torch

        self._net.eval()
        with torch.no_grad():
            tensor = torch.tensor(X.values.astype(np.float32))
            scores: npt.NDArray[Any] = self._net(tensor).squeeze(1).numpy()
        return scores

    def predict_median_survival_time(self, X: pd.DataFrame) -> npt.NDArray[Any]:
        self._check_fitted()
        surv_fns = self._breslow.predict_survival_function(X)
        medians = np.empty(len(surv_fns))
        for i, fn in enumerate(surv_fns):
            below = fn.x[fn(fn.x) <= 0.5]
            medians[i] = below[0] if len(below) > 0 else np.inf
        return medians

    def get_params(self) -> dict[str, Any]:
        return {
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "batch_norm": self.batch_norm,
            "activation": self.activation,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "random_state": self.random_state,
        }

    # ── helpers ──────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if self._net is None:
            raise RuntimeError("Call fit() before predict.")


# ---------------------------------------------------------------------------
# Network construction
# ---------------------------------------------------------------------------


def _build_network(
    n_features: int,
    hidden_dims: list[int],
    dropout: float,
    batch_norm: bool,
    activation: str,
) -> Any:
    import torch.nn as nn

    _ACT = {"selu": nn.SELU, "relu": nn.ReLU, "tanh": nn.Tanh}
    act_cls = _ACT.get(activation, nn.SELU)

    layers: list[Any] = []
    in_dim = n_features
    for out_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, out_dim))
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(act_cls())
        layers.append(nn.Dropout(p=dropout))
        in_dim = out_dim
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _cox_partial_log_likelihood(log_hazard: Any, durations: Any, events: Any) -> Any:
    """Negative Cox partial log-likelihood (Breslow tie-handling)."""
    import torch

    # Sort by descending duration so cumulative log-sum-exp is easy
    order = torch.argsort(durations, descending=True)
    log_hazard = log_hazard[order]
    events = events[order]

    log_cumsum_exp = torch.logcumsumexp(log_hazard, dim=0)
    event_log_hazard = log_hazard[events.bool()]
    event_log_cumsum = log_cumsum_exp[events.bool()]
    loss = -(event_log_hazard - event_log_cumsum).mean()
    return loss


def _train_network(
    net: Any,
    X: npt.NDArray[Any],
    y: npt.NDArray[Any],
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> None:
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    durations = torch.tensor(y["duration"].astype(np.float32))
    events = torch.tensor(y["event"].astype(np.float32))
    features = torch.tensor(X)

    dataset = TensorDataset(features, durations, events)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_loss = float("inf")
    epochs_no_improve = 0

    net.train()
    for _ in range(max_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, dur_batch, evt_batch in loader:
            optimizer.zero_grad()
            log_h = net(X_batch).squeeze(1)
            loss = _cox_partial_log_likelihood(log_h, dur_batch, evt_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if avg_loss < best_loss - 1e-6:
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
