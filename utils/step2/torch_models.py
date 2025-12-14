# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = None  # type: ignore
    TensorDataset = None  # type: ignore


def _set_seed(seed: int) -> None:
    if torch is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-12
    return (X - mean) / std, mean, std


def _standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / (std + 1e-12)


def build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (seq_X, seq_y) for given target indices.

    Only keeps indices i where the whole window [i-seq_len+1, i] is contained in indices set.
    This prevents leakage when indices are not fully contiguous.
    """
    idx_set = set(int(i) for i in indices.tolist())
    seq_X, seq_y = [], []
    for i in indices:
        i = int(i)
        j0 = i - seq_len + 1
        if j0 < 0:
            continue
        ok = True
        for j in range(j0, i + 1):
            if j not in idx_set:
                ok = False
                break
        if not ok:
            continue
        seq_X.append(X[j0 : i + 1])
        seq_y.append(y[i])
    if len(seq_X) == 0:
        return np.zeros((0, seq_len, X.shape[1]), dtype=float), np.zeros((0,), dtype=float)
    return np.asarray(seq_X, dtype=float), np.asarray(seq_y, dtype=float)


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,T,F)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return x


class TransformerRegressor(nn.Module):
    def __init__(self, n_features: int, d_model: int, nhead: int, num_layers: int, dropout: float, seq_len: int):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model=d_model, max_len=max(512, seq_len + 10))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x)
        z = self.pos(z)
        z = self.enc(z)
        last = z[:, -1, :]
        return self.fc(last).squeeze(-1)


@dataclass
class TorchSequenceRegressor:
    """A light-weight torch regressor with sklearn-like fit/predict.

    It expects tabular X and internally builds sequences of length seq_len.
    """

    arch: str
    seq_len: int
    standardize: bool = True
    random_seed: int = 42

    # training params
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    d_model: int = 64
    nhead: int = 4
    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 20
    device: str = "cpu"

    _mean: Optional[np.ndarray] = None
    _std: Optional[np.ndarray] = None
    _model: Any = None

    def _make_model(self, n_features: int) -> nn.Module:
        if self.arch == "LSTM":
            return LSTMRegressor(n_features=n_features, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.dropout)
        if self.arch == "Transformer":
            return TransformerRegressor(n_features=n_features, d_model=self.d_model, nhead=self.nhead, num_layers=self.num_layers, dropout=self.dropout, seq_len=self.seq_len)
        raise ValueError(f"Unknown torch arch: {self.arch}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchSequenceRegressor":
        if torch is None:
            raise ImportError("torch not available")
        _set_seed(self.random_seed)

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if self.standardize:
            Xs, mean, std = _standardize_fit(X)
            self._mean, self._std = mean, std
        else:
            Xs = X

        indices = np.arange(len(Xs))
        seq_X, seq_y = build_sequences(Xs, y, indices, self.seq_len)
        if len(seq_X) < 50:
            raise ValueError("Not enough sequences to train torch model. Reduce seq_len or increase data.")

        device = torch.device(self.device if torch.cuda.is_available() and self.device.startswith("cuda") else "cpu")

        model = self._make_model(n_features=X.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=float(self.lr))
        loss_fn = nn.MSELoss()

        ds = TensorDataset(torch.tensor(seq_X, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.float32))
        dl = DataLoader(ds, batch_size=int(self.batch_size), shuffle=True, drop_last=False)

        model.train()
        for _ in range(int(self.epochs)):
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

        self._model = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if torch is None or self._model is None:
            raise RuntimeError("Model not fitted")
        X = np.asarray(X, dtype=float)

        if self.standardize and self._mean is not None and self._std is not None:
            Xs = _standardize_apply(X, self._mean, self._std)
        else:
            Xs = X

        # Build sequences for all indices
        indices = np.arange(len(Xs))
        seq_X, _ = build_sequences(Xs, np.zeros(len(Xs)), indices, self.seq_len)
        # predict is defined for indices >= seq_len-1; we will pad front with nan then drop upstream
        device = next(self._model.parameters()).device
        self._model.eval()
        with torch.no_grad():
            xb = torch.tensor(seq_X, dtype=torch.float32).to(device)
            pred_seq = self._model(xb).cpu().numpy().reshape(-1)

        out = np.full((len(Xs),), np.nan, dtype=float)
        out[self.seq_len - 1 :] = pred_seq
        return out
