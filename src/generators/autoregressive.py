"""Probabilistic autoregressive GRU utilities for factor trajectories."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


N_MARKET_FEATURES = 1
N_SECTOR_FEATURES = 30
N_IDIO_FEATURES = 30


class ProbabilisticGRU(nn.Module):
    """Predict a diagonal Gaussian distribution for the next factor vector."""

    def __init__(
        self,
        input_dim: int = 61,
        hidden_dims: Iterable[int] = (128,),
        dropout: float = 0.0,
        min_logvar: float = -12.0,
        max_logvar: float = 6.0,
    ) -> None:
        super().__init__()
        hidden_dims = [int(value) for value in hidden_dims]
        if not hidden_dims:
            raise ValueError("hidden_dims no puede estar vacío")
        self.input_dim = int(input_dim)
        self.hidden_dims = hidden_dims
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)
        self.gru_layers = nn.ModuleList()
        current_dim = self.input_dim
        for hidden_dim in hidden_dims:
            self.gru_layers.append(nn.GRU(current_dim, hidden_dim, batch_first=True))
            current_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.mu_head = nn.Linear(current_dim, self.input_dim)
        self.logvar_head = nn.Linear(current_dim, self.input_dim)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = context
        for index, gru in enumerate(self.gru_layers):
            hidden, _ = gru(hidden)
            if index < len(self.gru_layers) - 1:
                hidden = self.dropout(hidden)
        state = self.dropout(hidden[:, -1])
        mu = self.mu_head(state)
        logvar = torch.clamp(self.logvar_head(state), self.min_logvar, self.max_logvar)
        return mu, logvar


class ProbabilisticMultiHorizonGRU(nn.Module):
    """Predict diagonal-Gaussian parameters for several consecutive future days."""

    def __init__(self, input_dim=61, prediction_horizon=5, hidden_dims=(64,), dropout=0.1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.prediction_horizon = int(prediction_horizon)
        self.hidden_dims = [int(value) for value in hidden_dims]
        if not self.hidden_dims:
            raise ValueError("hidden_dims no puede estar vacío")
        self.gru_layers = nn.ModuleList()
        current_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            self.gru_layers.append(nn.GRU(current_dim, hidden_dim, batch_first=True))
            current_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        output_dim = self.prediction_horizon * self.input_dim
        self.mu_head = nn.Linear(current_dim, output_dim)
        self.logvar_head = nn.Linear(current_dim, output_dim)

    def forward(self, context):
        hidden = context
        for index, gru in enumerate(self.gru_layers):
            hidden, _ = gru(hidden)
            if index < len(self.gru_layers) - 1:
                hidden = self.dropout(hidden)
        state = self.dropout(hidden[:, -1])
        shape = (-1, self.prediction_horizon, self.input_dim)
        mu = self.mu_head(state).reshape(shape)
        logvar = self.logvar_head(state).reshape(shape).clamp(-12.0, 6.0)
        return mu, logvar


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def autoregressive_target_indices(n_rows: int, context_length: int) -> np.ndarray:
    if context_length <= 0:
        raise ValueError("context_length debe ser positivo")
    return np.arange(context_length, n_rows, dtype=int)


def split_autoregressive_targets(
    target_indices: Iterable[int],
    validation_blocks: Iterable[tuple[int, int]],
    context_length: int,
    purge_size: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split targets and purge training contexts around validation blocks."""
    train, validation = [], []
    blocks = [(int(start), int(end)) for start, end in validation_blocks]
    for target in np.asarray(list(target_indices), dtype=int):
        context_start = target - context_length
        if any(start <= target <= end for start, end in blocks):
            validation.append(target)
            continue
        overlaps_embargo = any(
            context_start <= end + purge_size and target >= start - purge_size
            for start, end in blocks
        )
        if not overlaps_embargo:
            train.append(target)
    return np.asarray(train, dtype=int), np.asarray(validation, dtype=int)


def make_autoregressive_samples(
    frame: pd.DataFrame,
    context_length: int,
    target_indices: Iterable[int],
) -> tuple[np.ndarray, np.ndarray]:
    values = frame.to_numpy(dtype=np.float32)
    targets = np.asarray(list(target_indices), dtype=int)
    contexts = np.stack([values[t - context_length:t] for t in targets])
    next_days = np.stack([values[t] for t in targets])
    return contexts, next_days


def multihorizon_target_indices(n_dates, context_length, prediction_horizon):
    return np.arange(context_length, n_dates - prediction_horizon + 1, dtype=int)


def split_multihorizon_targets(
    target_indices, validation_blocks, context_length, prediction_horizon, purge_size=0,
):
    """Split complete context and five-day target spans without leakage."""
    train, validation = [], []
    blocks = [(int(start), int(end)) for start, end in validation_blocks]
    for target_start in np.asarray(list(target_indices), dtype=int):
        target_end = target_start + prediction_horizon - 1
        if any(target_start >= start and target_end <= end for start, end in blocks):
            validation.append(target_start)
            continue
        sample_start = target_start - context_length
        overlaps = any(
            sample_start <= end + purge_size and target_end >= start - purge_size
            for start, end in blocks
        )
        if not overlaps:
            train.append(target_start)
    return np.asarray(train, dtype=int), np.asarray(validation, dtype=int)


def make_multihorizon_samples(frame, context_length, prediction_horizon, target_indices):
    values = frame.to_numpy(dtype=np.float32)
    targets = np.asarray(list(target_indices), dtype=int)
    contexts = np.stack([values[t-context_length:t] for t in targets])
    futures = np.stack([values[t:t+prediction_horizon] for t in targets])
    return contexts, futures


def gaussian_nll_elements(
    target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    """Stable Gaussian NLL without the irrelevant log(2π) constant."""
    return 0.5 * (logvar + (target - mu).pow(2) * torch.exp(-logvar))


def probabilistic_metrics(
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    loss_weights: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    if loss_weights is None:
        loss_weights = {"market": 1 / 3, "sector": 1 / 3, "idio": 1 / 3}
    required = {"market", "sector", "idio"}
    if set(loss_weights) != required:
        raise ValueError(f"loss_weights debe contener exactamente {sorted(required)}")
    if not np.isclose(sum(loss_weights.values()), 1.0):
        raise ValueError("Los pesos de loss_weights deben sumar 1")
    nll = gaussian_nll_elements(target, mu, logvar)
    squared_error = (target - mu).pow(2)
    market = slice(0, N_MARKET_FEATURES)
    sector = slice(N_MARKET_FEATURES, N_MARKET_FEATURES + N_SECTOR_FEATURES)
    idio = slice(N_MARKET_FEATURES + N_SECTOR_FEATURES, None)
    market_nll, sector_nll, idio_nll = (
        nll[..., market].mean(), nll[..., sector].mean(), nll[..., idio].mean()
    )
    selection_score = (
        loss_weights["market"] * market_nll
        + loss_weights["sector"] * sector_nll
        + loss_weights["idio"] * idio_nll
    )
    sigma = torch.exp(0.5 * logvar)
    standardized = (target - mu) / sigma.clamp_min(1e-6)
    return {
        "total_nll": nll.mean(),
        "selection_score": selection_score,
        "market_nll": market_nll,
        "sector_nll": sector_nll,
        "idio_nll": idio_nll,
        "mse": squared_error.mean(),
        "market_mse": squared_error[..., market].mean(),
        "sector_mse": squared_error[..., sector].mean(),
        "idio_mse": squared_error[..., idio].mean(),
        "coverage_68": (standardized.abs() <= 1.0).float().mean(),
        "coverage_95": (standardized.abs() <= 1.96).float().mean(),
        "standardized_residual_mean": standardized.mean(),
        "standardized_residual_std": standardized.std(unbiased=False),
    }


def _loader(
    contexts: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(contexts).float(), torch.from_numpy(targets).float())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def _epoch(
    model: ProbabilisticGRU,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    loss_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    n_samples = 0
    context_manager = torch.enable_grad() if training else torch.no_grad()
    with context_manager:
        for contexts, targets in loader:
            contexts, targets = contexts.to(device), targets.to(device)
            mu, logvar = model(contexts)
            metrics = probabilistic_metrics(targets, mu, logvar, loss_weights)
            loss = metrics["selection_score"]
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            size = contexts.shape[0]
            n_samples += size
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * size
    return {name: value / max(n_samples, 1) for name, value in totals.items()}


def train_probabilistic_gru(
    model: ProbabilisticGRU,
    train_contexts: np.ndarray,
    train_targets: np.ndarray,
    validation_contexts: np.ndarray,
    validation_targets: np.ndarray,
    *,
    epochs: int = 300,
    batch_size: int = 128,
    learning_rate: float = 5e-4,
    patience: int = 40,
    seed: int = 42,
    device: str | torch.device = "cpu",
    checkpoint_path: str | Path | None = None,
    loss_weights: dict[str, float] | None = None,
    weight_decay: float = 0.0,
) -> tuple[dict[str, list[float]], int, float]:
    device = torch.device(device)
    model.to(device)
    if loss_weights is None:
        loss_weights = {"market": 1 / 3, "sector": 1 / 3, "idio": 1 / 3}
    train_loader = _loader(train_contexts, train_targets, batch_size, True, seed)
    val_loader = _loader(validation_contexts, validation_targets, batch_size, False, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    metric_names = list(probabilistic_metrics(
        torch.zeros(2, model.input_dim), torch.zeros(2, model.input_dim),
        torch.zeros(2, model.input_dim), loss_weights,
    ))
    history = {"epoch": []}
    for prefix in ("train", "val"):
        for metric in metric_names:
            history[f"{prefix}_{metric}"] = []
    best_score, best_epoch = float("inf"), 0
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    for epoch in range(1, epochs + 1):
        train_metrics = _epoch(model, train_loader, optimizer, device, loss_weights)
        val_metrics = _epoch(model, val_loader, None, device, loss_weights)
        history["epoch"].append(epoch)
        for name in metric_names:
            history[f"train_{name}"].append(train_metrics[name])
            history[f"val_{name}"].append(val_metrics[name])
        score = val_metrics["selection_score"]
        if score < best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
            if checkpoint_path is not None:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": best_state,
                    "best_epoch": best_epoch,
                    "best_validation_score": best_score,
                    "loss_weights": loss_weights,
                }, checkpoint_path)
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return history, best_epoch, best_score


def train_probabilistic_gru_full(
    model: ProbabilisticGRU,
    contexts: np.ndarray,
    targets: np.ndarray,
    *,
    epochs: int,
    batch_size: int = 128,
    learning_rate: float = 5e-4,
    seed: int = 42,
    device: str | torch.device = "cpu",
    checkpoint_path: str | Path | None = None,
    loss_weights: dict[str, float] | None = None,
    weight_decay: float = 0.0,
) -> dict[str, list[float]]:
    device = torch.device(device)
    model.to(device)
    if loss_weights is None:
        loss_weights = {"market": 1 / 3, "sector": 1 / 3, "idio": 1 / 3}
    loader = _loader(contexts, targets, batch_size, True, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: dict[str, list[float]] = {"epoch": []}
    for epoch in range(1, epochs + 1):
        metrics = _epoch(model, loader, optimizer, device, loss_weights)
        history["epoch"].append(epoch)
        for name, value in metrics.items():
            history.setdefault(f"train_{name}", []).append(value)
    if checkpoint_path is not None:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(), "epochs": epochs,
            "loss_weights": loss_weights,
        }, checkpoint_path)
    return history


def generate_recursive(
    model: ProbabilisticGRU,
    normalized_context: np.ndarray,
    horizon: int,
    *,
    temperature: float = 1.0,
    seed: int = 42,
    epsilon_sequence: np.ndarray | None = None,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate recursively without using any future real observations."""
    device = torch.device(device)
    model.to(device).eval()
    context = np.asarray(normalized_context, dtype=np.float32).copy()
    if epsilon_sequence is None:
        epsilon_sequence = np.random.default_rng(seed).normal(
            size=(horizon, model.input_dim)
        ).astype(np.float32)
    epsilon_sequence = np.asarray(epsilon_sequence, dtype=np.float32)
    if epsilon_sequence.shape != (horizon, model.input_dim):
        raise ValueError("epsilon_sequence tiene forma incompatible")
    generated, predicted_mu, predicted_sigma = [], [], []
    with torch.no_grad():
        for step in range(horizon):
            tensor = torch.from_numpy(context[None]).float().to(device)
            mu, logvar = model(tensor)
            mu_np = mu.cpu().numpy()[0]
            sigma_np = torch.exp(0.5 * logvar).cpu().numpy()[0]
            next_day = mu_np + float(temperature) * sigma_np * epsilon_sequence[step]
            generated.append(next_day)
            predicted_mu.append(mu_np)
            predicted_sigma.append(sigma_np)
            context = np.concatenate([context[1:], next_day[None]], axis=0)
    return (
        np.asarray(generated, dtype=np.float32),
        np.asarray(predicted_mu, dtype=np.float32),
        np.asarray(predicted_sigma, dtype=np.float32),
    )


def generate_recursive_batch(
    model: ProbabilisticGRU,
    normalized_context: np.ndarray,
    horizon: int,
    *,
    temperature: float = 1.0,
    seed: int = 42,
    epsilon_sequences: np.ndarray | None = None,
    n_futures: int = 1,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate several recursive futures in parallel.

    Returns arrays shaped ``(n_futures, horizon, input_dim)``. Every future
    starts from the same real context but evolves with its own noise sequence.
    """
    device = torch.device(device)
    model.to(device).eval()
    context = np.asarray(normalized_context, dtype=np.float32)
    if context.ndim != 2 or context.shape[1] != model.input_dim:
        raise ValueError("normalized_context debe tener forma (context_length, input_dim)")
    if epsilon_sequences is None:
        epsilon_sequences = np.random.default_rng(seed).normal(
            size=(n_futures, horizon, model.input_dim)
        ).astype(np.float32)
    epsilon_sequences = np.asarray(epsilon_sequences, dtype=np.float32)
    expected_shape = (n_futures, horizon, model.input_dim)
    if epsilon_sequences.shape != expected_shape:
        raise ValueError(f"epsilon_sequences debe tener forma {expected_shape}")

    contexts = np.repeat(context[None, :, :], n_futures, axis=0)
    generated, predicted_mu, predicted_sigma = [], [], []
    with torch.no_grad():
        for step in range(horizon):
            tensor = torch.from_numpy(contexts).float().to(device)
            mu, logvar = model(tensor)
            sigma = torch.exp(0.5 * logvar)
            epsilon = torch.from_numpy(epsilon_sequences[:, step]).float().to(device)
            next_day = mu + float(temperature) * sigma * epsilon
            next_np = next_day.cpu().numpy()
            generated.append(next_np)
            predicted_mu.append(mu.cpu().numpy())
            predicted_sigma.append(sigma.cpu().numpy())
            contexts = np.concatenate([contexts[:, 1:], next_np[:, None, :]], axis=1)
    return (
        np.stack(generated, axis=1).astype(np.float32),
        np.stack(predicted_mu, axis=1).astype(np.float32),
        np.stack(predicted_sigma, axis=1).astype(np.float32),
    )


def generate_multihorizon_rolling_batch(
    model: ProbabilisticMultiHorizonGRU,
    normalized_context: np.ndarray,
    generation_horizon: int,
    *,
    temperature: float = 1.0,
    n_paths: int = 1,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Refresh a five-day forecast after appending each newly sampled day.

    The model learns all five leads jointly. During a rollout, calendar
    alignment requires sampling lead 1 of the refreshed distribution: after
    appending that day, the network is run again and produces a new forecast
    for the next five days.
    """
    device = torch.device(device)
    model.to(device).eval()
    context = np.asarray(normalized_context, dtype=np.float32)
    contexts = np.repeat(context[None], n_paths, axis=0)
    rng = np.random.default_rng(seed)
    generated, lead_mu, lead_sigma = [], [], []
    with torch.no_grad():
        for _ in range(generation_horizon):
            tensor = torch.from_numpy(contexts).float().to(device)
            mu_five, logvar_five = model(tensor)
            mu = mu_five[:, 0]
            sigma = torch.exp(0.5 * logvar_five[:, 0])
            epsilon = torch.from_numpy(
                rng.normal(size=(n_paths, model.input_dim)).astype(np.float32)
            ).to(device)
            next_day = mu + float(temperature) * sigma * epsilon
            next_np = next_day.cpu().numpy()
            generated.append(next_np)
            lead_mu.append(mu.cpu().numpy())
            lead_sigma.append(sigma.cpu().numpy())
            contexts = np.concatenate([contexts[:, 1:], next_np[:, None]], axis=1)
    return (
        np.stack(generated, axis=1).astype(np.float32),
        np.stack(lead_mu, axis=1).astype(np.float32),
        np.stack(lead_sigma, axis=1).astype(np.float32),
    )
