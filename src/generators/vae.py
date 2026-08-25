"""Modular GRU variational autoencoder utilities for market episodes."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - gives a useful notebook error
    raise ImportError(
        "El VAE requiere PyTorch. Instala las dependencias de requirements.txt."
    ) from exc


class GRUVAE(nn.Module):
    """VAE that encodes and decodes complete (batch, time, feature) episodes."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int] = (64,),
        latent_dim: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims)
        if not hidden_dims:
            raise ValueError("hidden_dims debe contener al menos una dimensión")

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.dropout = dropout

        encoder_layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(
                nn.GRU(current_dim, hidden_dim, batch_first=True)
            )
            current_dim = hidden_dim
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.mu_layer = nn.Linear(current_dim, latent_dim)
        self.logvar_layer = nn.Linear(current_dim, latent_dim)

        decoder_layers = []
        current_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(
                nn.GRU(current_dim, hidden_dim, batch_first=True)
            )
            current_dim = hidden_dim
        self.decoder_layers = nn.ModuleList(decoder_layers)
        self.output_layer = nn.Linear(current_dim, input_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = x
        for layer in self.encoder_layers:
            hidden, _ = layer(hidden)
            hidden = self.dropout_layer(hidden)
        summary = hidden[:, -1, :]
        return self.mu_layer(summary), self.logvar_layer(summary)

    @staticmethod
    def reparameterize(
        mu: torch.Tensor, logvar: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        if noise is None:
            noise = torch.randn_like(std)
        return mu + noise * std

    def decode(self, z: torch.Tensor, sequence_length: int) -> torch.Tensor:
        hidden = z.unsqueeze(1).repeat(1, sequence_length, 1)
        for layer in self.decoder_layers:
            hidden, _ = layer(hidden)
            hidden = self.dropout_layer(hidden)
        return self.output_layer(hidden)

    def forward(
        self, x: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, noise=noise)
        reconstruction = self.decode(z, x.shape[1])
        return reconstruction, mu, logvar


class MultiBranchGRUVAE(nn.Module):
    """GRU-VAE with market, sector and idiosyncratic encoder/decoder branches."""

    n_market_features = 1
    n_sector_features = 30
    n_idio_features = 30

    def __init__(
        self,
        latent_dim: int = 8,
        market_hidden: Iterable[int] = (8,),
        sector_hidden: Iterable[int] = (32,),
        idio_hidden: Iterable[int] = (32,),
        fusion_hidden: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.market_hidden = list(market_hidden)
        self.sector_hidden = list(sector_hidden)
        self.idio_hidden = list(idio_hidden)
        self.fusion_hidden = fusion_hidden
        self.dropout = dropout
        self.input_dim = self.n_market_features + self.n_sector_features + self.n_idio_features
        if not self.market_hidden or not self.sector_hidden or not self.idio_hidden:
            raise ValueError("Cada rama debe tener al menos una capa GRU")

        self.market_encoder = self._build_gru_stack(self.n_market_features, self.market_hidden)
        self.sector_encoder = self._build_gru_stack(self.n_sector_features, self.sector_hidden)
        self.idio_encoder = self._build_gru_stack(self.n_idio_features, self.idio_hidden)
        representation_dim = self.market_hidden[-1] + self.sector_hidden[-1] + self.idio_hidden[-1]
        self.fusion = nn.Sequential(
            nn.Linear(representation_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mu_layer = nn.Linear(fusion_hidden, latent_dim)
        self.logvar_layer = nn.Linear(fusion_hidden, latent_dim)

        self.market_decoder = self._build_gru_stack(latent_dim, self.market_hidden[::-1])
        self.sector_decoder = self._build_gru_stack(latent_dim, self.sector_hidden[::-1])
        self.idio_decoder = self._build_gru_stack(latent_dim, self.idio_hidden[::-1])
        self.market_output = nn.Linear(self.market_hidden[0], self.n_market_features)
        self.sector_output = nn.Linear(self.sector_hidden[0], self.n_sector_features)
        self.idio_output = nn.Linear(self.idio_hidden[0], self.n_idio_features)
        self.dropout_layer = nn.Dropout(dropout)

    @staticmethod
    def _build_gru_stack(input_dim: int, hidden_dims: list[int]) -> nn.ModuleList:
        layers = nn.ModuleList()
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.GRU(current_dim, hidden_dim, batch_first=True))
            current_dim = hidden_dim
        return layers

    def _run_stack(self, layers: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        hidden = x
        for layer in layers:
            hidden, _ = layer(hidden)
            hidden = self.dropout_layer(hidden)
        return hidden

    def _encode_branch(self, x: torch.Tensor, layers: nn.ModuleList) -> torch.Tensor:
        return self._run_stack(layers, x)[:, -1, :]

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        market = x[:, :, : self.n_market_features]
        sector_end = self.n_market_features + self.n_sector_features
        sector = x[:, :, self.n_market_features : sector_end]
        idio = x[:, :, sector_end :]
        representations = torch.cat(
            [
                self._encode_branch(market, self.market_encoder),
                self._encode_branch(sector, self.sector_encoder),
                self._encode_branch(idio, self.idio_encoder),
            ],
            dim=1,
        )
        fused = self.fusion(representations)
        return self.mu_layer(fused), self.logvar_layer(fused)

    @staticmethod
    def reparameterize(
        mu: torch.Tensor, logvar: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        if noise is None:
            noise = torch.randn_like(std)
        return mu + noise * std

    def _decode_branch(
        self,
        z: torch.Tensor,
        sequence_length: int,
        layers: nn.ModuleList,
        output_layer: nn.Linear,
    ) -> torch.Tensor:
        repeated = z.unsqueeze(1).repeat(1, sequence_length, 1)
        return output_layer(self._run_stack(layers, repeated))

    def decode(self, z: torch.Tensor, sequence_length: int) -> torch.Tensor:
        market = self._decode_branch(z, sequence_length, self.market_decoder, self.market_output)
        sector = self._decode_branch(z, sequence_length, self.sector_decoder, self.sector_output)
        idio = self._decode_branch(z, sequence_length, self.idio_decoder, self.idio_output)
        return torch.cat([market, sector, idio], dim=2)

    def forward(
        self, x: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, noise=noise)
        reconstruction = self.decode(z, x.shape[1])
        return reconstruction, mu, logvar


def set_reproducible_seed(seed: int) -> None:
    """Set Python, NumPy and PyTorch seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_component_frame(
    market_factor: pd.Series,
    sector_factors: pd.DataFrame,
    idiosyncratic: pd.DataFrame,
) -> pd.DataFrame:
    """Combine the 1 + 30 + 30 factor columns in a deterministic order."""
    if not market_factor.index.equals(sector_factors.index):
        raise ValueError("market_factor y sector_factors deben compartir el índice")
    if not market_factor.index.equals(idiosyncratic.index):
        raise ValueError("market_factor e idiosyncratic deben compartir el índice")
    if list(sector_factors.columns) != list(idiosyncratic.columns):
        raise ValueError("sector_factors e idiosyncratic deben compartir columnas")

    frame = pd.concat(
        [market_factor.rename("market_factor"), sector_factors, idiosyncratic],
        axis=1,
    )
    if frame.isna().any().any():
        raise ValueError("Los componentes contienen NaN; no se puede entrenar el VAE")
    return frame.astype("float64")


def window_start_indices(n_rows: int, window_size: int, stride: int = 1) -> np.ndarray:
    """Return valid start positions for fixed-length temporal windows."""
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size y stride deben ser positivos")
    if n_rows < window_size:
        return np.array([], dtype=int)
    return np.arange(0, n_rows - window_size + 1, stride, dtype=int)


def make_windows(
    frame: pd.DataFrame,
    window_size: int,
    starts: Iterable[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create windows and return their start positions."""
    if starts is None:
        starts_array = window_start_indices(len(frame), window_size)
    else:
        starts_array = np.asarray(list(starts), dtype=int)
    values = frame.to_numpy(dtype=np.float32)
    windows = np.stack(
        [values[start : start + window_size] for start in starts_array], axis=0
    ) if len(starts_array) else np.empty((0, window_size, frame.shape[1]), dtype=np.float32)
    return windows, starts_array


def select_validation_blocks(
    dates: pd.DatetimeIndex,
    n_blocks: int = 4,
    block_length: int = 80,
    purge_size: int = 79,
    seed: int = 42,
) -> list[tuple[int, int]]:
    """Select reproducible, separated date-position validation blocks."""
    n_dates = len(dates)
    if n_blocks < 1 or block_length <= 0:
        raise ValueError("n_blocks y block_length deben ser positivos")
    if n_dates < n_blocks * block_length:
        raise ValueError("No hay suficientes fechas para los bloques de validación")

    rng = np.random.default_rng(seed)
    candidates = np.arange(0, n_dates - block_length + 1)
    rng.shuffle(candidates)
    selected: list[tuple[int, int]] = []
    for start in candidates:
        end = int(start + block_length - 1)
        if all(
            end + purge_size < old_start or start - purge_size > old_end
            for old_start, old_end in selected
        ):
            selected.append((int(start), end))
            if len(selected) == n_blocks:
                break
    if len(selected) != n_blocks:
        raise ValueError("No se pudieron encontrar bloques separados de validación")
    return sorted(selected)


def split_window_starts(
    starts: np.ndarray,
    validation_blocks: list[tuple[int, int]],
    window_size: int,
    purge_size: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split windows and embargo train windows around validation blocks."""
    train_starts: list[int] = []
    validation_starts: list[int] = []
    for start in starts:
        window_end = int(start + window_size - 1)
        belongs_to_validation = any(
            start >= block_start and window_end <= block_end
            for block_start, block_end in validation_blocks
        )
        overlaps_validation_or_purge = any(
            start <= block_end + purge_size and window_end >= block_start - purge_size
            for block_start, block_end in validation_blocks
        )
        if belongs_to_validation:
            validation_starts.append(int(start))
        elif not overlaps_validation_or_purge:
            train_starts.append(int(start))
    return np.asarray(train_starts, dtype=int), np.asarray(validation_starts, dtype=int)


def vae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta_kl: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-element reconstruction, KL and total losses."""
    reconstruction_loss = torch.mean((reconstruction - target) ** 2)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + beta_kl * kl_loss, reconstruction_loss, kl_loss


def _make_loader(
    windows: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(windows).float())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        drop_last=False,
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    beta_kl: float,
    device: torch.device,
    branch_loss_weights: tuple[float, float, float] | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "total_loss": 0.0,
        "reconstruction_loss": 0.0,
        "common_reconstruction_loss": 0.0,
        "common_total_loss": 0.0,
        "kl_loss": 0.0,
    }
    if branch_loss_weights is not None:
        totals.update({"market_loss": 0.0, "sector_loss": 0.0, "idio_loss": 0.0})
    n_samples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for (batch,) in loader:
            batch = batch.to(device)
            reconstruction, mu, logvar = model(batch)
            if branch_loss_weights is None:
                total, reconstruction_loss, kl_loss = vae_loss(
                    reconstruction, batch, mu, logvar, beta_kl=beta_kl
                )
            else:
                market_end = MultiBranchGRUVAE.n_market_features
                sector_end = market_end + MultiBranchGRUVAE.n_sector_features
                market_loss = torch.mean((reconstruction[:, :, :market_end] - batch[:, :, :market_end]) ** 2)
                sector_loss = torch.mean((reconstruction[:, :, market_end:sector_end] - batch[:, :, market_end:sector_end]) ** 2)
                idio_loss = torch.mean((reconstruction[:, :, sector_end:] - batch[:, :, sector_end:]) ** 2)
                reconstruction_loss = (
                    branch_loss_weights[0] * market_loss
                    + branch_loss_weights[1] * sector_loss
                    + branch_loss_weights[2] * idio_loss
                )
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                total = reconstruction_loss + beta_kl * kl_loss
            common_reconstruction_loss = torch.mean((reconstruction - batch) ** 2)
            common_total_loss = common_reconstruction_loss + beta_kl * kl_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            batch_size = batch.shape[0]
            n_samples += batch_size
            totals["total_loss"] += float(total.detach()) * batch_size
            totals["reconstruction_loss"] += float(reconstruction_loss.detach()) * batch_size
            totals["common_reconstruction_loss"] += float(common_reconstruction_loss.detach()) * batch_size
            totals["common_total_loss"] += float(common_total_loss.detach()) * batch_size
            totals["kl_loss"] += float(kl_loss.detach()) * batch_size
            if branch_loss_weights is not None:
                totals["market_loss"] += float(market_loss.detach()) * batch_size
                totals["sector_loss"] += float(sector_loss.detach()) * batch_size
                totals["idio_loss"] += float(idio_loss.detach()) * batch_size
    return {key: value / max(n_samples, 1) for key, value in totals.items()}


def train_vae(
    model: nn.Module,
    train_windows: np.ndarray,
    validation_windows: np.ndarray,
    *,
    epochs: int = 40,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    beta_kl: float = 1e-3,
    patience: int = 8,
    seed: int = 42,
    device: str | torch.device = "cpu",
    checkpoint_path: str | Path | None = None,
    branch_loss_weights: tuple[float, float, float] | None = None,
    selection_metric: str = "reconstruction_loss",
) -> tuple[dict[str, list[float]], int, float]:
    """Train with early stopping and return history, best epoch and score."""
    device = torch.device(device)
    model.to(device)
    train_loader = _make_loader(train_windows, batch_size, True, seed)
    validation_loader = _make_loader(validation_windows, batch_size, False, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    valid_selection_metrics = {"reconstruction_loss", "common_reconstruction_loss", "common_total_loss"}
    if selection_metric not in valid_selection_metrics:
        raise ValueError(f"selection_metric debe ser uno de {valid_selection_metrics}")
    history: dict[str, list[float]] = {
        "epoch": [],
        "train_total_loss": [],
        "val_total_loss": [],
        "train_reconstruction_loss": [],
        "val_reconstruction_loss": [],
        "train_common_reconstruction_loss": [],
        "val_common_reconstruction_loss": [],
        "train_common_total_loss": [],
        "val_common_total_loss": [],
        "train_kl_loss": [],
        "val_kl_loss": [],
    }
    if branch_loss_weights is not None:
        for branch in ("market", "sector", "idio"):
            history[f"train_{branch}_loss"] = []
            history[f"val_{branch}_loss"] = []
    best_score = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, beta_kl, device, branch_loss_weights)
        val_metrics = _run_epoch(model, validation_loader, None, beta_kl, device, branch_loss_weights)
        history["epoch"].append(epoch)
        for metric_name in ("total_loss", "reconstruction_loss", "kl_loss"):
            history[f"train_{metric_name}"].append(train_metrics[metric_name])
            history[f"val_{metric_name}"].append(val_metrics[metric_name])
        history["train_common_reconstruction_loss"].append(train_metrics["common_reconstruction_loss"])
        history["val_common_reconstruction_loss"].append(val_metrics["common_reconstruction_loss"])
        history["train_common_total_loss"].append(train_metrics["common_total_loss"])
        history["val_common_total_loss"].append(val_metrics["common_total_loss"])
        if branch_loss_weights is not None:
            for branch in ("market", "sector", "idio"):
                history[f"train_{branch}_loss"].append(train_metrics[f"{branch}_loss"])
                history[f"val_{branch}_loss"].append(val_metrics[f"{branch}_loss"])

        score = val_metrics[selection_metric]
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            if checkpoint_path is not None:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "best_epoch": best_epoch,
                        "best_validation_score": best_score,
                    },
                    checkpoint_path,
                )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    return history, best_epoch, best_score


def train_vae_full_data(
    model: nn.Module,
    windows: np.ndarray,
    *,
    epochs: int,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    beta_kl: float = 1e-3,
    seed: int = 42,
    device: str | torch.device = "cpu",
    checkpoint_path: str | Path | None = None,
    branch_loss_weights: tuple[float, float, float] | None = None,
) -> dict[str, list[float]]:
    """Refit a freshly initialized model on all windows without validation."""
    device = torch.device(device)
    model.to(device)
    loader = _make_loader(windows, batch_size, True, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: dict[str, list[float]] = {
        "epoch": [],
        "train_total_loss": [],
        "train_reconstruction_loss": [],
        "train_common_reconstruction_loss": [],
        "train_common_total_loss": [],
        "train_kl_loss": [],
    }
    if branch_loss_weights is not None:
        for branch in ("market", "sector", "idio"):
            history[f"train_{branch}_loss"] = []
    for epoch in range(1, epochs + 1):
        metrics = _run_epoch(model, loader, optimizer, beta_kl, device, branch_loss_weights)
        history["epoch"].append(epoch)
        for metric_name in ("total_loss", "reconstruction_loss", "kl_loss"):
            history[f"train_{metric_name}"].append(metrics[metric_name])
        history["train_common_reconstruction_loss"].append(metrics["common_reconstruction_loss"])
        history["train_common_total_loss"].append(metrics["common_total_loss"])
        if branch_loss_weights is not None:
            for branch in ("market", "sector", "idio"):
                history[f"train_{branch}_loss"].append(metrics[f"{branch}_loss"])
    if checkpoint_path is not None:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "epochs": epochs}, checkpoint_path)
    return history


def decode_latent_variants(
    model: GRUVAE,
    normalized_window: np.ndarray,
    alphas: Iterable[float],
    *,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> dict[float, np.ndarray]:
    """Decode one real episode along one shared latent perturbation direction."""
    device = torch.device(device)
    model.eval()
    model.to(device)
    with torch.no_grad():
        window_tensor = torch.from_numpy(normalized_window[None]).float().to(device)
        mu, logvar = model.encode(window_tensor)
        direction = torch.randn(mu.shape, generator=torch.Generator(device=device).manual_seed(seed), device=device)
        sigma = torch.exp(0.5 * logvar)
        outputs = {}
        for alpha in alphas:
            z = mu + float(alpha) * sigma * direction
            outputs[float(alpha)] = model.decode(z, normalized_window.shape[0]).cpu().numpy()[0]
    return outputs


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Save JSON configs with NumPy scalar compatibility."""
    def convert(value: Any) -> Any:
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=convert), encoding="utf-8")
