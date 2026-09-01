"""
Bucle de entrenamiento para RecurrentCovariancePredictor, con early
stopping (mismo presupuesto acordado para todos los modelos generativos y
el predictor: máximo 1000 épocas, paciencia 150).
"""

from typing import List
import numpy as np
import torch
import torch.nn as nn

from src.predictor.cholesky_torch import cholesky_vector_to_sigma_torch


def _to_tensor_batch(X: List[np.ndarray], y: List[np.ndarray], device: str):
    X_tensor = torch.tensor(np.stack(X), dtype=torch.float32, device=device)
    y_tensor = torch.tensor(np.stack(y), dtype=torch.float32, device=device)
    return X_tensor, y_tensor


def train_predictor_residual(
    model: nn.Module,
    X_train: List[np.ndarray],
    y_train: List[np.ndarray],
    baseline_vectors_train: List[np.ndarray],
    X_val: List[np.ndarray],
    y_val: List[np.ndarray],
    baseline_vectors_val: List[np.ndarray],
    n_assets: int,
    max_epochs: int = 1000,
    patience: int = 150,
    learning_rate: float = 0.0005,
    device: str = "cpu",
) -> dict:
    """
    Igual que train_predictor, pero con conexión residual al baseline: la
    red predice una CORRECCIÓN sobre el vector Cholesky del baseline (no la
    Sigma completa), y la predicción final es baseline + corrección, antes
    de reconstruir con cholesky_vector_to_sigma_torch. Con el modelo
    inicializado a cero (init_output_layer_to_zero), la corrección inicial
    es nula y la primera predicción coincide exactamente con el baseline.

    Returns
    -------
    Igual que train_predictor.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_train_t, y_train_t = _to_tensor_batch(X_train, y_train, device)
    X_val_t, y_val_t = _to_tensor_batch(X_val, y_val, device)
    baseline_train_t = torch.tensor(np.stack(baseline_vectors_train), dtype=torch.float32, device=device)
    baseline_val_t = torch.tensor(np.stack(baseline_vectors_val), dtype=torch.float32, device=device)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()

        correction = model(X_train_t)
        pred_vector = baseline_train_t + correction
        pred_sigma = cholesky_vector_to_sigma_torch(pred_vector, n=n_assets)
        train_loss = torch.mean((pred_sigma - y_train_t) ** 2)

        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            correction_val = model(X_val_t)
            pred_vector_val = baseline_val_t + correction_val
            pred_sigma_val = cholesky_vector_to_sigma_torch(pred_vector_val, n=n_assets)
            val_loss = torch.mean((pred_sigma_val - y_val_t) ** 2).item()

        history.append({"epoch": epoch, "train_loss": train_loss.item(), "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"history": history, "best_epoch": best_epoch, "best_val_loss": best_val_loss}


def predict_sigmas_residual(model, X_scaled, baseline_vectors, n_assets, device):
    """
    Predicción con conexión residual: suma el vector del baseline a la
    corrección predicha por la red antes de reconstruir Sigma.
    """
    model.eval()
    X_t = torch.tensor(np.stack(X_scaled), dtype=torch.float32, device=device)
    baseline_t = torch.tensor(np.stack(baseline_vectors), dtype=torch.float32, device=device)
    with torch.no_grad():
        correction = model(X_t)
        pred_vector = baseline_t + correction
        sigmas = cholesky_vector_to_sigma_torch(pred_vector, n=n_assets)
    return [s.cpu().numpy() for s in sigmas]


def train_predictor(
    model: nn.Module,
    X_train: List[np.ndarray],
    y_train: List[np.ndarray],
    X_val: List[np.ndarray],
    y_val: List[np.ndarray],
    n_assets: int,
    max_epochs: int = 1000,
    patience: int = 150,
    learning_rate: float = 0.0005,
    device: str = "cpu",
) -> dict:
    """
    Entrena el modelo minimizando el MSE entre la Sigma reconstruida
    (a partir del vector de Cholesky predicho) y la Sigma real, con early
    stopping sobre el error de validación.

    Returns
    -------
    dict con 'history' (lista de errores train/val por época) y
    'best_epoch' (época en la que se guardó el mejor modelo; el modelo
    recibido queda con esos pesos cargados al finalizar).
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_train_t, y_train_t = _to_tensor_batch(X_train, y_train, device)
    X_val_t, y_val_t = _to_tensor_batch(X_val, y_val, device)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()

        pred_vector = model(X_train_t)
        pred_sigma = cholesky_vector_to_sigma_torch(pred_vector, n=n_assets)
        train_loss = torch.mean((pred_sigma - y_train_t) ** 2)

        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_vector_val = model(X_val_t)
            pred_sigma_val = cholesky_vector_to_sigma_torch(pred_vector_val, n=n_assets)
            val_loss = torch.mean((pred_sigma_val - y_val_t) ** 2).item()

        history.append({"epoch": epoch, "train_loss": train_loss.item(), "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"history": history, "best_epoch": best_epoch, "best_val_loss": best_val_loss}
