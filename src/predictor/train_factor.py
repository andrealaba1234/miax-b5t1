"""
Bucle de entrenamiento para AssetFactorEncoder (modelo de tamaño de
activos variable). A diferencia de train_predictor (train.py), aquí no se
puede procesar todos los ejemplos de entrenamiento a la vez en un único
tensor, porque cada ejemplo puede tener un número distinto de activos
(forma "irregular", no se pueden apilar en un solo tensor rectangular).
Por eso se entrena ejemplo a ejemplo: en cada época se recorren todos los
ejemplos de entrenamiento, uno a uno, actualizando los pesos tras cada
uno (descenso de gradiente estocástico), y se promedia el error de
validación de la misma forma.
"""

from typing import List
import csv
import os
import numpy as np
import torch
import torch.nn as nn

from src.predictor.factor_model import reconstruct_sigma_factor_torch


def _forward_example(model: nn.Module, X: np.ndarray, device: str) -> torch.Tensor:
    """
    X: shape (history_days, n_assets_del_ejemplo). Se reorganiza a
    (n_assets, history_days, 1) para que el modelo trate cada activo como
    un elemento del lote.
    """
    n_assets = X.shape[1]
    X_t = torch.tensor(X.T, dtype=torch.float32, device=device).unsqueeze(-1)  # (n_assets, history_days, 1)
    factor_output = model(X_t)  # (n_assets, n_factors + 1)
    return reconstruct_sigma_factor_torch(factor_output)


def train_predictor_factor(
    model: nn.Module,
    X_train: List[np.ndarray],
    y_train: List[np.ndarray],
    X_val: List[np.ndarray],
    y_val: List[np.ndarray],
    max_epochs: int = 1000,
    patience: int = 150,
    learning_rate: float = 0.0002,
    weight_decay: float = 0.0,
    batch_size: int = 8,
    grad_clip_norm: float = 1.0,
    device: str = "cpu",
    verbose_every: int = 0,
    history_csv_path: str = None,
) -> dict:
    """
    Entrena AssetFactorEncoder ejemplo a ejemplo (cada ejemplo puede tener
    un número de activos distinto, así que no se pueden apilar en un único
    tensor rectangular), con early stopping sobre el error de validación
    promedio.

    **Mini-batch por acumulación de gradiente:** aunque no se puede formar
    un batch real (tensores de tamaños distintos), sí se puede acumular el
    gradiente de `batch_size` ejemplos antes de cada `optimizer.step()` —
    se promedia la pérdida de ese grupo y se hace backward una sola vez.
    Sin esto (`batch_size=1`, actualizar tras cada ejemplo), el ruido de
    cada actualización es muy alto porque los subconjuntos de activos
    (ver `build_variable_asset_windows`) tienen escalas de covarianza muy
    distintas entre sí, y Adam amplifica ese ruido — la curva oscila en
    vez de converger, sin que sea un problema de capacidad del modelo.
    `grad_clip_norm` acota además cualquier gradiente puntual extremo (un
    ejemplo con covarianzas mucho más grandes que el resto del batch).

    Parameters
    ----------
    verbose_every : int
        Si > 0, imprime el progreso cada `verbose_every` épocas (útil para
        confirmar que el entrenamiento avanza, dado que puede tardar más
        que train_predictor al procesar ejemplo a ejemplo).
    history_csv_path : str, optional
        Si se indica, guarda una fila en este CSV al final de CADA época
        (no solo al terminar el entrenamiento). Así, si se interrumpe la
        ejecución (por ejemplo con el botón de stop de Jupyter, dado que
        este entrenamiento puede tardar horas), el historial hasta ese
        punto queda disponible en disco de todas formas — sin este
        guardado incremental, interrumpir la celda pierde TODO el
        historial, porque la función solo devuelve el resultado al
        terminar el bucle completo.

    Returns
    -------
    dict con 'history' (train_loss/val_loss promedio por época) y
    'best_epoch'.
    """
    model.to(device)
    # weight_decay solo sobre pesos, no sobre sesgos: el sesgo de la capa de
    # salida puede llevar una calibración de escala deliberadamente no-nula
    # (ver init_factor_encoder_near_baseline), y penalizarlo hacia cero
    # deshace esa calibración en vez de limitarse a frenar el sobreajuste.
    weight_params = [p for n, p in model.named_parameters() if "bias" not in n]
    bias_params = [p for n, p in model.named_parameters() if "bias" in n]
    optimizer = torch.optim.Adam(
        [
            {"params": weight_params, "weight_decay": weight_decay},
            {"params": bias_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    if history_csv_path is not None:
        os.makedirs(os.path.dirname(history_csv_path), exist_ok=True)
        with open(history_csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss"])

    n_train = len(X_train)
    rng = np.random.default_rng(0)

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(n_train)
        train_losses = []

        for batch_start in range(0, n_train, batch_size):
            batch_indices = order[batch_start:batch_start + batch_size]
            optimizer.zero_grad()
            batch_losses = []
            for i in batch_indices:
                pred_sigma = _forward_example(model, X_train[i], device)
                y_t = torch.tensor(y_train[i], dtype=torch.float32, device=device)
                batch_losses.append(torch.mean((pred_sigma - y_t) ** 2))
            batch_loss = torch.stack(batch_losses).mean()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            train_losses.extend(l.item() for l in batch_losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for j in range(len(X_val)):
                pred_sigma_val = _forward_example(model, X_val[j], device)
                y_val_t = torch.tensor(y_val[j], dtype=torch.float32, device=device)
                val_losses.append(torch.mean((pred_sigma_val - y_val_t) ** 2).item())

        train_loss_epoch = float(np.mean(train_losses))
        val_loss_epoch = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": train_loss_epoch, "val_loss": val_loss_epoch})

        if history_csv_path is not None:
            with open(history_csv_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, train_loss_epoch, val_loss_epoch])

        if verbose_every and epoch % verbose_every == 0:
            print(f"Época {epoch}: train_loss={train_loss_epoch:.4e}, val_loss={val_loss_epoch:.4e}")

        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
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


def predict_sigma_factor(model: nn.Module, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Predicción para un único ejemplo (una ventana con su número de activos concreto)."""
    model.eval()
    with torch.no_grad():
        pred_sigma = _forward_example(model, X, device)
    return pred_sigma.cpu().numpy()


def _forward_example_residual(model: nn.Module, X: np.ndarray, baseline_sigma: np.ndarray, device: str) -> torch.Tensor:
    from src.predictor.factor_model import reconstruct_sigma_factor_residual_torch

    n_assets = X.shape[1]
    X_t = torch.tensor(X.T, dtype=torch.float32, device=device).unsqueeze(-1)
    factor_output = model(X_t)
    baseline_t = torch.tensor(baseline_sigma, dtype=torch.float32, device=device)
    return reconstruct_sigma_factor_residual_torch(factor_output, baseline_t)


def train_predictor_factor_residual(
    model: nn.Module,
    X_train: List[np.ndarray],
    y_train: List[np.ndarray],
    baseline_train: List[np.ndarray],
    X_val: List[np.ndarray],
    y_val: List[np.ndarray],
    baseline_val: List[np.ndarray],
    max_epochs: int = 1000,
    patience: int = 150,
    learning_rate: float = 0.0002,
    weight_decay: float = 0.001,
    lr_step_size: int = 30,
    lr_gamma: float = 0.5,
    min_learning_rate: float = 1e-6,
    batch_size: int = 8,
    grad_clip_norm: float = 1.0,
    device: str = "cpu",
    verbose_every: int = 0,
    history_csv_path: str = None,
) -> dict:
    """
    Igual que train_predictor_factor, pero con conexión residual al
    baseline: Sigma_final = Sigma_baseline + (F @ F.T + diag(idio^2)).
    Combina los 3 cambios acordados: purga en el split (aplicada antes,
    al construir X_train/X_val), tamaño de activos variable (esta
    arquitectura), y conexión residual al baseline (este entrenamiento).

    El mismo mini-batch por acumulación de gradiente y `grad_clip_norm`
    que en train_predictor_factor aplican aquí, por el mismo motivo (ver
    docstring de esa función).

    `weight_decay` y el calendario de `learning_rate` (se reduce a la
    mitad cada `lr_step_size` épocas hasta un suelo de
    `min_learning_rate`) atacan el sobreajuste temprano confirmado tras
    arreglar el punto de silla de la inicialización: una vez el gradiente
    fluye de verdad, el train_loss mejora de forma sostenida pero el
    val_loss deja de acompañarlo pasadas unas pocas decenas de épocas
    (visto en la práctica: mejor época estancada en la 27 incluso con
    150 disponibles). Es la misma receta que usa Miriam en el generador
    autorregresivo (notebook 6) para el mismo síntoma.

    Parameters
    ----------
    baseline_train, baseline_val : list de np.ndarray
        La covarianza de la ventana de entrada de cada ejemplo (el
        baseline), alineada con X_train/X_val respectivamente. Se puede
        calcular con baseline_predict(X) de src/predictor/baseline.py.

    Returns
    -------
    Igual que train_predictor_factor.
    """
    model.to(device)
    # weight_decay solo sobre pesos, no sobre sesgos: mismo motivo que en
    # train_predictor_factor (ver comentario ahí) — output_layer.bias lleva
    # la calibración de escala de init_factor_encoder_near_baseline.
    weight_params = [p for n, p in model.named_parameters() if "bias" not in n]
    bias_params = [p for n, p in model.named_parameters() if "bias" in n]
    optimizer = torch.optim.Adam(
        [
            {"params": weight_params, "weight_decay": weight_decay},
            {"params": bias_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    if history_csv_path is not None:
        os.makedirs(os.path.dirname(history_csv_path), exist_ok=True)
        with open(history_csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "learning_rate"])

    n_train = len(X_train)
    rng = np.random.default_rng(0)

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(n_train)
        train_losses = []

        for batch_start in range(0, n_train, batch_size):
            batch_indices = order[batch_start:batch_start + batch_size]
            optimizer.zero_grad()
            batch_losses = []
            for i in batch_indices:
                pred_sigma = _forward_example_residual(model, X_train[i], baseline_train[i], device)
                y_t = torch.tensor(y_train[i], dtype=torch.float32, device=device)
                batch_losses.append(torch.mean((pred_sigma - y_t) ** 2))
            batch_loss = torch.stack(batch_losses).mean()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            train_losses.extend(l.item() for l in batch_losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for j in range(len(X_val)):
                pred_sigma_val = _forward_example_residual(model, X_val[j], baseline_val[j], device)
                y_val_t = torch.tensor(y_val[j], dtype=torch.float32, device=device)
                val_losses.append(torch.mean((pred_sigma_val - y_val_t) ** 2).item())

        train_loss_epoch = float(np.mean(train_losses))
        val_loss_epoch = float(np.mean(val_losses))
        current_lr = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": epoch, "train_loss": train_loss_epoch, "val_loss": val_loss_epoch,
            "learning_rate": current_lr,
        })

        if history_csv_path is not None:
            with open(history_csv_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, train_loss_epoch, val_loss_epoch, current_lr])

        if verbose_every and epoch % verbose_every == 0:
            print(f"Época {epoch}: train_loss={train_loss_epoch:.4e}, val_loss={val_loss_epoch:.4e}, lr={current_lr:.2e}")

        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

        if current_lr > min_learning_rate:
            scheduler.step()
            for group in optimizer.param_groups:
                group["lr"] = max(group["lr"], min_learning_rate)

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"history": history, "best_epoch": best_epoch, "best_val_loss": best_val_loss}


def predict_sigma_factor_residual(model: nn.Module, X: np.ndarray, baseline_sigma: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Predicción con conexión residual para un único ejemplo."""
    model.eval()
    with torch.no_grad():
        pred_sigma = _forward_example_residual(model, X, baseline_sigma, device)
    return pred_sigma.cpu().numpy()
