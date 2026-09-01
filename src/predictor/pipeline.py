"""
Ejecuta el pipeline completo del predictor (split, escalado, baseline, GRU,
LSTM, evaluación) para una configuración concreta de ventanas
(history_days, horizon_days, step_days). Se usa para poder repetir todo el
proceso con varias configuraciones distintas y comparar resultados en una
única tabla, en vez de rehacer el trabajo a mano para cada una.
"""

import numpy as np
import pandas as pd
import torch

from src.predictor.windowing import build_covariance_windows
from src.predictor.split import temporal_split_purged, fit_scaler, apply_scaler
from src.predictor.baseline import baseline_predict, evaluate_predictions
from src.predictor.models import RecurrentCovariancePredictor, init_output_bias_from_data
from src.predictor.train import train_predictor
from src.predictor.cholesky import cholesky_vector_length
from src.predictor.cholesky_torch import cholesky_vector_to_sigma_torch


def run_pipeline(
    returns_real: pd.DataFrame,
    history_days: int,
    horizon_days: int,
    step_days: int,
    hidden_size: int = 16,
    dropout: float = 0.2,
    max_epochs: int = 1000,
    patience: int = 150,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """
    Ejecuta el pipeline completo para una configuración de ventanas dada.

    Returns
    -------
    dict con:
        'config': los parámetros usados (para identificar la fila en la tabla comparativa)
        'n_ejemplos_total', 'n_train', 'n_val', 'n_test'
        'baseline_mse', 'gru_mse', 'lstm_mse' (MSE medio de test de cada candidato)
        'gru_best_epoch', 'lstm_best_epoch'
    """
    windows = build_covariance_windows(
        returns_real, history_days=history_days, horizon_days=horizon_days, step_days=step_days
    )
    n_assets = returns_real.shape[1]

    if len(windows["X"]) < 15:
        # Muy pocos ejemplos para hacer un split en 3 partes con sentido
        return {
            "config": {"history_days": history_days, "horizon_days": horizon_days, "step_days": step_days},
            "n_ejemplos_total": len(windows["X"]),
            "error": "Muy pocos ejemplos para entrenar (< 15)",
        }

    split_test = temporal_split_purged(
        windows, test_fraction=0.2,
        history_days=history_days, horizon_days=horizon_days, step_days=step_days,
    )
    windows_train_only = {
        "X": split_test["X_train"], "y": split_test["y_train"], "cutoff_dates": split_test["dates_train"],
    }
    split_val = temporal_split_purged(
        windows_train_only, test_fraction=0.2,
        history_days=history_days, horizon_days=horizon_days, step_days=step_days,
    )

    X_train, y_train = split_val["X_train"], split_val["y_train"]
    X_val, y_val = split_val["X_test"], split_val["y_test"]
    X_test, y_test = split_test["X_test"], split_test["y_test"]

    if len(X_train) < 5 or len(X_val) < 3:
        return {
            "config": {"history_days": history_days, "horizon_days": horizon_days, "step_days": step_days},
            "n_ejemplos_total": len(windows["X"]),
            "error": f"Tras la purga quedan muy pocos ejemplos (train={len(X_train)}, val={len(X_val)})",
        }

    scaler = fit_scaler(X_train)
    X_train_scaled = apply_scaler(X_train, scaler)
    X_val_scaled = apply_scaler(X_val, scaler)
    X_test_scaled = apply_scaler(X_test, scaler)

    # Baseline
    baseline_preds = baseline_predict(X_test)
    baseline_mse = evaluate_predictions(baseline_preds, y_test)["mse_medio"]

    output_dim = cholesky_vector_length(n_assets)

    # GRU
    torch.manual_seed(seed)
    gru_model = RecurrentCovariancePredictor(
        n_assets=n_assets, output_dim=output_dim, cell_type="gru", hidden_size=hidden_size, dropout=dropout,
    )
    init_output_bias_from_data(gru_model, y_train, n_assets)
    gru_result = train_predictor(
        gru_model, X_train_scaled, y_train, X_val_scaled, y_val,
        n_assets=n_assets, max_epochs=max_epochs, patience=patience, device=device,
    )
    gru_preds = _predict_sigmas(gru_model, X_test_scaled, n_assets, device)
    gru_mse = evaluate_predictions(gru_preds, y_test)["mse_medio"]

    # LSTM
    torch.manual_seed(seed)
    lstm_model = RecurrentCovariancePredictor(
        n_assets=n_assets, output_dim=output_dim, cell_type="lstm", hidden_size=hidden_size, dropout=dropout,
    )
    init_output_bias_from_data(lstm_model, y_train, n_assets)
    lstm_result = train_predictor(
        lstm_model, X_train_scaled, y_train, X_val_scaled, y_val,
        n_assets=n_assets, max_epochs=max_epochs, patience=patience, device=device,
    )
    lstm_preds = _predict_sigmas(lstm_model, X_test_scaled, n_assets, device)
    lstm_mse = evaluate_predictions(lstm_preds, y_test)["mse_medio"]

    return {
        "config": {"history_days": history_days, "horizon_days": horizon_days, "step_days": step_days},
        "n_ejemplos_total": len(windows["X"]),
        "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
        "baseline_mse": baseline_mse,
        "gru_mse": gru_mse, "gru_best_epoch": gru_result["best_epoch"],
        "lstm_mse": lstm_mse, "lstm_best_epoch": lstm_result["best_epoch"],
    }


def _predict_sigmas(model, X_scaled, n_assets, device):
    model.eval()
    X_t = torch.tensor(np.stack(X_scaled), dtype=torch.float32, device=device)
    with torch.no_grad():
        vectors = model(X_t)
        sigmas = cholesky_vector_to_sigma_torch(vectors, n=n_assets)
    return [s.cpu().numpy() for s in sigmas]


def run_all_configs(returns_real: pd.DataFrame, configs: list, **kwargs) -> pd.DataFrame:
    """
    Ejecuta run_pipeline para una lista de configuraciones
    [(history_days, horizon_days, step_days), ...] y devuelve un DataFrame
    comparativo, una fila por configuración.
    """
    rows = []
    for history_days, horizon_days, step_days in configs:
        print(f"Ejecutando: history={history_days}, horizon={horizon_days}, step={step_days}...")
        result = run_pipeline(returns_real, history_days, horizon_days, step_days, **kwargs)

        row = {**result["config"], "n_ejemplos_total": result["n_ejemplos_total"]}
        if "error" in result:
            row["error"] = result["error"]
        else:
            row.update({
                "n_train": result["n_train"], "n_val": result["n_val"], "n_test": result["n_test"],
                "baseline_mse": result["baseline_mse"],
                "gru_mse": result["gru_mse"], "gru_best_epoch": result["gru_best_epoch"],
                "lstm_mse": result["lstm_mse"], "lstm_best_epoch": result["lstm_best_epoch"],
            })
        rows.append(row)

    return pd.DataFrame(rows)
