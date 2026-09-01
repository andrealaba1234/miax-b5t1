"""
Búsqueda de la mejor configuración de ventanas (history_days, horizon_days)
usando el modelo FINAL combinado (tamaño de activos variable + conexión
residual al baseline) — no el modelo simple usado en la búsqueda inicial
del Paso 3. La configuración óptima para un modelo puede no serlo para
otro, así que esta búsqueda se repite específicamente sobre la
arquitectura final.

Se fija step_days=5 (no se repite la exploración de step_days: ya se
comprobó que step_days=1 es computacionalmente inviable con este tipo de
entrenamiento ejemplo a ejemplo, y step_days=5 fue consistentemente de las
configuraciones con mejor resultado en la búsqueda del Paso 3).

Para que la búsqueda sea viable en tiempo, se usa un presupuesto de épocas
reducido en esta fase (search_max_epochs, search_patience) — solo para
COMPARAR configuraciones entre sí. La configuración ganadora se reentrena
después con el presupuesto de épocas completo para el resultado final.
"""

import time
import numpy as np
import pandas as pd
import torch

from src.predictor.windowing import build_covariance_windows
from src.predictor.asset_subsets import build_variable_asset_windows
from src.predictor.split import temporal_split_purged, fit_scaler, apply_scaler
from src.predictor.baseline import baseline_predict, evaluate_predictions
from src.predictor.factor_model import AssetFactorEncoder, init_factor_encoder_near_baseline
from src.predictor.train_factor import train_predictor_factor_residual, predict_sigma_factor_residual


def _prepare_combined_data(
    returns_real: pd.DataFrame,
    history_days: int,
    horizon_days: int,
    step_days: int,
    n_assets: int,
    n_subsets_per_example: int = 5,
    seed: int = 42,
):
    windows = build_covariance_windows(
        returns_real, history_days=history_days, horizon_days=horizon_days, step_days=step_days,
    )
    var_windows = build_variable_asset_windows(
        windows, n_assets=n_assets, n_subsets_per_example=n_subsets_per_example, seed=seed,
    )

    split_test = temporal_split_purged(
        var_windows, test_fraction=0.2,
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

    baseline_train = baseline_predict(X_train)
    baseline_val = baseline_predict(X_val)
    baseline_test = baseline_predict(X_test)

    return {
        "X_train": X_train, "y_train": y_train, "baseline_train": baseline_train,
        "X_val": X_val, "y_val": y_val, "baseline_val": baseline_val,
        "X_test": X_test, "y_test": y_test, "baseline_test": baseline_test,
    }


def run_combined_config(
    returns_real: pd.DataFrame,
    history_days: int,
    horizon_days: int,
    step_days: int,
    n_assets: int,
    max_epochs: int,
    patience: int,
    n_subsets_per_example: int = 5,
    n_factors: int = 4,
    hidden_size: int = 16,
    dropout: float = 0.25,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """
    Prepara los datos y entrena el modelo combinado (tamaño variable +
    residual) para una única configuración de ventanas. Devuelve el
    resultado junto con los datos preparados, por si se quiere reutilizar
    para un reentrenamiento posterior con más épocas.
    """
    data = _prepare_combined_data(
        returns_real, history_days, horizon_days, step_days, n_assets, n_subsets_per_example, seed,
    )

    if len(data["X_train"]) < 5 or len(data["X_val"]) < 3:
        return {
            "config": {"history_days": history_days, "horizon_days": horizon_days, "step_days": step_days},
            "error": "Muy pocos ejemplos tras la purga",
        }

    baseline_mse = evaluate_predictions(data["baseline_test"], data["y_test"])["mse_medio"]

    torch.manual_seed(seed)
    model = AssetFactorEncoder(n_factors=n_factors, cell_type="gru", hidden_size=hidden_size, dropout=dropout)
    init_factor_encoder_near_baseline(model, baseline_train=data["baseline_train"])

    start = time.time()
    result = train_predictor_factor_residual(
        model, data["X_train"], data["y_train"], data["baseline_train"],
        data["X_val"], data["y_val"], data["baseline_val"],
        max_epochs=max_epochs, patience=patience, device=device,
    )
    elapsed_min = (time.time() - start) / 60

    preds_test = [
        predict_sigma_factor_residual(model, x, b, device=device)
        for x, b in zip(data["X_test"], data["baseline_test"])
    ]
    model_mse = evaluate_predictions(preds_test, data["y_test"])["mse_medio"]

    return {
        "config": {"history_days": history_days, "horizon_days": horizon_days, "step_days": step_days},
        "n_train": len(data["X_train"]), "n_val": len(data["X_val"]), "n_test": len(data["X_test"]),
        "baseline_mse": baseline_mse,
        "model_mse": model_mse,
        "best_epoch": result["best_epoch"],
        "elapsed_min": round(elapsed_min, 1),
        "model": model,
        "data": data,
    }


def search_combined_configs(
    returns_real: pd.DataFrame,
    configs: list,
    n_assets: int,
    search_max_epochs: int = 30,
    search_patience: int = 10,
    **kwargs,
) -> pd.DataFrame:
    """
    Ejecuta run_combined_config para varias configuraciones (history_days,
    horizon_days), con step_days=5 fijo, usando un presupuesto de épocas
    REDUCIDO (search_max_epochs/search_patience) — solo para comparar
    configuraciones rápido, no para el entrenamiento final.

    Returns
    -------
    pd.DataFrame con una fila por configuración (sin las claves 'model' ni
    'data', para que la tabla sea legible).
    """
    rows = []
    for history_days, horizon_days in configs:
        print(f"Probando: history_days={history_days}, horizon_days={horizon_days}, step_days=5...")
        result = run_combined_config(
            returns_real, history_days, horizon_days, step_days=5, n_assets=n_assets,
            max_epochs=search_max_epochs, patience=search_patience, **kwargs,
        )
        row = {k: v for k, v in result.items() if k not in ("model", "data")}
        row.update(row.pop("config", {}))
        rows.append(row)
        print(f"  -> baseline_mse={row.get('baseline_mse')}, model_mse={row.get('model_mse')}, "
              f"best_epoch={row.get('best_epoch')}, tardó {row.get('elapsed_min')} min")

    return pd.DataFrame(rows)
