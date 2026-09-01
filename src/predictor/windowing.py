"""
Construcción de ejemplos de entrenamiento para el predictor de la matriz
de covarianzas: cada ejemplo es (ventana histórica de retornos, matriz de
covarianzas REAL calculada sobre el periodo siguiente).

Terminología:
    history_days : tamaño de la ventana de entrada (el "pasado" que ve el
                   modelo). 1 año de bolsa ≈ 252 días.
    horizon_days : tamaño del periodo a predecir (el "futuro"). 2 meses de
                   bolsa ≈ 42 días.
    step_days    : cada cuántos días se desliza la ventana para generar el
                   siguiente ejemplo. Con step_days pequeño (p.ej. 1) los
                   ejemplos se solapan mucho entre sí (casi el mismo dato
                   repetido); con step_days más grande (p.ej. 21, ~1 mes)
                   los ejemplos son más independientes, a costa de tener
                   menos ejemplos en total.
"""

import numpy as np
import pandas as pd


def build_covariance_windows(
    returns: pd.DataFrame,
    history_days: int = 252,
    horizon_days: int = 42,
    step_days: int = 21,
) -> dict:
    """
    Trocea el histórico de retornos en ejemplos (ventana pasada -> Sigma
    futura realizada).

    Parameters
    ----------
    returns : pd.DataFrame
        Retornos diarios, índice=fecha, columnas=símbolo.
    history_days : int
        Longitud de la ventana histórica de entrada, en días de bolsa.
    horizon_days : int
        Longitud del periodo futuro sobre el que se calcula la Sigma
        objetivo, en días de bolsa.
    step_days : int
        Paso entre ejemplos consecutivos, en días de bolsa.

    Returns
    -------
    dict con:
        'X': lista de np.ndarray, cada uno de shape (history_days, n_assets)
             — la ventana histórica de retornos de cada ejemplo.
        'y': lista de np.ndarray, cada uno de shape (n_assets, n_assets)
             — la matriz de covarianzas real del periodo siguiente.
        'cutoff_dates': lista de fechas — el "punto de corte" de cada
             ejemplo (el día en que termina la ventana histórica y empieza
             el periodo a predecir). Útil para hacer un split train/test
             respetando el orden temporal (nunca mezclar pasado y futuro).
    """
    values = returns.values
    dates = returns.index
    n_days, n_assets = values.shape

    X_list = []
    y_list = []
    cutoff_dates = []

    # El primer ejemplo posible necesita history_days hacia atrás y
    # horizon_days hacia adelante desde el punto de corte.
    start = history_days
    end = n_days - horizon_days

    for cutoff_idx in range(start, end, step_days):
        history_window = values[cutoff_idx - history_days: cutoff_idx]
        future_window = values[cutoff_idx: cutoff_idx + horizon_days]

        sigma_real = np.cov(future_window, rowvar=False)

        X_list.append(history_window)
        y_list.append(sigma_real)
        cutoff_dates.append(dates[cutoff_idx])

    return {
        "X": X_list,
        "y": y_list,
        "cutoff_dates": cutoff_dates,
    }


def summarize_windows(windows: dict, n_assets: int) -> dict:
    """
    Pequeño resumen de las ventanas construidas, útil para comprobar que
    todo tiene el shape esperado antes de seguir.
    """
    n_examples = len(windows["X"])
    history_days = windows["X"][0].shape[0] if n_examples > 0 else 0

    return {
        "n_ejemplos": n_examples,
        "shape_X_por_ejemplo": (history_days, n_assets),
        "shape_y_por_ejemplo": (n_assets, n_assets),
        "primera_fecha_corte": windows["cutoff_dates"][0] if n_examples > 0 else None,
        "ultima_fecha_corte": windows["cutoff_dates"][-1] if n_examples > 0 else None,
    }
