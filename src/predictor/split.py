"""
Split temporal de los ejemplos de entrenamiento del predictor.

A diferencia de un split aleatorio (habitual en muchos problemas de ML),
aquí el orden temporal es obligatorio: los ejemplos de test deben ser
siempre posteriores a los de entrenamiento. Mezclar al azar permitiría que
el modelo "viera" indirectamente información de fechas posteriores a las
que se usan para evaluarlo, lo cual no sería representativo de un uso real
(en producción, el modelo solo tiene acceso al pasado, nunca al futuro).
"""

from typing import List
import numpy as np


def temporal_split(windows: dict, test_fraction: float = 0.2) -> dict:
    """
    Divide los ejemplos en entrenamiento (los más antiguos) y test (los
    más recientes), preservando el orden cronológico.

    Parameters
    ----------
    windows : dict
        Salida de build_covariance_windows (con claves 'X', 'y', 'cutoff_dates').
    test_fraction : float
        Proporción de ejemplos (los más recientes) reservados para test.

    Returns
    -------
    dict con 'X_train', 'y_train', 'dates_train', 'X_test', 'y_test', 'dates_test'.
    """
    n = len(windows["X"])
    n_test = max(1, int(round(n * test_fraction)))
    n_train = n - n_test

    return {
        "X_train": windows["X"][:n_train],
        "y_train": windows["y"][:n_train],
        "dates_train": windows["cutoff_dates"][:n_train],
        "X_test": windows["X"][n_train:],
        "y_test": windows["y"][n_train:],
        "dates_test": windows["cutoff_dates"][n_train:],
    }


def temporal_split_purged(
    windows: dict,
    test_fraction: float,
    history_days: int,
    horizon_days: int,
    step_days: int,
) -> dict:
    """
    Igual que temporal_split, pero eliminando además los ejemplos de
    entrenamiento que estén demasiado cerca de la frontera con el test
    (una técnica llamada "purga", habitual en validación de series
    temporales con ventanas solapadas).

    Motivo: aunque el solapamiento ENTRE ejemplos del mismo conjunto
    (todos en train, o todos en test) no es un problema — es la práctica
    habitual para generar más ejemplos —, si un ejemplo de train y uno de
    test comparten casi todos sus días históricos (por estar justo al
    lado en el tiempo), el modelo estaría prácticamente "viendo" el
    contenido del ejemplo de test durante el entrenamiento, lo cual
    infla artificialmente el resultado de la evaluación.

    Cada ejemplo abarca desde (cutoff - history_days) hasta
    (cutoff + horizon_days). Dos ejemplos dejan de solaparse cuando sus
    fechas de corte distan al menos history_days + horizon_days días de
    bolsa. Se eliminan de train los últimos ejemplos que queden dentro de
    esa distancia respecto al primer ejemplo de test.

    Parameters
    ----------
    history_days, horizon_days, step_days : int
        Los mismos parámetros usados para construir `windows`
        (build_covariance_windows), necesarios para calcular el tamaño de
        la zona de purga.

    Returns
    -------
    Mismo formato que temporal_split.
    """
    base_split = temporal_split(windows, test_fraction=test_fraction)

    purge_trading_days = history_days + horizon_days
    n_purge_examples = int(np.ceil(purge_trading_days / step_days))

    n_train_original = len(base_split["X_train"])
    n_train_purged = max(0, n_train_original - n_purge_examples)

    return {
        "X_train": base_split["X_train"][:n_train_purged],
        "y_train": base_split["y_train"][:n_train_purged],
        "dates_train": base_split["dates_train"][:n_train_purged],
        "X_test": base_split["X_test"],
        "y_test": base_split["y_test"],
        "dates_test": base_split["dates_test"],
        "n_purged": n_train_original - n_train_purged,
    }


def fit_scaler(X_train: List[np.ndarray]) -> dict:
    """
    Calcula media y desviación estándar por columna (activo), usando
    ÚNICAMENTE los datos de entrenamiento — nunca los de test, para no
    filtrar información del futuro al proceso de normalización.

    Returns
    -------
    dict con 'mean' y 'std', cada uno un array de shape (n_assets,).
    """
    stacked = np.concatenate(X_train, axis=0)  # (n_train * history_days, n_assets)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std[std == 0] = 1.0  # salvaguarda por si algún activo tuviera varianza nula
    return {"mean": mean, "std": std}


def apply_scaler(X: List[np.ndarray], scaler: dict) -> List[np.ndarray]:
    """Aplica la normalización (ya ajustada) a una lista de ventanas."""
    return [(x - scaler["mean"]) / scaler["std"] for x in X]
