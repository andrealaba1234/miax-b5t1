"""
Baseline simple (sin red neuronal): "el riesgo futuro se parece al riesgo
reciente". Se calcula la covarianza directamente sobre la ventana de
entrada (los últimos history_days) y se usa como predicción del periodo
siguiente, sin ningún aprendizaje de por medio.

Sirve de referencia obligatoria: cualquier modelo con red neuronal debería
superar este resultado para justificar su complejidad añadida.
"""

from typing import List
import numpy as np

from src.predictor.cholesky import sigma_to_cholesky_vector


def baseline_predict(X: List[np.ndarray]) -> List[np.ndarray]:
    """
    Predicción baseline para una lista de ventanas de entrada: la
    covarianza de cada ventana, tal cual.

    Parameters
    ----------
    X : list de np.ndarray, cada uno shape (history_days, n_assets).

    Returns
    -------
    list de np.ndarray, cada uno shape (n_assets, n_assets).
    """
    return [np.cov(window, rowvar=False) for window in X]


def sigma_mse(sigma_pred: np.ndarray, sigma_real: np.ndarray) -> float:
    """
    Error cuadrático medio entre dos matrices de covarianzas, promediado
    sobre todas las casillas (incluye diagonal y ambos triángulos, ya que
    ambas matrices son simétricas por construcción).
    """
    return float(np.mean((sigma_pred - sigma_real) ** 2))


def baseline_cholesky_vectors(X: List[np.ndarray]) -> List[np.ndarray]:
    """
    Calcula el vector Cholesky del baseline (la covarianza de cada ventana
    de entrada), para usarlo como punto de partida en el modelo con
    conexión residual (ver src/predictor/train.py, train_predictor_residual):
    la red predice solo una corrección sobre este vector, en lugar de la
    Sigma completa desde cero.
    """
    baseline_sigmas = baseline_predict(X)
    return [sigma_to_cholesky_vector(sigma) for sigma in baseline_sigmas]


def evaluate_predictions(sigmas_pred: List[np.ndarray], sigmas_real: List[np.ndarray]) -> dict:
    """
    Evalúa una lista de predicciones frente a sus correspondientes Sigma
    reales, devolviendo el error medio y su desviación estándar entre
    ejemplos.
    """
    errors = [sigma_mse(p, r) for p, r in zip(sigmas_pred, sigmas_real)]
    return {
        "mse_medio": float(np.mean(errors)),
        "mse_std": float(np.std(errors)),
        "n_ejemplos": len(errors),
    }
