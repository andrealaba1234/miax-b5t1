"""
Representación de la matriz de covarianzas mediante descomposición de
Cholesky, para que cualquier predicción de la red sea automáticamente una
matriz de covarianzas válida (simétrica y semidefinida positiva).

Idea: en vez de predecir Sigma (n x n) directamente, la red predice un
vector de números libres, sin restricciones. Con ese vector se construye
una matriz triangular inferior L, y se calcula:

    Sigma = L @ L.T

Esta multiplicación garantiza que Sigma sea siempre válida, sean cuales
sean los valores de L. Para que la garantía sea estricta (Sigma definida
positiva, no solo semidefinida), los elementos de la diagonal de L se
parametrizan en escala logarítmica: la red predice log(L_ii), y se
recupera L_ii = exp(...), que siempre es positivo, sea cual sea el número
que prediga la red.
"""

import numpy as np


def sigma_to_cholesky_vector(sigma: np.ndarray) -> np.ndarray:
    """
    Descompone una matriz de covarianzas válida en el vector de parámetros
    libres que la representan (la "receta" que debería aprender a predecir
    la red).

    Devuelve un vector de longitud n*(n+1)/2: primero los n elementos de
    la diagonal de L en escala logarítmica (log(L_ii)), después los
    elementos por debajo de la diagonal (L_ij, i > j) sin transformar.

    Parameters
    ----------
    sigma : np.ndarray
        Matriz de covarianzas, shape (n, n), debe ser definida positiva
        (todas las matrices Sigma reales del Paso 1 lo son).

    Returns
    -------
    np.ndarray
        Vector de longitud n*(n+1)/2.
    """
    n = sigma.shape[0]
    L = np.linalg.cholesky(sigma)

    diag_log = np.log(np.diag(L))
    off_diag = L[np.tril_indices(n, k=-1)]

    return np.concatenate([diag_log, off_diag])


def cholesky_vector_to_sigma(vector: np.ndarray, n: int) -> np.ndarray:
    """
    Operación inversa: reconstruye Sigma a partir del vector de parámetros
    libres (lo que predeciría la red). Sea cual sea el contenido de
    `vector`, el resultado es siempre una matriz de covarianzas válida.

    Parameters
    ----------
    vector : np.ndarray
        Vector de longitud n*(n+1)/2, con el mismo formato que produce
        sigma_to_cholesky_vector (diagonal en log, luego resto sin
        transformar).
    n : int
        Número de activos (dimensión de la matriz resultante).

    Returns
    -------
    np.ndarray
        Matriz de covarianzas, shape (n, n), garantizada válida.
    """
    diag_log = vector[:n]
    off_diag = vector[n:]

    L = np.zeros((n, n))
    L[np.diag_indices(n)] = np.exp(diag_log)
    L[np.tril_indices(n, k=-1)] = off_diag

    sigma = L @ L.T
    return sigma


def cholesky_vector_length(n_assets: int) -> int:
    """Longitud del vector de parámetros libres para n_assets activos."""
    return n_assets * (n_assets + 1) // 2
