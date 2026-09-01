"""
Versión en PyTorch de la transformación Cholesky (ver src/predictor/cholesky.py
para la versión NumPy, usada solo para validación fuera del entrenamiento).

Esta versión debe ser diferenciable (usar únicamente operaciones de
PyTorch, no NumPy) porque se usa DENTRO del cálculo del error durante el
entrenamiento: la red predice el vector de parámetros libres, se
reconstruye Sigma con esta función, y el error se mide sobre esa Sigma
reconstruida — para que el entrenamiento pueda propagar el gradiente hacia
atrás, cada paso de la transformación tiene que ser una operación de
PyTorch.
"""

import torch


def cholesky_vector_to_sigma_torch(vector: torch.Tensor, n: int) -> torch.Tensor:
    """
    Reconstruye Sigma a partir del vector de parámetros libres, igual que
    cholesky_vector_to_sigma pero en PyTorch (soporta lotes de ejemplos).

    Parameters
    ----------
    vector : torch.Tensor
        Shape (batch_size, n*(n+1)/2).
    n : int
        Número de activos.

    Returns
    -------
    torch.Tensor
        Shape (batch_size, n, n) — matrices de covarianzas, garantizadas
        definidas positivas.
    """
    batch_size = vector.shape[0]
    device = vector.device

    diag_log = vector[:, :n]
    off_diag = vector[:, n:]

    L = torch.zeros(batch_size, n, n, device=device, dtype=vector.dtype)

    diag_idx = torch.arange(n, device=device)
    L[:, diag_idx, diag_idx] = torch.exp(diag_log)

    tril_rows, tril_cols = torch.tril_indices(n, n, offset=-1)
    L[:, tril_rows, tril_cols] = off_diag

    sigma = torch.bmm(L, L.transpose(1, 2))
    return sigma
