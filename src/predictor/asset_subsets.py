"""
Expande los ejemplos de tamaño fijo (build_covariance_windows, siempre con
los n_assets completos) en muchos más ejemplos con subconjuntos de tamaño
y composición variable — la propuesta de Miriam para dar al modelo mucha
más variedad genuina de "situaciones distintas" de las que aprender, no
solo más ventanas temporales del mismo grupo fijo de activos.

Idea clave: la covarianza de un subconjunto de activos es exactamente el
"recorte" (submatriz) correspondiente de la matriz de covarianzas completa
ya calculada — no hace falta recalcular nada con np.cov. Y una submatriz de
una matriz de covarianzas válida es, por una propiedad del álgebra lineal,
también una matriz de covarianzas válida — así que los objetivos (y) siguen
siendo correctos automáticamente, sin ningún trabajo extra.
"""

import numpy as np


def build_variable_asset_windows(
    windows: dict,
    n_assets: int,
    subset_sizes: list = None,
    n_subsets_per_example: int = 5,
    seed: int = None,
) -> dict:
    """
    Expande cada ejemplo de `windows` (construido con build_covariance_windows,
    siempre con los n_assets completos) en varios ejemplos con subconjuntos
    de activos de tamaño y composición aleatoria.

    Parameters
    ----------
    windows : dict
        Salida de build_covariance_windows (con los n_assets completos).
    n_assets : int
        Número total de activos disponibles (columnas de windows["X"]).
    subset_sizes : list[int], optional
        Tamaños de subconjunto a muestrear (p.ej. [2, 5, 10, 20, 30]). Si
        no se especifica, se permite cualquier tamaño entre 2 y n_assets.
    n_subsets_per_example : int
        Cuántos subconjuntos distintos generar por cada fecha de corte
        original.
    seed : int, optional

    Returns
    -------
    dict con:
        'X': lista de np.ndarray, cada uno (history_days, tamaño_variable)
        'y': lista de np.ndarray, cada uno (tamaño_variable, tamaño_variable)
        'cutoff_dates': lista de fechas (repetidas, una por subconjunto)
        'asset_indices': lista de arrays con los índices de activos usados
                         en cada ejemplo (útil para trazabilidad)
    """
    rng = np.random.default_rng(seed)

    if subset_sizes is None:
        subset_sizes = list(range(2, n_assets + 1))

    X_list, y_list, dates_list, assets_list = [], [], [], []

    for X_full, y_full, date in zip(windows["X"], windows["y"], windows["cutoff_dates"]):
        for _ in range(n_subsets_per_example):
            size = rng.choice(subset_sizes)
            idx = np.sort(rng.choice(n_assets, size=size, replace=False))

            X_list.append(X_full[:, idx])
            y_list.append(y_full[np.ix_(idx, idx)])
            dates_list.append(date)
            assets_list.append(idx)

    return {
        "X": X_list,
        "y": y_list,
        "cutoff_dates": dates_list,
        "asset_indices": assets_list,
    }
