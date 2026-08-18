"""
Modelo generativo 1 — "muy tonto": x_sintetico = x_real + ruido.

Es el método de referencia obligatorio del taller (Taller B5-T1, punto 4:
"un cuarto modelo simple, por ejemplo que coja datos originales y les añada
ruido"). Sirve como base de comparación: cualquier modelo generativo más
sofisticado (gaussiana, VAE, AR/GRU) debería superarlo en la tarea
downstream (predicción de la matriz de covarianzas) para justificar su uso.

Actúa sobre los COMPONENTES de la descomposición factorial (Notebook 2:
factor de mercado, factores sectoriales, componentes idiosincráticos), no
sobre los retornos en bruto. Esto es deliberado: aunque el modelo de ruido
en sí es "tonto", operar a nivel de componente mantiene la misma interfaz
que usarán los modelos más sofisticados (Notebook 4-6), que sí necesitan
generar cada pieza por separado para poder simular escenarios condicionados
(p.ej. "más ruido específicamente en el factor sectorial de banca").
"""

import numpy as np
import pandas as pd


def add_gaussian_noise(
    data,
    noise_std_ratio: float = 1.0,
    seed: int = None,
):
    """
    Añade ruido gaussiano a una serie o DataFrame de retornos: cada columna
    recibe ruido ~ N(0, noise_std_ratio * std_historico_de_esa_columna).

    El ruido se escala con la volatilidad propia de cada serie (no un
    mismo ruido absoluto para todas) para que el resultado sea comparable
    en magnitud a los datos reales, sea cual sea la escala de cada
    componente (el factor de mercado y el componente idiosincrático de un
    activo muy volátil no tienen la misma escala).

    Parameters
    ----------
    data : pd.Series o pd.DataFrame
        Componente(s) real(es) sobre los que generar la versión sintética.
    noise_std_ratio : float, default 1.0
        Ratio entre la desviación estándar del ruido añadido y la
        desviación estándar histórica de cada columna. 1.0 = el ruido
        tiene, en promedio, la misma magnitud que la propia serie.
    seed : int, optional
        Semilla para reproducibilidad.

    Returns
    -------
    Mismo tipo que `data` (Series o DataFrame), con el ruido añadido.
    """
    rng = np.random.default_rng(seed)

    if isinstance(data, pd.Series):
        noise_std = data.std() * noise_std_ratio
        noise = rng.normal(loc=0.0, scale=noise_std, size=len(data))
        return data + pd.Series(noise, index=data.index, name=data.name)

    elif isinstance(data, pd.DataFrame):
        synthetic = data.copy()
        for col in data.columns:
            noise_std = data[col].std() * noise_std_ratio
            noise = rng.normal(loc=0.0, scale=noise_std, size=len(data))
            synthetic[col] = data[col].values + noise
        return synthetic

    else:
        raise TypeError(f"add_gaussian_noise espera pd.Series o pd.DataFrame, recibió {type(data)}")


def generate_synthetic_components_noise(
    market_factor: pd.Series,
    sector_factors: pd.DataFrame,
    idiosyncratic: pd.DataFrame,
    noise_std_ratio: float = 1.0,
    seed: int = None,
) -> dict:
    """
    Genera una versión sintética completa de los tres componentes de la
    descomposición factorial (mercado, sector, idiosincrático), aplicando
    ruido gaussiano de forma independiente a cada uno.

    Se usan semillas derivadas de `seed` (si se proporciona) para que cada
    componente reciba ruido distinto pero reproducible.

    Returns
    -------
    dict con 'market_factor', 'sector_factors', 'idiosyncratic' (versiones
    sintéticas, mismo formato que los inputs).
    """
    seed_market = seed
    seed_sector = seed + 1 if seed is not None else None
    seed_idio = seed + 2 if seed is not None else None

    return {
        "market_factor": add_gaussian_noise(market_factor, noise_std_ratio, seed=seed_market),
        "sector_factors": add_gaussian_noise(sector_factors, noise_std_ratio, seed=seed_sector),
        "idiosyncratic": add_gaussian_noise(idiosyncratic, noise_std_ratio, seed=seed_idio),
    }
