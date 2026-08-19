"""
Modelo generativo 2 — "no tan tonto": x ~ N(mu, Sigma).

A diferencia del modelo de ruido (Notebook 3), que perturba cada serie de
forma independiente, este modelo ajusta una distribución normal (univariante
para el factor de mercado, multivariante para los factores sectoriales y los
componentes idiosincráticos) a partir del histórico real, y genera datos
completamente nuevos muestreando de esa distribución ajustada.

Al usar una normal MULTIVARIANTE (no una normal por columna, independiente),
la matriz de covarianzas real queda incorporada en la propia generación: si
dos activos se movían juntos en los datos reales, las muestras sintéticas
tenderán a mantener esa relación.
"""

import numpy as np
import pandas as pd


def fit_and_sample_gaussian_univariate(
    data: pd.Series,
    seed: int = None,
) -> pd.Series:
    """
    Ajusta una normal univariante (media y desviación estándar) a `data` y
    genera una muestra sintética de la misma longitud.

    Se usa para el factor de mercado (una única serie).
    """
    rng = np.random.default_rng(seed)
    mu = data.mean()
    sigma = data.std()
    synthetic_values = rng.normal(loc=mu, scale=sigma, size=len(data))
    return pd.Series(synthetic_values, index=data.index, name=data.name)


def fit_and_sample_gaussian_multivariate(
    data: pd.DataFrame,
    seed: int = None,
) -> pd.DataFrame:
    """
    Ajusta una normal MULTIVARIANTE (vector de medias + matriz de
    covarianzas conjunta) a `data` y genera una muestra sintética de la
    misma longitud (mismo número de filas que `data`).

    A diferencia de ajustar una normal por columna de forma independiente,
    esto preserva la estructura de covarianza entre columnas: si dos
    columnas correlacionaban en los datos reales, las muestras sintéticas
    tenderán a mantener esa correlación.

    Se usa para los factores sectoriales y los componentes idiosincráticos
    (30 columnas cada uno, una por activo).

    Parameters
    ----------
    data : pd.DataFrame
        Índice=fecha, columnas=símbolo (o sector).
    seed : int, optional

    Returns
    -------
    pd.DataFrame
        Mismo shape e índice/columnas que `data`, con valores sintéticos.
    """
    rng = np.random.default_rng(seed)

    mean_vector = data.mean().values
    cov_matrix = data.cov().values

    synthetic_values = rng.multivariate_normal(
        mean=mean_vector, cov=cov_matrix, size=len(data)
    )

    return pd.DataFrame(synthetic_values, index=data.index, columns=data.columns)


def fit_and_sample_gaussian_multivariate_shrinkage(
    data: pd.DataFrame,
    seed: int = None,
) -> pd.DataFrame:
    """
    Igual que fit_and_sample_gaussian_multivariate, pero estimando la
    matriz de covarianzas con shrinkage de Ledoit-Wolf en lugar de la
    covarianza muestral simple.

    Motivación: con pocas observaciones relativas al número de columnas
    (aquí, 2785 días para 30 activos), la covarianza muestral puede estar
    mal condicionada y ser ruidosa. El shrinkage la "encoge" hacia una
    matriz más estructurada (un múltiplo de la identidad), lo que suele dar
    una estimación más estable y generalizable — la misma técnica que se
    menciona como línea de trabajo en la estimación de Sigma del TFM.

    Requiere scikit-learn (`sklearn.covariance.LedoitWolf`).
    """
    from sklearn.covariance import LedoitWolf

    rng = np.random.default_rng(seed)

    mean_vector = data.mean().values

    lw = LedoitWolf().fit(data.values)
    cov_matrix = lw.covariance_

    synthetic_values = rng.multivariate_normal(
        mean=mean_vector, cov=cov_matrix, size=len(data)
    )

    return pd.DataFrame(synthetic_values, index=data.index, columns=data.columns)


def generate_synthetic_components_gaussian(
    market_factor: pd.Series,
    sector_factors: pd.DataFrame,
    idiosyncratic: pd.DataFrame,
    seed: int = None,
    shrinkage: bool = False,
) -> dict:
    """
    Genera una versión sintética completa de los tres componentes de la
    descomposición factorial, ajustando y muestreando de una distribución
    normal (univariante para el mercado, multivariante para sector e
    idiosincrático).

    Parameters
    ----------
    shrinkage : bool, default False
        Si True, usa fit_and_sample_gaussian_multivariate_shrinkage
        (covarianza Ledoit-Wolf) para sector_factors e idiosyncratic en
        lugar de la covarianza muestral simple.

    Returns
    -------
    dict con 'market_factor', 'sector_factors', 'idiosyncratic' (versiones
    sintéticas, mismo formato que los inputs).
    """
    seed_market = seed
    seed_sector = seed + 1 if seed is not None else None
    seed_idio = seed + 2 if seed is not None else None

    multivariate_fn = (
        fit_and_sample_gaussian_multivariate_shrinkage if shrinkage
        else fit_and_sample_gaussian_multivariate
    )

    return {
        "market_factor": fit_and_sample_gaussian_univariate(market_factor, seed=seed_market),
        "sector_factors": multivariate_fn(sector_factors, seed=seed_sector),
        "idiosyncratic": multivariate_fn(idiosyncratic, seed=seed_idio),
    }
