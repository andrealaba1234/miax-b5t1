"""
Descomposición de retornos en factor de mercado + factor sectorial +
componente idiosincrático.

Esta descomposición es la pieza clave para poder simular después escenarios
sintéticos controlados (p.ej. "shock en el sector bancario", "shock
específico en un activo"): al aislar cada componente, se puede perturbar
uno de ellos sin afectar a los demás y reconstruir series coherentes.

Modelo para cada activo i en el sector s, en el día t:

    r_{i,t} = beta_mkt_i * f_mkt,t + beta_sec_i * f_sec(s)\i,t + eps_{i,t}

donde:
    f_mkt,t       = retorno medio de todos los activos del universo en t (factor mercado)
    f_sec(s)\i,t  = retorno medio de los activos del sector s EXCLUYENDO al
                    propio activo i ("leave-one-out"), en t
    eps_{i,t}     = componente idiosincrático (residuo de la regresión)

Nota sobre leave-one-out: si el factor sectorial de un activo se calculase
incluyéndolo a sí mismo, el activo aparecería en ambos lados de la
regresión (como variable a explicar y como parte de la variable
explicativa). Esto infla artificialmente su R² cuanto menor es el sector
(el caso extremo es un sector de 2 activos, donde cada uno pesa el 50% del
factor). Excluir al propio activo del cálculo de su factor sectorial evita
este problema. Como contrapartida, el factor sectorial pasa a ser
específico de cada activo (ya no hay "un" factor por sector, sino uno por
activo, construido a partir de sus compañeros de sector).
"""

import numpy as np
import pandas as pd


def orthogonalize_series(x: pd.Series, reference: pd.Series) -> pd.Series:
    """
    Elimina de `x` la parte linealmente explicada por `reference`,
    devolviendo el residuo (la parte de `x` no relacionada con `reference`).

    Se usa para "purificar" cada factor sectorial respecto al factor de
    mercado antes de la regresión por activo: el mercado y los sectores
    están correlacionados entre sí (el mercado incluye a los sectores), y
    sin esta corrección la regresión puede repartir el efecto de forma
    poco intuitiva entre beta_market y beta_sector (incluso con signos
    contrarios a lo esperado). Tras ortogonalizar, beta_market representa
    la exposición al mercado en general, y beta_sector la exposición
    *adicional* específica del sector, sin solaparse.

    Regresión sin intercepto: x = beta * reference + residuo.
    """
    aligned = pd.DataFrame({"x": x, "ref": reference}).dropna()
    beta = np.dot(aligned["ref"], aligned["x"]) / np.dot(aligned["ref"], aligned["ref"])
    residual = aligned["x"] - beta * aligned["ref"]
    return residual.reindex(x.index)


def compute_market_factor(returns: pd.DataFrame) -> pd.Series:
    """
    Factor de mercado: retorno medio (equiponderado) de todos los activos
    del universo en cada fecha.

    Parameters
    ----------
    returns : pd.DataFrame
        Retornos diarios, índice=fecha, columnas=símbolos.

    Returns
    -------
    pd.Series
        Factor de mercado por fecha.
    """
    return returns.mean(axis=1).rename("market_factor")


def compute_sector_factors(
    returns: pd.DataFrame,
    sector_map: dict,
) -> pd.DataFrame:
    """
    Factores sectoriales "estándar" (incluyendo a todos los activos del
    sector, sin excluir a nadie): retorno medio de los activos de cada
    sector en cada fecha. Se usa solo para visualización (Notebook 2,
    sección 3) y como referencia; la regresión usa la versión leave-one-out
    (ver compute_sector_factors_loo) para evitar la autocontaminación.

    Parameters
    ----------
    returns : pd.DataFrame
        Retornos diarios, índice=fecha, columnas=símbolos.
    sector_map : dict
        Diccionario {símbolo: sector}, p.ej. {"BAC": "Financials", ...}

    Returns
    -------
    pd.DataFrame
        Índice=fecha, columnas=sectores, valores=factor sectorial.
    """
    sectors = sorted(set(sector_map.values()))
    factors = {}

    for sector in sectors:
        symbols_in_sector = [s for s, sec in sector_map.items() if sec == sector]
        symbols_in_sector = [s for s in symbols_in_sector if s in returns.columns]
        if not symbols_in_sector:
            continue
        factors[sector] = returns[symbols_in_sector].mean(axis=1)

    return pd.DataFrame(factors)


def compute_sector_factors_loo(
    returns: pd.DataFrame,
    sector_map: dict,
) -> pd.DataFrame:
    """
    Factor sectorial "leave-one-out": para cada activo, la media de los
    retornos del RESTO de activos de su mismo sector, excluyéndose a sí
    mismo. Evita que un activo aparezca en ambos lados de su propia
    regresión (ver nota del módulo).

    Con sectores de exactamente 2 activos, el factor de cada uno se reduce
    a la serie del otro en solitario (sin promediar) — sigue siendo válido,
    pero es más sensible al ruido idiosincrático de ese único compañero que
    en sectores con más miembros.

    Parameters
    ----------
    returns : pd.DataFrame
        Retornos diarios, índice=fecha, columnas=símbolos.
    sector_map : dict
        Diccionario {símbolo: sector}.

    Returns
    -------
    pd.DataFrame
        Índice=fecha, columnas=símbolo (no sector), valores=factor
        sectorial leave-one-out específico de ese activo.
    """
    factors = {}

    for symbol in returns.columns:
        sector = sector_map.get(symbol)
        peers = [
            s for s, sec in sector_map.items()
            if sec == sector and s != symbol and s in returns.columns
        ]
        if not peers:
            print(f"Aviso: '{symbol}' no tiene compañeros de sector ('{sector}'); "
                  f"su factor sectorial quedará vacío (NaN).")
            factors[symbol] = pd.Series(np.nan, index=returns.index)
        else:
            factors[symbol] = returns[peers].mean(axis=1)

    return pd.DataFrame(factors)


def compute_sector_factors_loo_weighted(
    returns: pd.DataFrame,
    sector_map: dict,
) -> pd.DataFrame:
    """
    Factor sectorial "leave-one-out ponderado por correlación": para cada
    activo, la media PONDERADA de los retornos del resto de activos de su
    sector (excluyéndose a sí mismo), donde el peso de cada compañero es
    proporcional a su correlación histórica con el activo.

    Motivación: con leave-one-out equiponderado, un activo puede quedar
    "explicado" por compañeros de sector con un comportamiento bursátil
    poco parecido al suyo (p.ej. una empresa ferroviaria en un sector de
    "Industrials" dominado por aerolíneas), generando betas sectoriales
    poco intuitivos aunque el sector ya tenga varios miembros. Ponderar por
    correlación da más peso a los compañeros realmente parecidos y menos
    (o ninguno) a los que no lo son, sin depender de ampliar el universo.

    Las correlaciones negativas se recortan a 0 (un compañero
    anticorrelacionado no resta peso al factor, simplemente no contribuye).
    Si todos los compañeros tienen correlación <= 0, se recurre a peso
    igual entre ellos como salvaguarda, para no dejar el factor vacío.

    Parameters
    ----------
    returns : pd.DataFrame
        Retornos diarios, índice=fecha, columnas=símbolos.
    sector_map : dict
        Diccionario {símbolo: sector}.

    Returns
    -------
    pd.DataFrame
        Índice=fecha, columnas=símbolo, valores=factor sectorial
        leave-one-out ponderado por correlación, específico de ese activo.
    """
    corr_matrix = returns.corr()
    factors = {}

    for symbol in returns.columns:
        sector = sector_map.get(symbol)
        peers = [
            s for s, sec in sector_map.items()
            if sec == sector and s != symbol and s in returns.columns
        ]
        if not peers:
            print(f"Aviso: '{symbol}' no tiene compañeros de sector ('{sector}'); "
                  f"su factor sectorial quedará vacío (NaN).")
            factors[symbol] = pd.Series(np.nan, index=returns.index)
            continue

        weights = corr_matrix.loc[symbol, peers].clip(lower=0)
        if weights.sum() == 0:
            weights = pd.Series(1.0, index=peers)  # salvaguarda: todos anticorrelacionados
        weights = weights / weights.sum()

        factors[symbol] = (returns[peers] * weights).sum(axis=1)

    return pd.DataFrame(factors)


def decompose_asset(
    asset_returns: pd.Series,
    market_factor: pd.Series,
    sector_factor: pd.Series,
) -> dict:
    """
    Descompone los retornos de un activo en su exposición al factor de
    mercado, al factor sectorial (leave-one-out), y su componente
    idiosincrático, mediante una regresión lineal (mínimos cuadrados) de:

        asset_returns ~ beta_mkt * market_factor + beta_sec * sector_factor

    Returns
    -------
    dict con:
        'beta_market': float
        'beta_sector': float
        'idiosyncratic': pd.Series (residuos, misma longitud que asset_returns)
        'r_squared': float (proporción de varianza explicada por los factores)
    """
    df = pd.DataFrame({
        "asset": asset_returns,
        "market": market_factor,
        "sector": sector_factor,
    }).dropna()

    X = np.column_stack([df["market"].values, df["sector"].values])
    y = df["asset"].values

    # Mínimos cuadrados: y = X @ [beta_mkt, beta_sec]
    betas, residuals_sum, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    beta_market, beta_sector = betas

    fitted = X @ betas
    idiosyncratic = y - fitted

    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    idio_series = pd.Series(idiosyncratic, index=df.index, name="idiosyncratic")

    return {
        "beta_market": beta_market,
        "beta_sector": beta_sector,
        "idiosyncratic": idio_series,
        "r_squared": r_squared,
    }


def decompose_universe(
    returns: pd.DataFrame,
    sector_map: dict,
    weighted: bool = True,
) -> dict:
    """
    Aplica decompose_asset a todos los activos del universo, usando el
    factor sectorial leave-one-out y ortogonalizándolo respecto al factor
    de mercado (orthogonalize_series).

    Parameters
    ----------
    weighted : bool, default True
        Si True, usa compute_sector_factors_loo_weighted (ponderado por
        correlación). Si False, usa compute_sector_factors_loo
        (equiponderado). Se deja como parámetro para poder comparar ambas
        versiones fácilmente.

    Returns
    -------
    dict con:
        'market_factor': pd.Series
        'sector_factors_raw': pd.DataFrame (factores por sector, sin LOO ni ortogonalizar; solo para referencia/gráficas)
        'sector_factors_loo_equal': pd.DataFrame (factores LOO equiponderados, sin ortogonalizar; para comparación)
        'sector_factors': pd.DataFrame (factores LOO -ponderados o equiponderados según `weighted`- y ortogonalizados, índice=fecha, columnas=símbolo; los usados en la regresión y en reconstruct_returns)
        'betas': pd.DataFrame (índice=símbolo, columnas=['beta_market', 'beta_sector', 'r_squared', 'sector'])
        'idiosyncratic': pd.DataFrame (índice=fecha, columnas=símbolo)
    """
    market_factor = compute_market_factor(returns)
    sector_factors_raw = compute_sector_factors(returns, sector_map)
    sector_factors_loo_equal = compute_sector_factors_loo(returns, sector_map)

    if weighted:
        sector_factors_loo = compute_sector_factors_loo_weighted(returns, sector_map)
    else:
        sector_factors_loo = sector_factors_loo_equal

    sector_factors_orth = pd.DataFrame({
        symbol: orthogonalize_series(sector_factors_loo[symbol], market_factor)
        for symbol in sector_factors_loo.columns
    })

    betas_rows = []
    idio_cols = {}

    for symbol in returns.columns:
        sector = sector_map.get(symbol)
        if sector is None or symbol not in sector_factors_orth.columns:
            continue

        result = decompose_asset(
            returns[symbol], market_factor, sector_factors_orth[symbol]
        )

        betas_rows.append({
            "symbol": symbol,
            "sector": sector,
            "beta_market": result["beta_market"],
            "beta_sector": result["beta_sector"],
            "r_squared": result["r_squared"],
        })
        idio_cols[symbol] = result["idiosyncratic"]

    betas_df = pd.DataFrame(betas_rows).set_index("symbol")
    idio_df = pd.DataFrame(idio_cols)

    return {
        "market_factor": market_factor,
        "sector_factors_raw": sector_factors_raw,
        "sector_factors_loo_equal": sector_factors_loo_equal,
        "sector_factors": sector_factors_orth,
        "betas": betas_df,
        "idiosyncratic": idio_df,
    }


def reconstruct_returns(
    market_factor: pd.Series,
    sector_factors: pd.DataFrame,
    idiosyncratic: pd.DataFrame,
    betas: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reconstruye los retornos a partir de los tres componentes (operación
    inversa a decompose_universe). Útil como comprobación de consistencia,
    y más adelante como base para reconstruir series sintéticas tras
    perturbar alguno de los factores.

    Parameters
    ----------
    sector_factors : pd.DataFrame
        Factores sectoriales leave-one-out, indexados por SÍMBOLO (no por
        sector) — es decir, `decomposition["sector_factors"]` tal como lo
        devuelve decompose_universe, no `sector_factors_raw`.

    Returns
    -------
    pd.DataFrame
        Retornos reconstruidos, índice=fecha, columnas=símbolo.
    """
    reconstructed = {}

    for symbol in betas.index:
        beta_mkt = betas.loc[symbol, "beta_market"]
        beta_sec = betas.loc[symbol, "beta_sector"]

        reconstructed[symbol] = (
            beta_mkt * market_factor
            + beta_sec * sector_factors[symbol]
            + idiosyncratic[symbol]
        )

    return pd.DataFrame(reconstructed)
