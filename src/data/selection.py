"""
Selección de un subconjunto de activos líquidos con histórico completo,
y cálculo de retornos diarios logarítmicos.

Pensado específicamente para el Taller B5-T1 (generación de datos
sintéticos): a diferencia del TFM, aquí no necesitamos modelar un
universo dinámico con entradas/salidas del índice; nos basta con un
puñado fijo de activos con historia suficiente en el periodo de estudio.
"""

import numpy as np
import pandas as pd


def select_liquid_assets(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    n_assets: int = 10,
    symbol_col: str = "symbol",
    date_col: str = "date",
    volume_col: str = "volume",
) -> list:
    """
    Selecciona los `n_assets` activos con mayor volumen medio en el
    periodo, exigiendo que tengan cotización en (casi) todas las fechas
    del rango para evitar huecos.

    Parameters
    ----------
    df : pd.DataFrame
        Histórico ya limpio (símbolos normalizados, sin colisiones).
    start_date, end_date : str
        Rango de fechas 'YYYY-MM-DD' sobre el que se exige histórico completo.
    n_assets : int
        Número de activos a seleccionar.

    Returns
    -------
    list[str]
        Lista de símbolos seleccionados, ordenados por volumen medio descendente.
    """
    mask = (df[date_col] >= pd.Timestamp(start_date)) & (df[date_col] <= pd.Timestamp(end_date))
    window = df.loc[mask]

    n_dates = window[date_col].nunique()

    coverage = window.groupby(symbol_col)[date_col].nunique()
    full_history = coverage[coverage >= 0.98 * n_dates].index

    candidates = window[window[symbol_col].isin(full_history)]
    avg_volume = candidates.groupby(symbol_col)[volume_col].mean().sort_values(ascending=False)

    return avg_volume.head(n_assets).index.tolist()


def select_liquid_assets_by_sector(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    n_per_sector: dict,
    symbol_col: str = "symbol",
    date_col: str = "date",
    volume_col: str = "volume",
    sector_col: str = "sector",
) -> tuple:
    """
    Igual que select_liquid_assets, pero exigiendo un número concreto de
    activos por sector, para asegurar variedad sectorial suficiente y
    poder simular después shocks específicos por sector (p.ej. "más
    volatilidad en bancos").

    Parameters
    ----------
    n_per_sector : dict
        Diccionario {nombre_sector: n_activos}, p.ej.
        {"Financials": 5, "Energy": 4, "Information Technology": 4, ...}
        Los nombres de sector deben coincidir con los valores de la
        columna `sector` del dataset original. Se recomienda pedir al
        menos 2 activos por sector: con un único activo, el "factor
        sectorial" (su media) coincidiría exactamente con ese activo,
        haciendo que su componente idiosincrático salga artificialmente
        nulo en la descomposición de factores posterior.

    Returns
    -------
    tuple[list[str], dict[str, str]]
        (lista de símbolos seleccionados, diccionario {símbolo: sector}).
        El diccionario se devuelve para no tener que reescribir el mapa de
        sector a mano en notebooks posteriores.
    """
    mask = (df[date_col] >= pd.Timestamp(start_date)) & (df[date_col] <= pd.Timestamp(end_date))
    window = df.loc[mask]

    n_dates = window[date_col].nunique()
    coverage = window.groupby(symbol_col)[date_col].nunique()
    full_history = coverage[coverage >= 0.98 * n_dates].index

    candidates = window[window[symbol_col].isin(full_history)]

    selected = []
    sector_map = {}

    for sector, n in n_per_sector.items():
        if n < 2:
            print(f"Aviso: se pidió n={n} para '{sector}'; con menos de 2 activos "
                  f"el factor sectorial coincidirá con ese único activo.")

        sector_candidates = candidates[candidates[sector_col] == sector]
        avg_volume = (
            sector_candidates.groupby(symbol_col)[volume_col]
            .mean()
            .sort_values(ascending=False)
        )
        top_n = avg_volume.head(n).index.tolist()
        if len(top_n) < n:
            print(f"Aviso: solo se encontraron {len(top_n)} activos válidos en '{sector}' (se pedían {n})")

        selected.extend(top_n)
        for symbol in top_n:
            sector_map[symbol] = sector

    return selected, sector_map


def compute_daily_log_returns(
    df: pd.DataFrame,
    symbols: list,
    price_col: str = "close",
    symbol_col: str = "symbol",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Calcula retornos diarios logarítmicos para los activos indicados,
    devolviendo un DataFrame en formato ancho (fechas en filas, activos
    en columnas), que es el formato que necesitaremos para entrenar los
    modelos generativos y estimar la matriz de covarianzas.

    Returns
    -------
    pd.DataFrame
        Índice = fecha, columnas = símbolos, valores = retorno log diario.
    """
    subset = df[df[symbol_col].isin(symbols)].copy()

    wide_prices = subset.pivot(index=date_col, columns=symbol_col, values=price_col)
    wide_prices = wide_prices.sort_index()

    log_returns = np.log(wide_prices / wide_prices.shift(1))
    log_returns = log_returns.dropna(how="all")

    return log_returns
