"""
Funciones de limpieza y normalización de símbolos.

Basado en la lógica validada en el Notebook 2 original: homogeneización
de mayúsculas/espacios, eliminación de sufijos de versión de ticker
(tipo "-YYYYMM") y resolución de colisiones fecha+símbolo quedándose
con la clase de mayor volumen negociado.
"""

import re
import pandas as pd


_SUFFIX_PATTERN = re.compile(r"-\d{6}$")  # sufijos tipo "-202301"


def normalize_symbols(df: pd.DataFrame, symbol_col: str = "symbol") -> pd.DataFrame:
    """
    Homogeneiza los símbolos: mayúsculas, sin espacios, sin sufijos
    históricos de versión de ticker (p.ej. "ABC-202301" -> "ABC").

    Parameters
    ----------
    df : pd.DataFrame
    symbol_col : str
        Nombre de la columna con el símbolo del activo.

    Returns
    -------
    pd.DataFrame
        Copia del DataFrame con la columna de símbolo normalizada.
    """
    out = df.copy()
    out[symbol_col] = (
        out[symbol_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(_SUFFIX_PATTERN, "", regex=True)
    )
    return out


def resolve_symbol_collisions(
    df: pd.DataFrame,
    symbol_col: str = "symbol",
    date_col: str = "date",
    volume_col: str = "volume",
) -> pd.DataFrame:
    """
    Cuando tras normalizar símbolos aparecen colisiones (varias clases
    del mismo emisor en la misma fecha), conserva únicamente el registro
    con mayor volumen negociado, como criterio de liquidez.

    Returns
    -------
    pd.DataFrame
        DataFrame sin duplicados por (date, symbol).
    """
    out = df.sort_values(volume_col, ascending=False)
    out = out.drop_duplicates(subset=[date_col, symbol_col], keep="first")
    out = out.sort_values([date_col, symbol_col]).reset_index(drop=True)
    return out
