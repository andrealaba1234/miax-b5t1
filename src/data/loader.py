"""
Funciones de carga del histórico de precios del S&P 500.

Basado en la lógica validada en el Notebook 1 original (carga de datos
para la práctica de backtesting con momentum).
"""

import pandas as pd


REQUIRED_COLUMNS = [
    "date", "symbol", "in_sp500",
    "open", "high", "low", "close", "volume",
]


def load_sp500_history(path: str) -> pd.DataFrame:
    """
    Carga el parquet histórico completo del S&P 500 y valida
    que contiene las columnas mínimas esperadas.

    Parameters
    ----------
    path : str
        Ruta al archivo parquet con el histórico (p.ej. data/raw/sp500_history.parquet)

    Returns
    -------
    pd.DataFrame
        DataFrame con el histórico completo, con 'date' en formato datetime.
    """
    df = pd.read_parquet(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas esperadas en el dataset: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    return df


def filter_period(
    df: pd.DataFrame,
    start_date: str,
    warmup_months: int = 0,
) -> pd.DataFrame:
    """
    Filtra el histórico a partir de una fecha, reservando un margen de
    'warmup' hacia atrás (útil para poder calcular ventanas móviles,
    retornos, etc. justo al principio del periodo de interés).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columna 'date' en datetime.
    start_date : str
        Fecha de inicio del periodo de interés, formato 'YYYY-MM-DD'.
    warmup_months : int, default 0
        Meses adicionales hacia atrás que se conservan antes de start_date.

    Returns
    -------
    pd.DataFrame
        DataFrame filtrado.
    """
    start = pd.Timestamp(start_date)
    warmup_start = (start - pd.DateOffset(months=warmup_months)).normalize()

    out = df[df["date"] >= warmup_start].copy()
    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)

    return out


def summarize_universe(df: pd.DataFrame) -> dict:
    """
    Genera un pequeño resumen de calidad/consistencia del universo cargado:
    número de activos totales, número que ha pertenecido al S&P 500 en algún
    momento, y estadísticas de composición mensual del índice.

    Útil para reproducir las comprobaciones de sesgo de supervivencia que
    ya hicisteis en el Notebook 1.
    """
    n_total = df["symbol"].nunique()
    n_in_index = df.loc[df["in_sp500"] == 1, "symbol"].nunique()

    by_month = (
        df[df["in_sp500"] == 1]
        .assign(month=lambda x: x["date"].dt.to_period("M"))
        .groupby("month")["symbol"]
        .nunique()
    )

    return {
        "n_activos_totales": n_total,
        "n_activos_en_indice_alguna_vez": n_in_index,
        "composicion_mensual_describe": by_month.describe().to_dict(),
    }
