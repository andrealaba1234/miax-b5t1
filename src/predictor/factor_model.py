"""
Arquitectura para número de activos variable: en lugar de una red que mira
todos los activos a la vez (atada a un tamaño fijo, como
RecurrentCovariancePredictor en models.py), esta mini-red procesa UN
activo cada vez, sin necesitar saber nada de los demás — así se puede
aplicar tantas veces como activos tenga cada ejemplo, sea cual sea ese
número.

Para cada activo, predice un pequeño vector de "carga factorial" (parecido
a las betas de mercado/sector del Notebook 2, pero aprendido en vez de
calculado con regresión) y un valor de componente idiosincrático. Con esos
valores, para TODOS los activos del ejemplo, se reconstruye la matriz de
covarianzas con una fórmula fija:

    Sigma = F @ F.T + diag(idio^2)

donde F es la matriz de cargas factoriales (una fila por activo) e idio es
el vector de desviaciones idiosincráticas. Esta fórmula garantiza que el
resultado sea siempre una matriz de covarianzas válida, para cualquier
número de activos — es la versión "de tamaño variable" de la garantía que
daba Cholesky (que, al depender su longitud de vector de n_assets, solo
funcionaba con un tamaño fijo).
"""

import numpy as np
import torch
import torch.nn as nn


class AssetFactorEncoder(nn.Module):
    """
    Mini-red compartida que procesa la serie temporal de UN activo (su
    historial de retornos) y predice su vector de cargas factoriales y su
    componente idiosincrático.

    Al aplicarse activo por activo con los MISMOS pesos (compartidos), la
    misma red sirve para cualquier número de activos: se ejecuta una vez
    por cada uno, tratando el "número de activos" del ejemplo como
    dimensión de lote (batch), no como parte fija de la arquitectura.

    Parameters
    ----------
    n_factors : int
        Dimensión del vector de carga factorial por activo (k). Un valor
        pequeño (p.ej. 3-5) suele bastar para capturar exposiciones a
        pocos "factores comunes" latentes.
    hidden_size, num_layers, dropout : ver RecurrentCovariancePredictor.
    """

    def __init__(
        self,
        n_factors: int = 4,
        cell_type: str = "gru",
        hidden_size: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_factors = n_factors

        cell_type = cell_type.lower()
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}.get(cell_type)
        if rnn_cls is None:
            raise ValueError(f"cell_type debe ser 'gru' o 'lstm', recibido: {cell_type}")

        self.rnn = rnn_cls(
            input_size=1,  # cada activo se procesa como una serie univariante
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # +1 columna: la última es el componente idiosincrático (en escala log)
        self.output_layer = nn.Linear(hidden_size, n_factors + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (n_assets, seq_len, 1) — un ejemplo, con "n_assets" como
        dimensión de lote (puede ser cualquier número).

        Devuelve: shape (n_assets, n_factors + 1) — cargas factoriales +
        componente idiosincrático (en log) de cada activo.
        """
        _, hidden = self.rnn(x)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        last_layer_hidden = hidden[-1]
        last_layer_hidden = self.dropout(last_layer_hidden)
        return self.output_layer(last_layer_hidden)


def reconstruct_sigma_factor_torch(factor_output: torch.Tensor) -> torch.Tensor:
    """
    Reconstruye Sigma a partir de la salida de AssetFactorEncoder:
    Sigma = F @ F.T + diag(idio^2), donde F son las primeras n_factors
    columnas (cargas factoriales) e idio = exp(última columna) (para
    garantizar valores positivos).

    Parameters
    ----------
    factor_output : torch.Tensor
        Shape (n_assets, n_factors + 1) — salida de AssetFactorEncoder
        para todos los activos de un ejemplo.

    Returns
    -------
    torch.Tensor
        Shape (n_assets, n_assets) — matriz de covarianzas, garantizada
        semidefinida positiva para cualquier n_assets.
    """
    F = factor_output[:, :-1]  # (n_assets, n_factors)
    idio_log = factor_output[:, -1]  # (n_assets,)
    idio_var = torch.exp(2 * idio_log)  # siempre positivo

    sigma = F @ F.T + torch.diag(idio_var)
    return sigma


def reconstruct_sigma_factor_residual_torch(
    factor_output: torch.Tensor,
    baseline_sigma: torch.Tensor,
) -> torch.Tensor:
    """
    Versión con conexión residual al baseline: en lugar de que la fórmula
    de factores REEMPLACE la predicción, se SUMA al baseline:

        Sigma_final = Sigma_baseline + (F @ F.T + diag(idio^2))

    La suma de dos matrices semidefinidas positivas es también
    semidefinida positiva, así que el resultado sigue siendo siempre una
    matriz de covarianzas válida — misma garantía que la versión sin
    residual, pero ahora con el baseline como punto de partida.

    Con la red inicializada cerca de cero (ver init_factor_encoder_near_baseline),
    el término F @ F.T + diag(idio^2) es casi nulo al principio, así que la
    predicción inicial coincide (casi) exactamente con el baseline — igual
    que se hizo con la versión Cholesky (train_predictor_residual en
    train.py), pero adaptado a esta arquitectura de tamaño variable.

    Parameters
    ----------
    baseline_sigma : torch.Tensor
        Shape (n_assets, n_assets) — la covarianza de la ventana de
        entrada de este ejemplo (el baseline), ya como tensor.
    """
    correction = reconstruct_sigma_factor_torch(factor_output)
    return baseline_sigma + correction


def init_factor_encoder_near_baseline(
    model: "AssetFactorEncoder",
    baseline_train=None,
    correction_fraction: float = 0.03,
    idio_init: float = -10.0,
    weight_std: float = 1e-3,
) -> None:
    """
    Inicializa la capa de salida de AssetFactorEncoder para que, al
    arrancar (antes de entrenar), la corrección que aporta sobre el
    baseline sea PEQUEÑA pero no despreciable.

    Con `idio_init=-10` fijo (versión anterior), idio_var = exp(2*(-10))
    ~= 2e-9 — unas 5 órdenes de magnitud más pequeño que la escala típica
    de la diagonal de Sigma (~1e-4). El gradiente ya no es cero (ver nota
    más abajo), pero como se opera en escala logarítmica, cada paso mueve
    idio_var un porcentaje fijo (~2%/época): partiendo de un valor tan
    ínfimo, hacen falta cientos de épocas de crecimiento compuesto para
    que la corrección llegue a pesar algo frente al baseline — por eso el
    entrenamiento se veía "congelado" en la práctica aunque técnicamente
    se moviera.

    Si se pasa `baseline_train` (lista de matrices baseline, una por
    ejemplo), `idio_init` se calibra para que la corrección inicial sea
    una fracción pequeña pero razonable (`correction_fraction`, 3% por
    defecto) de la escala real de la diagonal del baseline — en vez de un
    valor arbitrario. Así arranca lo bastante cerca del baseline para
    mantener la garantía de seguridad, pero con margen para que el
    aprendizaje sea visible en unas pocas decenas de épocas, no cientos.
    Si no se pasa `baseline_train`, se usa el `idio_init` fijo (compatible
    con el comportamiento anterior).

    Los pesos NO se ponen a cero exacto (a diferencia de una primera
    versión de esta función): Sigma depende de F a través de F @ F.T, una
    forma cuadrática cuyo gradiente respecto a F es proporcional a F. Si F
    arranca en cero exacto, ese gradiente es cero exacto también — un punto
    de silla del que ningún learning rate puede escapar (confirmado en la
    práctica: 31 épocas seguidas con el loss bit a bit idéntico). Un ruido
    inicial pequeño (`weight_std`) mantiene la corrección inicial
    despreciable frente a la escala de Sigma, pero rompe la simetría para
    que el gradiente pueda fluir desde la primera época.
    """
    if baseline_train is not None:
        # Cada baseline puede tener un número de activos distinto (modelo de
        # tamaño variable), así que las diagonales no se pueden apilar en un
        # único array rectangular — se concatenan todas antes de promediar.
        avg_diag = np.mean(np.concatenate([np.diag(sigma) for sigma in baseline_train]))
        idio_var_init = correction_fraction * avg_diag
        idio_init = 0.5 * float(np.log(idio_var_init))

    with torch.no_grad():
        model.output_layer.weight.normal_(mean=0.0, std=weight_std)
        model.output_layer.bias.zero_()
        model.output_layer.bias[-1] = idio_init  # última columna = componente idiosincrático (log)
