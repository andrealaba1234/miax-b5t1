"""
Modelos con red neuronal (GRU y LSTM) para predecir la matriz de
covarianzas: reciben la ventana histórica de retornos como una secuencia,
y predicen el vector de parámetros libres de Cholesky (ver
src/predictor/cholesky_torch.py), que se transforma en Sigma de forma
garantizada válida.
"""

import numpy as np
import torch
import torch.nn as nn


class RecurrentCovariancePredictor(nn.Module):
    """
    Encoder recurrente (GRU o LSTM) + capa lineal final, que predice el
    vector de parámetros libres de Cholesky a partir de una ventana de
    retornos.

    Parameters
    ----------
    n_assets : int
        Número de activos (dimensión de entrada en cada paso temporal).
    output_dim : int
        Longitud del vector de salida (n_assets*(n_assets+1)/2).
    cell_type : str
        'gru' o 'lstm'.
    hidden_size : int
        Tamaño del estado oculto de la celda recurrente.
    num_layers : int
        Número de capas recurrentes apiladas.
    dropout : float
        Dropout aplicado sobre el estado oculto final, antes de la capa
        lineal de salida. A diferencia del dropout interno de PyTorch entre
        capas recurrentes (que solo tiene efecto si num_layers > 1), esta
        capa adicional sí funciona con una única capa recurrente.
    """

    def __init__(
        self,
        n_assets: int,
        output_dim: int,
        cell_type: str = "gru",
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        cell_type = cell_type.lower()
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}.get(cell_type)
        if rnn_cls is None:
            raise ValueError(f"cell_type debe ser 'gru' o 'lstm', recibido: {cell_type}")

        self.rnn = rnn_cls(
            input_size=n_assets,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (batch_size, seq_len, n_assets)
        Devuelve: shape (batch_size, output_dim) — vector de parámetros
        libres de Cholesky.
        """
        _, hidden = self.rnn(x)
        # GRU devuelve h_n directamente; LSTM devuelve (h_n, c_n)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        last_layer_hidden = hidden[-1]  # (batch_size, hidden_size), última capa apilada
        last_layer_hidden = self.output_dropout(last_layer_hidden)
        return self.output_layer(last_layer_hidden)


def init_output_layer_to_zero(model: "RecurrentCovariancePredictor") -> None:
    """
    Inicializa la capa de salida a cero (pesos y sesgo). Se usa junto con
    la conexión residual al baseline (ver train_predictor_residual): al
    arrancar en cero, la primera predicción de la red es "corrección nula",
    así que la predicción inicial completa coincide exactamente con el
    baseline — el modelo solo puede mejorar desde ahí, nunca partir peor.

    A diferencia de init_output_bias_from_data (que calibraba la escala
    real porque la red tenía que predecir Sigma completa desde cero), aquí
    ya no hace falta calibrar nada: el baseline aporta la escala correcta
    por sí mismo.
    """
    with torch.no_grad():
        model.output_layer.weight.zero_()
        model.output_layer.bias.zero_()


def init_output_bias_from_data(model: "RecurrentCovariancePredictor", y_train, n_assets: int) -> None:
    """
    Calibra el punto de partida de la capa de salida usando la escala real
    de los datos de entrenamiento, en lugar de dejar que la red arranque
    con pesos completamente aleatorios.

    Motivación: al inicializar una red neuronal, sus pesos son aleatorios,
    así que su primera predicción no tiene ninguna relación con la escala
    real del problema. Para retornos financieros diarios, esa escala es
    muy pequeña (varianzas del orden de 1e-4), y como la diagonal de L se
    recupera con una exponencial (ver cholesky_torch.py), un sesgo inicial
    aleatorio puede traducirse en una Sigma predicha varios órdenes de
    magnitud más grande de lo real — exactamente lo observado en la
    práctica (predicciones ~180 veces mayores que las reales). Con tan
    pocos ejemplos de entrenamiento (76), la red tiene muy poco margen
    para corregir un punto de partida tan alejado por sí sola.

    La calibración funciona así: la diagonal de Sigma es aproximadamente
    L_ii^2 cuando los términos fuera de la diagonal son pequeños (que es
    el caso al inicio, con pesos ~0). Por tanto, para que la Sigma inicial
    ya tenga aproximadamente la varianza media real de cada activo, basta
    con fijar el sesgo de la parte logarítmica de la diagonal a
    0.5*log(varianza_media_real), y poner a cero el resto de sesgos y
    todos los pesos de la capa de salida — así la primera predicción de
    la red, antes de cualquier entrenamiento, ya parte de la escala
    correcta, y el entrenamiento solo tiene que afinar los detalles en
    lugar de tener que descubrir la escala completa desde cero.
    """
    avg_diag = np.mean([np.diag(sigma) for sigma in y_train], axis=0)  # (n_assets,)
    log_L_diag = 0.5 * np.log(avg_diag)

    with torch.no_grad():
        model.output_layer.weight.zero_()
        model.output_layer.bias[:n_assets] = torch.tensor(
            log_L_diag, dtype=model.output_layer.bias.dtype
        )
        model.output_layer.bias[n_assets:] = 0.0
