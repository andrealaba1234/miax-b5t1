# Modelo autoregresivo final

El generador final es un `ProbabilisticMultiHorizonGRU` entrenado para predecir conjuntamente cinco días y utilizado mediante rollout rolling.

- Arquitectura: `[48, 24, 12]`
- Dropout: `0.35`
- Contexto: `20` días
- Mejor época: `100`
- Validation selection score: `0.4370733754975455`

Los artefactos reproducibles están en `multihorizon_5d/`: checkpoint, scaler, configuración e historial completo de entrenamiento.
