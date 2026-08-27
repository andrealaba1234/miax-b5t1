# Modelos VAE definitivos

Esta carpeta conserva una copia autocontenida de los cuatro generadores VAE seleccionados en el Notebook 5 por el trade-off entre fidelidad y diversidad temporal.

Cada subcarpeta contiene:

- `model.pt`: pesos entrenados;
- `config.json`: arquitectura, ventana, dimensión latente, variables y metadatos;
- `scaler.joblib`: normalización necesaria para entrada y salida;
- `training_history.csv`: historial completo del reentrenamiento final.

Las curvas correspondientes están en `reports/figures_final_models/vae`. Las rutas de todos los artefactos se recogen en `model_registry.csv`.

Los archivos `.pt` están ignorados globalmente por Git. Permanecen disponibles localmente, pero para publicarlos será necesario usar Git LFS o añadir explícitamente únicamente estos cuatro modelos.
