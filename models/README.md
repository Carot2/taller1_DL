# Carpeta de Modelos

Esta carpeta contiene los modelos entrenados y los preprocesadores utilizados en el sistema de predicción de precios de alquiler.

## Estructura

La carpeta contiene los siguientes archivos:

- `best_model_checkpoint.h5`: El mejor modelo guardado durante el entrenamiento (según la pérdida de validación).
- `preprocessor.joblib`: Preprocesador utilizado para transformar los datos antes del entrenamiento y predicción.

## Información Técnica

### Modelo de Red Neuronal

El modelo implementado es una red neuronal que utiliza la siguiente arquitectura:

```
- Capa Densa (256 neuronas) + LeakyReLU(alpha=0.1) + Dropout(0.3)
- Capa Densa (128 neuronas) + LeakyReLU(alpha=0.1) + Dropout(0.3)
- Capa Densa (64 neuronas) + LeakyReLU(alpha=0.1)
- Capa de Salida (1 neurona, activación lineal)
```

El modelo está compilado con:
- **Optimizador**: Adam (learning_rate=0.005)
- **Función de pérdida**: MSE (Error Cuadrático Medio)
- **Métricas**: MAE (Error Absoluto Medio) y MSE

### Checkpoints del Entrenamiento

Durante el entrenamiento, el modelo utiliza los siguientes callbacks:

- **Early Stopping**: Detiene el entrenamiento cuando la pérdida de validación deja de mejorar (paciencia de 15 épocas).
- **Model Checkpoint**: Guarda el mejor modelo según la pérdida de validación.
- **ReduceLROnPlateau**: Reduce la tasa de aprendizaje cuando el entrenamiento se estanca (paciencia de 5 épocas).

### Preprocesador

El preprocesador (guardado en formato joblib) contiene todas las transformaciones necesarias para procesar los datos nuevos de la misma manera que los datos de entrenamiento, incluyendo:

- Normalización/Estandarización de características numéricas
- Codificación de variables categóricas
- Selección de características relevantes

## Uso

### Carga del modelo y preprocesador

```python
import tensorflow as tf
import joblib
import os

# Rutas de los archivos
model_path = os.path.normpath('models/best_model_checkpoint.h5')
preprocessor_path = os.path.normpath('models/preprocessor.joblib')

# Cargar el preprocesador
preprocessor = joblib.load(preprocessor_path)

# Cargar el modelo (con manejo de métricas personalizadas)
try:
    model = tf.keras.models.load_model(model_path)
except Exception as error:
    print(f"Error al cargar el modelo principal: {str(error)}")
    # Intentar con el modelo de respaldo
    model = tf.keras.models.load_model(os.path.normpath('models/modelo_final.h5'))
```

### Realizar predicciones

```python
def predict_rental_price(data, model, preprocessor):
    """
    Realiza predicciones de precios de alquiler usando el modelo entrenado.
    
    Args:
        data: DataFrame con los datos de entrada
        model: Modelo cargado
        preprocessor: Preprocesador cargado
        
    Returns:
        Precio de alquiler predicho
    """
    # Preprocesar los datos de entrada
    X_processed = preprocessor.transform(data)
    
    # Realizar la predicción
    prediction = model.predict(X_processed)
    
    # Si el precio estaba en escala logarítmica durante el entrenamiento,
    # convertir de vuelta a la escala original
    # prediction = np.exp(prediction) - 1  # Descomentar si se aplica
    
    return prediction
```

## Notas importantes

- Al cargar el modelo guardado como checkpoint, puede haber problemas con la serialización de métricas como "mse" y "mae". Si esto ocurre, utilice el modelo final en su lugar.
- Si actualiza el modelo, asegúrese de actualizar también el preprocesador para mantener la coherencia.
- Los archivos en formato .h5 pueden ser grandes. Si el espacio es una preocupación, considere guardar solo los pesos del modelo (en lugar del modelo completo).

