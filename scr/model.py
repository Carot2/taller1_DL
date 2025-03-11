import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_shape):
    """
    Construye una red neuronal para predecir el precio de alquiler.
    
    Args:
        input_shape (int): Número de características de entrada.
    
    Returns:
        keras.Model: Modelo de Keras compilado.
    """
    model = keras.Sequential([
        layers.Dense(128, activation="relu", input_shape=(input_shape,)),
        layers.Dropout(0.2),  # Regularización para evitar overfitting
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="linear")  # Salida para regresión
    ])

    # Compilar el modelo
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss="mse",
                  metrics=["mae", "mse"])  # Se usan MAE y MSE como métricas

    return model

# Prueba el modelo si se ejecuta directamente el script
if __name__ == "__main__":
    # Simulación de entrada (10 características como ejemplo)
    input_dim = 10  # Reemplazar con el número real de features tras preprocesamiento
    model = build_model(input_dim)

    # Mostrar resumen del modelo
    model.summary()
