import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_shape):
    """
    Construye una red neuronal mejorada para predecir el precio de alquiler.
    """

    model = keras.Sequential([
        layers.Dense(256, input_shape=(input_shape,)),
        layers.LeakyReLU(alpha=0.1),  # Permite que los gradientes fluyan mejor
        layers.Dropout(0.3),
        layers.Dense(128),
        layers.LeakyReLU(alpha=0.1),
        layers.Dropout(0.3),
        layers.Dense(64),
        layers.LeakyReLU(alpha=0.1),
        layers.Dense(1, activation="linear")  # Regresión, salida sin activación
    ])

    # Compilar el modelo con una tasa de aprendizaje mayor
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005),  
                  loss="mse",
                  metrics=["mae", "mse"])
    
    return model

# Prueba el modelo si se ejecuta directamente el script
if __name__ == "__main__":
    # Simulación de entrada (10 características como ejemplo)
    input_dim = 10  # Reemplazar con el número real de features tras preprocesamiento
    model = build_model(input_dim)

    # Mostrar resumen del modelo
    model.summary()

