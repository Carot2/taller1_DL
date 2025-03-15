import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
<<<<<<< HEAD
from data_loader import load_and_preprocess_data, preprocess_data, prepare_train_test_data
=======
from dataloader import load_and_preprocess_data, preprocess_data, prepare_train_test_data
>>>>>>> 88ffc60f25e3b7744ae64e7c7762924b1369655f
from model import build_model

def get_callbacks(model_dir='models'):
    """
    Crea y devuelve los callbacks para el entrenamiento.
    
    Args:
        model_dir: Directorio donde guardar los checkpoints del modelo
    
    Returns:
        Lista de callbacks para usar en el entrenamiento
    """
    # Crear directorio para guardar modelos si no existe
    os.makedirs(model_dir, exist_ok=True)
    
<<<<<<< HEAD
    # Normalizar la ruta para evitar problemas con barras
    checkpoint_path = os.path.join(model_dir, 'best_model_checkpoint.h5')
    checkpoint_path = os.path.normpath(checkpoint_path)
    
=======
>>>>>>> 88ffc60f25e3b7744ae64e7c7762924b1369655f
    # Early Stopping para detener el entrenamiento cuando no hay mejora
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,  # Número de épocas sin mejora antes de detenerse
        restore_best_weights=True,  # Restaurar los mejores pesos encontrados
        verbose=1
    )
    
    # Model Checkpoint para guardar el mejor modelo durante el entrenamiento
    checkpoint = ModelCheckpoint(
<<<<<<< HEAD
        filepath=checkpoint_path,
=======
        filepath=os.path.join(model_dir, 'best_model_checkpoint.h5'),
>>>>>>> 88ffc60f25e3b7744ae64e7c7762924b1369655f
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Reducción de tasa de aprendizaje cuando el entrenamiento se estanca
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Factor por el que se reduce la tasa de aprendizaje
        patience=5,   # Esperar este número de épocas antes de reducir
        min_lr=0.00001,
        verbose=1
    )
    
    return [early_stopping, checkpoint, reduce_lr]

def train_model(X_train, y_train, input_shape, epochs=100, batch_size=32, validation_split=0.2):
    """
    Entrena el modelo con los datos proporcionados.
    
    Args:
        X_train: Datos de entrenamiento
        y_train: Valores objetivo de entrenamiento
        input_shape: Dimensión de entrada para el modelo
        epochs: Número máximo de épocas de entrenamiento
        batch_size: Tamaño del lote para entrenamiento
        validation_split: Proporción de datos para validación
        
    Returns:
        modelo entrenado e historial de entrenamiento
    """
    print("\n=== Entrenando modelo ===")
    
    # Crear el modelo
    modelo = build_model(input_shape)
    
    # Obtener callbacks
    callbacks = get_callbacks()
    
    # Tiempo de inicio
    tiempo_inicio = time.time()
    
    # Entrenar el modelo con callbacks
    history = modelo.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )
    
    # Calcular tiempo de entrenamiento
    tiempo_entrenamiento = time.time() - tiempo_inicio
    print(f"Tiempo de entrenamiento: {tiempo_entrenamiento:.2f} segundos")
    
    return modelo, history

<<<<<<< HEAD
def save_model(modelo, ruta='models/modelo_alquiler.h5', include_optimizer=True):
=======
def save_model(modelo, ruta='models/modelo_alquiler.h5'):
>>>>>>> 88ffc60f25e3b7744ae64e7c7762924b1369655f
    """
    Guarda el modelo entrenado en disco.
    
    Args:
        modelo: Modelo a guardar
        ruta: Ruta donde guardar el modelo
<<<<<<< HEAD
        include_optimizer: Si se debe incluir el optimizador
=======
>>>>>>> 88ffc60f25e3b7744ae64e7c7762924b1369655f
    """
    # Crear carpeta si no existe
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    
<<<<<<< HEAD
    # Normalizar la ruta (usar forward slashes en lugar de backslashes)
    ruta = ruta.replace('\\', '/')
    ruta = os.path.normpath(ruta)
    
    # Guardar modelo
    try:
        modelo.save(ruta, include_optimizer=include_optimizer)
        print(f"Modelo guardado en: {ruta}")
    except Exception as error:
        print(f"Error al guardar el modelo: {str(error)}")
=======
    # Guardar modelo
    modelo.save(ruta)
    print(f"Modelo guardado en: {ruta}")
>>>>>>> 88ffc60f25e3b7744ae64e7c7762924b1369655f

def save_preprocessor(preprocessor, ruta='models/preprocessor.joblib'):
    """
    Guarda el preprocesador en disco para uso futuro.
    
    Args:
        preprocessor: Preprocesador a guardar
        ruta: Ruta donde guardar el preprocesador
    """
    import joblib
    
    # Crear carpeta si no existe
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    
<<<<<<< HEAD
    # Normalizar la ruta (usar forward slashes en lugar de backslashes)
    ruta = ruta.replace('\\', '/')
    ruta = os.path.normpath(ruta)
    
=======
>>>>>>> 88ffc60f25e3b7744ae64e7c7762924b1369655f
    # Guardar preprocesador
    joblib.dump(preprocessor, ruta)
    print(f"Preprocesador guardado en: {ruta}")

def main(epochs=100, batch_size=32):
    """
    Función principal que ejecuta el proceso de entrenamiento.
    
    Args:
        epochs: Número máximo de épocas de entrenamiento
        batch_size: Tamaño del lote para entrenamiento
        
    Returns:
        modelo, preprocesador e historial de entrenamiento
    """
    # Configurar memoria GPU de forma dinámica (si está disponible)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Usando {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"Error al configurar GPU: {e}")
    
    # Cargar datos
    df = load_and_preprocess_data()
    if df is None:
        print("Error al cargar los datos. Finalizando el programa.")
        return None, None, None
    
    # Preprocesar datos
    X, y, y_original = preprocess_data(df)
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test, y_train_original, y_test_original, preprocessor = \
        prepare_train_test_data(X, y, y_original)
    
    # Entrenar modelo
    modelo, history = train_model(
        X_train, y_train, 
        input_shape=X_train.shape[1],
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Guardar preprocesador
    save_preprocessor(preprocessor)
    
<<<<<<< HEAD
    # Normalizar la ruta del checkpoint
    checkpoint_path = os.path.join('models', 'best_model_checkpoint.h5')
    checkpoint_path = os.path.normpath(checkpoint_path)
    
    # Verificar que el checkpoint existe
=======
    # No es necesario guardar el modelo final, ya que usaremos el mejor checkpoint
    # Pero puedes verificar que el mejor checkpoint existe
    checkpoint_path = os.path.join('models', 'best_model_checkpoint.h5')
>>>>>>> 88ffc60f25e3b7744ae64e7c7762924b1369655f
    if os.path.exists(checkpoint_path):
        print(f"\n✓ Mejor modelo guardado en: {checkpoint_path}")
    else:
        print(f"\n⚠ No se encontró el archivo del mejor modelo en: {checkpoint_path}")
        print("Guardando el modelo final como respaldo...")
<<<<<<< HEAD
        save_model(modelo, 'models/modelo_final.h5')
=======
        save_model(modelo, 'models/modelo_final.h5', include_optimizer=False)
>>>>>>> 88ffc60f25e3b7744ae64e7c7762924b1369655f
    
    print("\n=== Entrenamiento completado con éxito ===")
    
    # Devolver elementos necesarios para evaluación posterior
<<<<<<< HEAD
    # La solución más simple: usar el modelo final y evitar cargar el checkpoint
    print("Usando el modelo final directamente (evitando problemas de carga).")
    return modelo, preprocessor, history, (X_test, y_test, y_test_original)
=======
    # Intentar cargar el mejor modelo desde el checkpoint
    try:
        mejor_modelo = tf.keras.models.load_model(checkpoint_path)
        print("Mejor modelo cargado correctamente para devolución.")
        return mejor_modelo, preprocessor, history, (X_test, y_test, y_test_original)
    except:
        print("No se pudo cargar el mejor modelo. Devolviendo el modelo final.")
        return modelo, preprocessor, history, (X_test, y_test, y_test_original)
>>>>>>> 88ffc60f25e3b7744ae64e7c7762924b1369655f

if __name__ == "__main__":
    modelo, preprocessor, history, test_data = main(epochs=100, batch_size=32)