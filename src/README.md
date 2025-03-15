📂 Modelos entrenados
best_model_checkpoint.h5
    Mejor modelo guardado automáticamente por train.py.
preprocessor.joblib
    Escalador y preprocesador de datos usado durante el entrenamiento.
Cómo cargar un modelo guardado:
    import tensorflow as tf
    model = tf.keras.models.load_model('models/best_model_checkpoint.h5')
Cómo cargar el preprocesador:
    import joblib
    preprocessor = joblib.load('models/preprocessor.joblib')
