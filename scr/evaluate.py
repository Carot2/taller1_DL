import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from data_loader import load_and_preprocess_data

# 1️⃣ Cargar datos preprocesados
df = load_and_preprocess_data()

if df is None:
    print("Error al cargar los datos. Asegúrate de que el archivo existe y está bien formateado.")
    exit()

# Separar características (X) y variable objetivo (y)
X = df.drop(columns=["Rent"])  # Variables de entrada
y = df["Rent"]  # Variable objetivo

# Dividir en conjunto de entrenamiento y prueba (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cargar el modelo entrenado
model_path = "models/model_v1.h5"

if not os.path.exists(model_path):
    print(f"❌ Error: No se encontró el modelo en {model_path}. Ejecuta train.py primero.")
    exit()

model = tf.keras.models.load_model(model_path, custom_objects={
    "MeanSquaredError": tf.keras.losses.MeanSquaredError,
    "MeanAbsoluteError": tf.keras.metrics.MeanAbsoluteError
})
print(f"✅ Modelo cargado desde {model_path}")

# Hacer predicciones
y_pred = model.predict(X_test)

# Calcular métricas de rendimiento
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📌 Resultados del modelo:")
print(f"🔹 Mean Squared Error (MSE): {mse:.2f}")
print(f"🔹 Mean Absolute Error (MAE): {mae:.2f}")
print(f"🔹 R² Score: {r2:.2f}")

# Visualización: Comparación entre valores reales y predichos
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color="blue")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle="dashed", color="red")  # Línea de referencia
plt.xlabel("Precio Real de Alquiler")
plt.ylabel("Precio Predicho")
plt.title("Comparación entre Valores Reales y Predichos")
plt.show()
