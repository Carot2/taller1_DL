import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model import build_model
from data_loader import load_and_preprocess_data

#Cargar datos preprocesados
df = load_and_preprocess_data()

if df is None:
    print("Error al cargar los datos. Asegúrate de que el archivo existe y está bien formateado.")
    exit()

#Separar características (X) y variable objetivo (y)
X = df.drop(columns=["Rent"])  # Variables de entrada
y = df["Rent"]  # Variable objetivo

#Dividir en conjunto de entrenamiento y validación (80%-20%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#Construir el modelo
input_shape = X_train.shape[1]
model = build_model(input_shape)

#Definir hiperparámetros
batch_size = 32
epochs = 50

#Entrenar el modelo
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    verbose=1
)

#Crear la carpeta de modelos si no existe
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

#Guardar el modelo entrenado
model.save(os.path.join(model_dir, "model_v1.h5"))
print(f"Modelo guardado en {model_dir}/model_v1.h5")
