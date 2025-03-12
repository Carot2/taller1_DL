import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model import build_model
from data_loader import load_and_preprocess_data
from sklearn.preprocessing import StandardScaler
import joblib  # Para guardar y cargar el scaler

# 1️⃣ Cargar datos preprocesados
df = load_and_preprocess_data()

if df is None:
    print("❌ Error al cargar los datos. Asegúrate de que el archivo existe y está bien formateado.")
    exit()


# 2️⃣ Separar características (X) y variable objetivo (y)
X = df.drop(columns=["Rent"])  # Variables de entrada
y = df["Rent"].values.reshape(-1, 1)  # Convertir a array 2D para escalar

# 3️⃣ Escalar X e Y
scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)  # Ahora `y_train` estará correctamente escalado

# 4️⃣ Dividir en conjunto de entrenamiento y validación (80%-20%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Revisar si hay NaN o Inf en los datos de entrenamiento
print("Valores NaN en X_train:", np.isnan(X_train).sum())
print("Valores Inf en X_train:", np.isinf(X_train).sum())

print("Valores NaN en y_train:", np.isnan(y_train).sum())
print("Valores Inf en y_train:", np.isinf(y_train).sum())

# 5️⃣ Construir el modelo
input_shape = X_train.shape[1]
model = build_model(input_shape)

# 6️⃣ Definir hiperparámetros
batch_size = 32
epochs = 50

# 7️⃣ Entrenar el modelo
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    verbose=1
)

# 8️⃣ Crear la carpeta de modelos si no existe
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# 9️⃣ Guardar el modelo entrenado
model.save(os.path.join(model_dir, "model_v1.h5"))
print(f"✅ Modelo guardado en {model_dir}/model_v1.h5")

# 🔹 Guardar el scaler de y después del ajuste
scaler_path = os.path.join(model_dir, "scaler_y.pkl")
joblib.dump(scaler_y, scaler_path)
print(f"✅ Escalador de y_train guardado en {scaler_path}")
