import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Cargar el dataset
file_path = "C:/Users/USER/Downloads/House_Rent_Dataset.csv"
df = pd.read_csv(file_path)

# Convertir 'Posted On' a formato datetime y extraer características útiles
df["Posted On"] = pd.to_datetime(df["Posted On"], dayfirst=True, errors="coerce")
df["Year"] = df["Posted On"].dt.year
df["Month"] = df["Posted On"].dt.month
df.drop(columns=["Posted On"], inplace=True)  # Ya no es necesario

# Extraer el número de piso y el total de pisos de la columna 'Floor'
def extract_floor_info(floor_str):
    parts = str(floor_str).split(" out of ")
    floor_num = 0 if parts[0] == "Ground" else int(parts[0]) if parts[0].isdigit() else np.nan
    total_floors = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else np.nan
    return floor_num, total_floors

df[["Floor_Number", "Total_Floors"]] = df["Floor"].apply(lambda x: pd.Series(extract_floor_info(x)))
df.drop(columns=["Floor"], inplace=True)  # Eliminar la columna original

# Aplicar Label Encoding a columnas categóricas
label_encoders = {}
categorical_cols = ["Area Type", "Area Locality", "City", "Furnishing Status", "Tenant Preferred", "Point of Contact"]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Guardar el encoder para revertir si es necesario

# Calcular la correlación solo con variables numéricas
correlation_matrix = df.corr()
rent_correlation = correlation_matrix["Rent"].sort_values(ascending=False)

# Mostrar las variables más relevantes (mayor correlación con Rent)
print("\n📊 Variables con mayor correlación con Rent:")
print(rent_correlation)

# Opcional: Mostrar las variables con correlación superior a 0.1
relevant_features = rent_correlation[abs(rent_correlation) > 0.1]
print("\n✅ Variables relevantes para la predicción:")
print(relevant_features)
