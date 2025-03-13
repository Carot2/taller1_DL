import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
from sklearn.preprocessing import StandardScaler, LabelEncoder

def select_file():
    """ Abre una ventana de selección de archivos y devuelve la ruta del archivo seleccionado. """
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal de Tkinter
    file_path = filedialog.askopenfilename(title="Selecciona el archivo de datos",
                                           filetypes=[("Archivos Excel", "*.xlsx"), ("Archivos CSV", "*.csv")])
    return file_path

def load_and_preprocess_data():
    """
    Carga y preprocesa el dataset de precios de alquiler de viviendas, eliminando variables irrelevantes y NaN.
    Returns:
        pd.DataFrame: DataFrame con los datos preprocesados.
    """

    # SELECCIONAR EL ARCHIVO
    file_path = select_file()
    #file_path = "C:/Users/USER/Downloads/House_Rent_Dataset.csv"

    if not file_path:
        print("❌ No se seleccionó ningún archivo. Saliendo...")
        return None

    print(f"📂 Archivo seleccionado: {file_path}")

    # Cargar el archivo
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("❌ Formato de archivo no soportado. Usa .csv o .xlsx")

    # 🔹 Verificar valores NaN antes del preprocesamiento
    print(f"⚠️ Valores NaN antes del preprocesamiento: {df.isna().sum().sum()}")

    # Convertir la columna 'Posted On' a tipo datetime y extraer el mes (el año se elimina)
    df["Posted On"] = pd.to_datetime(df["Posted On"], dayfirst=True, errors="coerce")
    df["Month"] = df["Posted On"].dt.month
    df.drop(columns=["Posted On"], inplace=True)  # Eliminar columna original

    # Extraer el número de piso y el total de pisos de la columna 'Floor'
    def extract_floor_info(floor_str):
        parts = str(floor_str).split(" out of ")
        floor_num = 0 if parts[0] == "Ground" else int(parts[0]) if parts[0].isdigit() else np.nan
        total_floors = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else np.nan
        return floor_num, total_floors

    df[["Floor_Number", "Total_Floors"]] = df["Floor"].apply(lambda x: pd.Series(extract_floor_info(x)))
    df.drop(columns=["Floor"], inplace=True)  # Eliminar columna original

    # Convertir 'Furnishing Status' en variables dummies
    df = pd.get_dummies(df, columns=["Furnishing Status"], prefix="Furnish")

    # Convertir 'Area Type' en variables dummies
    df = pd.get_dummies(df, columns=["Area Type"], prefix="Area")

    # **Manejo de 'City'**
    if df["City"].nunique() > 1:  # Solo si hay más de una ciudad
        df = pd.get_dummies(df, columns=["City"], prefix="City")

    # 🔹 ELIMINAR COLUMNAS IRRELEVANTES
    df.drop(columns=["Tenant Preferred", "Area Locality", "Point of Contact"], inplace=True)

    # 🔹 Manejo de valores NaN: Eliminación o Imputación
    df.dropna(inplace=True)  # 🔹 Opción 1: Eliminar filas con NaN
    # Alternativa: df.fillna(df.mean(), inplace=True)  # 🔹 Opción 2: Reemplazar con la media

    # Verificar que no haya valores NaN después del preprocesamiento
    print(f"✅ Filas con NaN eliminadas. Total de filas ahora: {df.shape[0]}")
    
    # Normalizar variables numéricas
    num_cols = ["BHK", "Rent", "Size", "Bathroom", "Floor_Number", "Total_Floors", "Month"]
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    print("✅ Preprocesamiento completado con éxito.")
    return df

# 🔹 Muestra de dataset:
if __name__ == "__main__":
    processed_df = load_and_preprocess_data()
    if processed_df is not None:
        print(processed_df.head())
        print(processed_df.columns)
