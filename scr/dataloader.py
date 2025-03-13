
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

def select_file():
    """ Abre una ventana de selección de archivos y devuelve la ruta del archivo seleccionado. """
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal de Tkinter
    file_path = filedialog.askopenfilename(title="Selecciona el archivo de datos",
                                           filetypes=[("Archivos CSV", "*.csv")])
    return file_path

def extract_floor_info(floor_str):
    """
    Extrae información del piso y total de pisos.
    Retorna: (número_de_piso, total_de_pisos)
    """
    try:
        floor_str = str(floor_str).strip()

        # Procesar valores especiales
        if "Upper Basement" in floor_str:
            floor_num = -1
        elif "Lower Basement" in floor_str:
            floor_num = -2
        elif "Ground" in floor_str:
            floor_num = 0
        else:
            # Extraer número de piso
            parts = floor_str.split("out of")
            if len(parts) > 0 and parts[0].strip().isdigit():
                floor_num = int(parts[0].strip())
            else:
                floor_num = None

        # Extraer total de pisos
        if "out of" in floor_str:
            parts = floor_str.split("out of")
            if len(parts) > 1 and parts[1].strip().isdigit():
                total_floors = int(parts[1].strip())
            else:
                total_floors = None
        else:
            total_floors = None

        return floor_num, total_floors
    except:
        return None, None

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
    return df    
def preprocess_data(df):
    """
    Realiza el preprocesamiento completo del dataset.
    
    Args:
        df: DataFrame con los datos crudos
        
    Returns:
        X, y, y_original: Features y target en forma logarítmica y original
        preprocessor: Objeto ColumnTransformer entrenado
    """
    if df is None:
        return None, None, None, None
    
    print("=== Preprocesando datos ===")
    
    # Filtrar registros con categorías poco frecuentes
    df_proc = df[(df['Area Type'] != 'Built Area') &
                 (df['Point of Contact'] != 'Contact Builder')].copy()
    
    print(f"Dimensiones después de filtrado: {df_proc.shape}")
    
    # Procesar fechas
    df_proc["Posted On"] = pd.to_datetime(df_proc["Posted On"], dayfirst=True, errors="coerce")
    df_proc["Month"] = df_proc["Posted On"].dt.month
    
    # Procesar pisos
    df_proc[['Floor_Number', 'Total_Floors']] = df_proc['Floor'].apply(
        lambda x: pd.Series(extract_floor_info(x))
    )
    
    # Calcular ratio de pisos
    df_proc['Floor_Ratio'] = None
    mask = (~df_proc['Floor_Number'].isna()) & (~df_proc['Total_Floors'].isna()) & (df_proc['Total_Floors'] > 0)
    df_proc.loc[mask, 'Floor_Ratio'] = df_proc.loc[mask, 'Floor_Number'] / df_proc.loc[mask, 'Total_Floors']
    
    # Rellenar valores nulos en columnas de pisos
    df_proc['Floor_Number'] = df_proc['Floor_Number'].fillna(df_proc['Floor_Number'].median())
    df_proc['Total_Floors'] = df_proc['Total_Floors'].fillna(df_proc['Total_Floors'].median())
    df_proc['Floor_Ratio'] = df_proc['Floor_Ratio'].fillna(df_proc['Floor_Ratio'].median())
    
    # Transformaciones logarítmicas
    df_proc['Rent_log'] = np.log(df_proc['Rent'])
    df_proc['Size_log'] = np.log(df_proc['Size'])
    
    # Codificar variables categóricas
    df_proc = pd.get_dummies(df_proc, columns=["Furnishing Status"], prefix="Furnish")
    df_proc = pd.get_dummies(df_proc, columns=["Area Type"], prefix="Area")
    
    # Si hay múltiples ciudades, convertir a dummies
    if df_proc["City"].nunique() > 1:
        df_proc = pd.get_dummies(df_proc, columns=["City"], prefix="City")
    
    # Eliminar columnas innecesarias
    columnas_a_eliminar = ['Posted On', 'Area Locality', 'Floor', 'Tenant Preferred', 'Point of Contact']
    X = df_proc.drop(columns=columnas_a_eliminar + ['Rent', 'Rent_log']).copy()
    y = df_proc['Rent_log'].copy()
    y_original = df_proc['Rent'].copy()
    
    # Manejar valores nulos restantes
    for col in X.columns:
        if X[col].isna().any():
            if X[col].dtype.kind in 'ifc':  # Numérico
                X[col] = X[col].fillna(X[col].median())
            else:  # Categórico
                X[col] = X[col].fillna(X[col].mode()[0])
    
    print(f"✅ Preprocesamiento completado. Features: {X.shape[1]}")
    return X, y, y_original

def prepare_train_test_data(X, y, y_original, test_size=0.2, random_state=42):
    """
    Divide los datos en conjuntos de entrenamiento y prueba, y aplica escalado.
    
    Returns:
        Conjuntos de datos preparados y el preprocesador
    """
    # División en train/test
    X_train, X_test, y_train, y_test, y_train_original, y_test_original = train_test_split(
        X, y, y_original, test_size=test_size, random_state=random_state
    )
    
    # Identificar tipos de columnas
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Preprocesador para escalado y codificación
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    
    # Aplicar preprocesamiento
    preprocessor.fit(X_train)
    X_train_prep = preprocessor.transform(X_train)
    X_test_prep = preprocessor.transform(X_test)
    
    # Verificar y manejar NaN
    if np.isnan(X_train_prep).any() or np.isnan(X_test_prep).any():
        X_train_prep = np.nan_to_num(X_train_prep)
        X_test_prep = np.nan_to_num(X_test_prep)
    
    return X_train_prep, X_test_prep, y_train, y_test, y_train_original, y_test_original, preprocessor

if __name__ == "__main__":
    # Ejemplo de uso
    df = load_and_preprocess_data()
    if df is not None:
        X, y, y_original = preprocess_data(df)
        X_train_prep, X_test_prep, y_train, y_test, y_train_original, y_test_original, preprocessor = prepare_train_test_data(X, y, y_original)
        print(f"Datos de entrenamiento: {X_train_prep.shape}")
        print(f"Datos de prueba: {X_test_prep.shape}")