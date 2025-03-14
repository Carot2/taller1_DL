import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import argparse
import tkinter as tk
from tkinter import filedialog
from data_loader import preprocess_data, extract_floor_info

def select_file(title="Selecciona un archivo", filetypes=(("Archivos CSV", "*.csv"),)):
    """ Abre una ventana de selección de archivos y devuelve la ruta del archivo seleccionado. """
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    return file_path

def load_model_and_preprocessor(model_path='models/best_model_checkpoint.h5', 
                               preprocessor_path='models/preprocessor.joblib'):
    """
    Carga el modelo entrenado y el preprocesador.
    
    Args:
        model_path: Ruta al modelo guardado
        preprocessor_path: Ruta al preprocesador guardado
        
    Returns:
        Tupla (modelo, preprocesador)
    """
    # Verificar que los archivos existen
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encuentra el modelo en {model_path}")
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"No se encuentra el preprocesador en {preprocessor_path}")
    
    # Cargar modelo (sin compilación para evitar problemas)
    model = tf.keras.models.load_model(model_path, compile=False)
    # Recompilar manualmente
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(),
            tf.keras.metrics.MeanSquaredError()
        ]
    )
    
    # Cargar preprocesador
    preprocessor = joblib.load(preprocessor_path)
    
    return model, preprocessor

def predict_single(model, preprocessor, data_dict):
    """
    Realiza una predicción para un único registro.
    
    Args:
        model: Modelo entrenado
        preprocessor: Preprocesador entrenado
        data_dict: Diccionario con características de la propiedad
        
    Returns:
        Precio de alquiler predicho
    """
    # Convertir diccionario a DataFrame
    df = pd.DataFrame([data_dict])
    
    # Preprocesar los datos con la misma función utilizada en entrenamiento
    X, _, _ = preprocess_data(df)
    
    # Aplicar el preprocesador
    X_prep = preprocessor.transform(X)
    
    # Realizar predicción (en escala logarítmica)
    y_pred_log = model.predict(X_prep).flatten()[0]
    
    # Convertir a escala original
    rent_predicted = np.exp(y_pred_log)
    
    return rent_predicted

def predict_batch(model, preprocessor, raw_data):
    """
    Realiza predicciones para un conjunto de datos.
    
    Args:
        model: Modelo entrenado
        preprocessor: Preprocesador entrenado
        raw_data: DataFrame con datos sin procesar
        
    Returns:
        Array con precios de alquiler predichos
    """
    # Preprocesar los datos con la misma función utilizada en entrenamiento
    X, _, _ = preprocess_data(raw_data)
    
    # Aplicar el preprocesador
    X_prep = preprocessor.transform(X)
    
    # Realizar predicciones (en escala logarítmica)
    y_pred_log = model.predict(X_prep).flatten()
    
    # Convertir a escala original
    rent_predicted = np.exp(y_pred_log)
    
    return rent_predicted

def predict_from_file(file_path=None, model_path='models/best_model_checkpoint.h5', 
                     preprocessor_path='models/preprocessor.joblib', 
                     output_path=None):
    """
    Realiza predicciones para datos en un archivo y opcionalmente guarda los resultados.
    
    Args:
        file_path: Ruta al archivo con datos para predicción
        model_path: Ruta al modelo guardado
        preprocessor_path: Ruta al preprocesador guardado
        output_path: Ruta para guardar resultados (opcional)
        
    Returns:
        DataFrame con datos originales y predicciones
    """
    # Si no se proporciona ruta al archivo, pedir selección
    if file_path is None:
        file_path = select_file(title="Selecciona el archivo con datos para predicción",
                               filetypes=[("Archivos CSV", "*.csv")])
        if not file_path:
            print("No se seleccionó ningún archivo. Saliendo...")
            return None
    
    # Verificar que el archivo existe
    if not os.path.exists(file_path):
        print(f"Error: No se encuentra el archivo {file_path}")
        return None
    
    try:
        # Cargar datos
        if file_path.endswith('.csv'):
            raw_data = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            raw_data = pd.read_excel(file_path)
        else:
            print(f"Error: Formato de archivo no soportado. Use .csv o .xlsx")
            return None
        
        print(f"Datos cargados correctamente. Registros: {len(raw_data)}")
        
        # Guardar una copia de los datos originales
        original_data = raw_data.copy()
        
        # Cargar modelo y preprocesador
        model, preprocessor = load_model_and_preprocessor(model_path, preprocessor_path)
        print("Modelo y preprocesador cargados correctamente.")
        
        # Realizar predicciones
        predictions = predict_batch(model, preprocessor, raw_data)
        print(f"Predicciones realizadas correctamente para {len(predictions)} registros.")
        
        # Añadir predicciones al DataFrame original
        original_data['Rent_Predicted'] = predictions
        
        # Calcular métricas si existe la columna 'Rent'
        if 'Rent' in original_data.columns:
            original_data['Error'] = original_data['Rent'] - original_data['Rent_Predicted']
            original_data['Error_Percent'] = (original_data['Error'] / original_data['Rent']) * 100
            
            # Calcular métricas globales
            mae = np.mean(np.abs(original_data['Error']))
            mape = np.mean(np.abs(original_data['Error_Percent']))
            
            print(f"\n=== Métricas de predicción ===")
            print(f"MAE: {mae:.2f}")
            print(f"MAPE: {mape:.2f}%")
        
        # Guardar resultados si se especifica una ruta
        if output_path:
            # Determinar formato de salida
            if output_path.endswith('.csv'):
                original_data.to_csv(output_path, index=False)
            elif output_path.endswith('.xlsx'):
                original_data.to_excel(output_path, index=False)
            else:
                original_data.to_csv(output_path + '.csv', index=False)
                
            print(f"Resultados guardados en {output_path}")
        
        return original_data
    
    except Exception as e:
        print(f"Error durante la predicción: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Realizar predicciones con modelo entrenado')
    parser.add_argument('--file', type=str, default=None,
                        help='Ruta al archivo con datos para predicción')
    parser.add_argument('--model', type=str, default='models/best_model_checkpoint.h5',
                        help='Ruta al modelo guardado')
    parser.add_argument('--preprocessor', type=str, default='models/preprocessor.joblib',
                        help='Ruta al preprocesador guardado')
    parser.add_argument('--output', type=str, default=None,
                        help='Ruta para guardar resultados')
    
    args = parser.parse_args()
    
    # Realizar predicciones
    results = predict_from_file(
        args.file, 
        args.model, 
        args.preprocessor, 
        args.output
    )
    
    # Mostrar primeras filas de resultados si hay resultados
    if results is not None:
        print("\n=== Primeras predicciones ===")
        cols_to_show = ['Rent_Predicted']
        if 'Rent' in results.columns:
            cols_to_show = ['Rent', 'Rent_Predicted', 'Error_Percent']
        print(results[cols_to_show].head())