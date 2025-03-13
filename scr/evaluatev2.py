import numpy as np
import tkinter as tk
from tkinter import filedialog
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

def select_file(title="Selecciona un archivo", filetypes=(("Archivos CSV", "*.csv"),)):
    """ Abre una ventana de selección de archivos y devuelve la ruta del archivo seleccionado. """
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal de Tkinter
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    return file_path

def select_model_file():
    """ Abre una ventana de selección para un archivo de modelo .h5 """
    return select_file(
        title="Selecciona el archivo del modelo (.h5)",
        filetypes=[("Archivos de modelo", "*.h5")]
    )

def select_data_file():
    """ Abre una ventana de selección para un archivo de datos CSV """
    return select_file(
        title="Selecciona el archivo de datos (.csv)",
        filetypes=[("Archivos CSV", "*.csv")]
    )

def select_preprocessor_file():
    """ Abre una ventana de selección para un archivo de preprocesador .joblib """
    return select_file(
        title="Selecciona el archivo del preprocesador (.joblib)",
        filetypes=[("Archivos joblib", "*.joblib")]
    )

def evaluate_model(model, X_test, y_test, y_test_original=None):
    """
    Evalúa el rendimiento del modelo utilizando múltiples métricas.
    
    Args:
        model: Modelo entrenado para evaluar
        X_test: Features de prueba procesadas
        y_test: Target de prueba (en escala logarítmica)
        y_test_original: Target de prueba en escala original (opcional)
        
    Returns:
        Dict con métricas de evaluación
    """
    print("\nNOTA: Este módulo está diseñado para evaluar modelos de regresión lineal.")
    
    # Realizar predicciones (en escala logarítmica)
    y_pred_log = model.predict(X_test).flatten()
    
    # Métricas en escala logarítmica
    metrics = {
        'mse_log': mean_squared_error(y_test, y_pred_log),
        'mae_log': mean_absolute_error(y_test, y_pred_log),
        'rmse_log': np.sqrt(mean_squared_error(y_test, y_pred_log)),
        'r2_log': r2_score(y_test, y_pred_log)
    }
    
    # Si se proporcionan los valores originales, calcular métricas en escala original
    if y_test_original is not None:
        # Transformar predicciones a escala original
        y_pred_original = np.exp(y_pred_log)
        
        # Métricas en escala original
        metrics.update({
            'mse': mean_squared_error(y_test_original, y_pred_original),
            'mae': mean_absolute_error(y_test_original, y_pred_original),
            'rmse': np.sqrt(mean_squared_error(y_test_original, y_pred_original)),
            'r2': r2_score(y_test_original, y_pred_original),
            'mape': np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100,
            'predictions': y_pred_original
        })
    
    return metrics

def print_metrics(metrics, original_scale=True):
    """
    Imprime las métricas de evaluación en un formato legible.
    
    Args:
        metrics: Diccionario con métricas de evaluación
        original_scale: Si se deben mostrar métricas en escala original
    """
    print("\n=== Métricas de Evaluación ===")
    
    # Métricas en escala logarítmica
    print("\nEscala Logarítmica:")
    print(f"MSE (log): {metrics['mse_log']:.4f}")
    print(f"MAE (log): {metrics['mae_log']:.4f}")
    print(f"RMSE (log): {metrics['rmse_log']:.4f}")
    print(f"R² (log): {metrics['r2_log']:.4f}")
    
    # Métricas en escala original (si están disponibles)
    if original_scale and 'mse' in metrics:
        print("\nEscala Original:")
        print(f"MSE: {metrics['mse']:.2f}")
        print(f"MAE: {metrics['mae']:.2f}")
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"R²: {metrics['r2']:.4f}")
        print(f"MAPE: {metrics['mape']:.2f}%")

def analyze_errors(metrics, y_test_original):
    """
    Realiza un análisis estadístico de los errores de predicción.
    
    Args:
        metrics: Diccionario con métricas y predicciones
        y_test_original: Valores reales (escala original)
        
    Returns:
        Dict con estadísticas de errores
    """
    if 'predictions' not in metrics:
        raise ValueError("Las predicciones no están disponibles en las métricas proporcionadas")
    
    # Calcular errores
    errors = y_test_original - metrics['predictions']
    percent_errors = (errors / y_test_original) * 100
    
    # Calcular estadísticas
    error_stats = {
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'std_error': np.std(errors),
        'max_error': np.max(errors),
        'min_error': np.min(errors),
        'mean_percent_error': np.mean(percent_errors),
        'median_percent_error': np.median(percent_errors),
        'errors': errors,
        'percent_errors': percent_errors
    }
    
    # Mostrar estadísticas principales
    print("\n=== Análisis de Errores ===")
    print(f"Error Medio: {error_stats['mean_error']:.2f}")
    print(f"Error Mediano: {error_stats['median_error']:.2f}")
    print(f"Desviación Estándar: {error_stats['std_error']:.2f}")
    print(f"Error Máximo: {error_stats['max_error']:.2f}")
    print(f"Error Mínimo: {error_stats['min_error']:.2f}")
    print(f"Error Porcentual Medio: {error_stats['mean_percent_error']:.2f}%")
    
    return error_stats

def evaluate_from_files(model_path=None, test_data_path=None, preprocessor_path=None):
    """
    Carga un modelo y datos de prueba desde archivos y realiza la evaluación.
    
    Args:
        model_path: Ruta al modelo guardado
        test_data_path: Ruta a los datos de prueba
        preprocessor_path: Ruta al preprocesador (opcional)
        
    Returns:
        Dict con métricas de evaluación
    """
    import tensorflow as tf
    import pandas as pd
    import joblib
    from dataloader import preprocess_data  # Importamos la función de preprocesamiento
    
    # Si no se proporciona ruta al modelo, pedir selección
    if model_path is None:
        model_path = select_model_file()
        if not model_path:
            print("No se seleccionó ningún archivo de modelo. Saliendo...")
            return None, None
    
    # Si no se proporciona ruta a los datos, pedir selección
    if test_data_path is None:
        test_data_path = select_data_file()
        if not test_data_path:
            print("No se seleccionó ningún archivo de datos. Saliendo...")
            return None, None
    
    # Si no se proporciona ruta al preprocesador, pedir selección
    if preprocessor_path is None:
        preprocessor_path = select_preprocessor_file()
        if not preprocessor_path:
            print("No se seleccionó ningún archivo de preprocesador. Intente cargar el modelo sin preprocesamiento...")
            return None, None
    
    # Verificar que los archivos existen
    if not os.path.exists(model_path):
        print(f"Error: No se encuentra el modelo en {model_path}")
        return None, None
    if not os.path.exists(test_data_path):
        print(f"Error: No se encuentran los datos en {test_data_path}")
        return None, None
    if preprocessor_path and not os.path.exists(preprocessor_path):
        print(f"Error: No se encuentra el preprocesador en {preprocessor_path}")
        return None, None
    
    print(f"\nCargando modelo: {model_path}")
    print(f"Cargando datos: {test_data_path}")
    if preprocessor_path:
        print(f"Cargando preprocesador: {preprocessor_path}")
    
    # Cargar modelo
    try:
        # Primer intento: sin objetos personalizados y sin compilación
        model = tf.keras.models.load_model(model_path, compile=False)
        # Recompilar manualmente con clases en lugar de strings
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[
                tf.keras.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanSquaredError()
            ]
        )
        print("Modelo cargado correctamente (recompilado).")
    except Exception as e:
        print(f"Error al cargar el modelo: {str(e)}")
        return None, None
    
    # Cargar datos de prueba
    try:
        # Cargar el archivo CSV o Excel
        if test_data_path.endswith('.csv'):
            raw_data = pd.read_csv(test_data_path)
        elif test_data_path.endswith('.xlsx'):
            raw_data = pd.read_excel(test_data_path)
        else:
            print(f"Error: Formato de archivo no soportado. Use .csv o .xlsx")
            return None, None
            
        print(f"Datos cargados correctamente. Registros: {len(raw_data)}")
        
        # Preprocesar los datos usando la misma función que en entrenamiento
        X, y, y_original = preprocess_data(raw_data)
        
        if X is None or y is None:
            print("Error durante el preprocesamiento de datos.")
            return None, None
            
        print("Datos preprocesados correctamente.")
    except Exception as e:
        print(f"Error al cargar o preprocesar los datos: {str(e)}")
        return None, None
    
    # Aplicar el preprocesador
    try:
        preprocessor = joblib.load(preprocessor_path)
        X_test = preprocessor.transform(X)
        y_test_log = y
        y_test_original = y_original
        
        print("Preprocesador aplicado correctamente.")
    except Exception as e:
        print(f"Error al aplicar el preprocesador: {str(e)}")
        return None, None
    
    # Evaluar modelo
    try:
        metrics = evaluate_model(model, X_test, y_test_log, y_test_original)
        print_metrics(metrics)
        
        # Analizar errores
        error_stats = analyze_errors(metrics, y_test_original)
        
        print("\nEvaluación completada con éxito.")
        return metrics, error_stats
    except Exception as e:
        print(f"Error durante la evaluación: {str(e)}")
        return None, None

if __name__ == "__main__":
    import argparse
    
    print("=== Módulo de Evaluación de Modelos de Regresión Lineal ===")
    print("Este módulo está diseñado para evaluar modelos de regresión que predicen precios de alquiler.")
    
    # Verificar si se proporcionan argumentos en línea de comandos
    import sys
    if len(sys.argv) > 1:
        # Configurar argumentos de línea de comandos
        parser = argparse.ArgumentParser(description='Evaluar modelo de predicción de alquiler')
        parser.add_argument('--model', type=str, default=None,
                            help='Ruta al modelo guardado (.h5)')
        parser.add_argument('--data', type=str, default=None,
                            help='Ruta a los datos de prueba (.csv)')
        parser.add_argument('--preprocessor', type=str, default=None,
                            help='Ruta al preprocesador (.joblib)')
        
        args = parser.parse_args()
        
        # Evaluar modelo con argumentos proporcionados
        metrics, error_stats = evaluate_from_files(args.model, args.data, args.preprocessor)
    else:
        # Modo interactivo con selección de archivos
        print("\nModo interactivo - Seleccione los archivos cuando se le solicite.")
        metrics, error_stats = evaluate_from_files()