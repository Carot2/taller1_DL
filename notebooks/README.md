# Notebooks

Este directorio contiene los cuadernos Jupyter utilizados para el desarrollo, entrenamiento y evaluación de nuestro modelo de predicción de alquiler de viviendas utilizando redes neuronales.

## Descripción General

Estos notebooks documentan nuestro flujo de trabajo completo de ciencia de datos, desde el análisis exploratorio hasta la implementación del modelo. Hemos utilizado un enfoque estructurado para construir un modelo de aprendizaje profundo que predice los precios de alquiler de viviendas basándose en diversas características.

## Descripción de los Notebooks

### 1. `01_exploracion.ipynb`
- Exploración inicial de datos y comprensión
- Análisis estadístico del Dataset de Alquiler de Viviendas
- Visualizaciones de relaciones y distribuciones clave
- Identificación de valores atípicos y patrones
- Limpieza y preparación de datos
- Ingeniería de características (incluyendo extracción de información de pisos)
- Manejo de variables categóricas
- Transformación logarítmica de valores de alquiler y tamaño
- Tratamiento de valores faltantes

### 2. `02_entrenamiento.ipynb`
- Diseño de arquitectura de redes neuronales
- Implementación de diferentes configuraciones de modelos:
  - Modelos con regularización Dropout
  - Modelos con regularización L2
  - Modelos combinados con Dropout, L2 y Normalización por Lotes
- Experimentación con hiperparámetros
- Entrenamiento con parada temprana y reducción de tasa de aprendizaje

### 3. `03_evaluacion.ipynb`
- Evaluación exhaustiva del modelo utilizando varias métricas:
  - Error Cuadrático Medio (MSE)
  - Error Absoluto Medio (MAE)
  - Error Porcentual Absoluto Medio (MAPE)
  - Coeficiente de determinación R²
- Comparación de diferentes arquitecturas de modelos
- Análisis de errores de predicción
- Análisis de rendimiento específico por ciudad
- Serialización del modelo
- Integración con pipeline de preprocesamiento
- Ejemplo de flujo de trabajo de predicción

## Características Principales

- **Ingeniería de Características**: Extracción personalizada de información de pisos a partir de datos de texto
- **Transformación de Datos**: Transformación logarítmica para manejar distribuciones de precios sesgadas
- **Comparación de Regularización**: Comparación sistemática de diferentes técnicas de regularización
- **Análisis Geográfico**: Evaluación de rendimiento específico por ciudad
- **Métricas Exhaustivas**: Evaluación utilizando métricas tanto en escala logarítmica como original

## Uso

Cada notebook está diseñado para ejecutarse secuencialmente, pero también puede utilizarse de forma independiente si los archivos de datos requeridos están disponibles. Los notebooks incluyen comentarios detallados y celdas markdown que explican la lógica detrás de cada paso.

## Resultados

Nuestro modelo con mejor rendimiento logró:
- Puntuación R² superior a 0.90 en el conjunto de prueba
- MAPE de aproximadamente 15% en el conjunto de prueba
- Rendimiento consistente en diferentes ciudades con algunas variaciones geográficas

## Dependencias

Los notebooks requieren las siguientes bibliotecas principales:
- pandas
- numpy
- matplotlib
- seaborn
- tensorflow
- scikit-learn

## Arquitectura del Modelo

El modelo final utiliza una red neuronal con:
- Múltiples capas densas (128 → 64 → 32 neuronas)
- Activación ReLU
- Regularización Dropout (0.2)
- Regularización L2 (0.01)
- Normalización por lotes
- Optimizador Adam con tasa de aprendizaje dinámica

## Trabajo Futuro

- Experimentar con arquitecturas más complejas (CNNs para características espaciales)
- Incorporar elementos de series temporales para predicción estacional
- Desarrollar métodos de ensamblaje combinando múltiples tipos de modelos
- Mayor ingeniería de características para capturar características del vecindario