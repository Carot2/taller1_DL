# Graphics

Esta carpeta contiene visualizaciones generadas durante el análisis exploratorio de datos y la evaluación del modelo para el proyecto de predicción de precios de alquiler.

## Contenido

### Distribuciones de Variables Categóricas

Estas visualizaciones muestran la distribución de las principales variables categóricas en el conjunto de datos:

- **distributionvarcat.png**: Gráficos de barras que muestran:
  - Distribución de Area Type (Super Area, Carpet Area, Built Area)
  - Distribución de City (Mumbai, Chennai, Bangalore, Hyderabad, Delhi, Kolkata)
  - Distribución de Furnishing Status (Semi-Furnished, Unfurnished, Furnished)
  - Distribución de Tenant Preferred (Bachelors/Family, Bachelors, Family)
  - Distribución de Point of Contact (Contact Owner, Contact Agent, Contact Builder)

### Distribuciones de Variables Numéricas

Estas visualizaciones muestran las distribuciones de las principales variables numéricas:

- **distributionvarnum.png**: Histogramas y curvas de densidad mostrando:
  - BHK (Bedroom, Hall, Kitchen): Distribución discreta con picos en 1, 2 y 3 BHK
  - Rent: Distribución altamente sesgada hacia la derecha
  - Size: Distribución sesgada con la mayoría de propiedades menores a 2000 unidades
  - Bathroom: Distribución discreta con concentraciones en 1, 2, 3 y 4 baños

### Transformaciones de Variables

Estas visualizaciones demuestran las transformaciones aplicadas a las variables clave para mejorar sus distribuciones:

- **rent.png**: Conjunto de gráficos mostrando las transformaciones del precio de alquiler:
  - Distribución Original de Precios: Altamente sesgada hacia la derecha
  - QQ-Plot Original: Muestra desviación significativa de la normalidad
  - Boxplot Original: Evidencia de múltiples valores atípicos
  - Distribución Log-Natural: Transformación logarítmica que mejora significativamente la normalidad
  - QQ-Plot Log-Natural: Muestra mejor ajuste a la normalidad
  - Distribución Box-Cox: Transformación alternativa para normalizar los datos

- **size.png**: Visualizaciones que muestran la distribución y transformación de la variable Size (tamaño de la propiedad).

### Importancia de Características

- **feature_importance.png**: Gráfico que muestra la importancia relativa de cada característica para el modelo final.

### Evaluación del Modelo

- **model_comparison.png**: Comparación de rendimiento entre diferentes configuraciones de modelos
- **error_distribution.png**: Histograma mostrando la distribución de errores de predicción
- **predictions_vs_actual.png**: Gráfico de dispersión mostrando predicciones vs valores reales
- **city_performance.png**: Gráfico de barras mostrando el rendimiento del modelo por ciudad

## Uso

Estas visualizaciones se utilizan en los notebooks del proyecto para:

1. **Exploración**: Entender la distribución de los datos y guiar decisiones de preprocesamiento
2. **Transformación**: Justificar transformaciones aplicadas a variables sesgadas
3. **Evaluación**: Analizar el rendimiento del modelo y sus limitaciones
4. **Documentación**: Comunicar hallazgos clave y decisiones metodológicas

## Notas Técnicas

- Las visualizaciones fueron generadas utilizando matplotlib y seaborn
- Los histogramas incluyen curvas de densidad KDE para mejor interpretación
- Las transformaciones logarítmicas y Box-Cox se aplicaron para normalizar la distribución de precios
- Los QQ-plots evalúan la normalidad de las distribuciones comparándolas con una distribución normal teórica