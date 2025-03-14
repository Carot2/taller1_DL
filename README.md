# **Taller: Predicción del Precio de Alquiler de Viviendas con una Red Neuronal Backpropagation**

Este proyecto implementa un modelo de red neuronal para predecir el precio de alquiler de viviendas basado en características como tamaño, ubicación, número de habitaciones, etc.

![Redes Neuronales](https://miro.medium.com/v2/resize:fit:500/1*LB10KFg5J7yK1MLxTXcLdQ.jpeg)


## Índice
1. [Introducción a la problemática](#introducción-a-la-problemática)
2. [Exploración del dataset](#exploración-del-dataset)
   - [Descripción de las variables](#descripción-de-las-variables)
   - [Análisis de características y preprocesamiento](#análisis-de-las-características-y-preprocesamiento-de-datos)
3. [Implementación de la red neuronal](#implementación-de-la-red-neuronal)
4. [Estructuración del repositorio](#estructuración-del-repositorio)
5. [Instrucciones de uso](#instrucciones-de-uso-de-la-red-neuronal-y-rendimiento-obtenido)
   - [Requisitos previos](#requisitos-previos)
   - [Instalación](#instalación)
   - [Cómo entrenar el modelo](#cómo-entrenar-el-modelo-desde-trainpy)
   - [Cómo evaluar el modelo](#cómo-evaluar-el-modelo-usando-evaluatepy)
   - [Cómo hacer predicciones](#cómo-hacer-predicciones-con-predictpy)
6. [Rendimiento obtenido](#rendimiento-obtenido)
7. [Conclusiones](#conclusiones)


## ⚡️ Introducción a la problemática

### Importancia de la predicción de precios de alquiler
La predicción precisa de precios de alquiler es fundamental en el mercado inmobiliario actual, ya que permite:
- Optimizar la valoración de propiedades
- Identificar oportunidades de inversión
- Proporcionar transparencia en el mercado para compradores, vendedores y arrendatarios
- Facilitar la toma de decisiones basadas en datos para agentes inmobiliarios

### Casos de uso en la industria inmobiliaria
- **Portales inmobiliarios**: Estimación automatizada de precios para nuevos listados
- **Inversores**: Análisis de rentabilidad potencial de propiedades
- **Agencias inmobiliarias**: Valoración rápida y objetiva de propiedades
- **Gobierno**: Planificación urbana y políticas de vivienda basadas en tendencias de precios
- **Bancos**: Evaluación de riesgos para préstamos hipotecarios



## ⚡️ Exploración del dataset

Para desarrollar nuestra red neuronal backpropagation, trabajamos con un conjunto de datos con 4.747 registros de propiedades residenciales disponibles para alquiler. Este conjunto de datos ofrece  variedad de características que permiten un análisis del mercado inmobiliario.


### **Descripción de las variables**

| Nombre original en inglés | Descripción en Español |
|------------------------|-------------|
| **BHK** | Número de habitaciones, sala y cocina. |
| **Rent** | Precio de alquiler de la propiedad. |
| **Size** | Tamaño de la propiedad en pies cuadrados. |
| **Floor** | Piso en el que está ubicada la propiedad y número total de pisos del edificio. |
| **Area Type** | Tipo de área utilizada en el cálculo del tamaño (Superficie, Área de Alfombra o Área Construida). |
| **Area Locality** | Localidad donde está ubicada la propiedad. |
| **City** | Ciudad donde se encuentra la propiedad. |
| **Furnishing Status** | Estado de amueblado de la propiedad: Amueblado, Semiamueblado o Sin amueblar. |
| **Tenant Preferred** | Tipo de inquilino preferido por el dueño o agente. |
| **Bathroom** | Número de baños en la propiedad. |
| **Point of Contact** | Persona o entidad a la que se debe contactar para más información sobre la propiedad. |
| **Posted On** | Fecha en la que la propiedad fue publicada en la plataforma. |

- ---

### Análisis de las características y preprocesamiento de datos



Durante la fase exploratoria, se identificaron las siguientes características clave:

1. **Variables numéricas**: `Rent`, `Size`, `BHK`, `Bathroom`, `Floor`

![Distribución variables númericas](https://github.com/jeremiaspabon/taller1_DL/blob/main/graphics/distributionvarnuum.png)


2. **Variables categóricas**: `Area Locality`, `Area Type`, `City`, `Furnishing Status`, `Tenant Preferred`, `Point of Contact`, `Posted On`

   *Nota: Durante la exploración del dataset, se identificó que las variables `Posted On` y `Area Locality` tenían mucha variabilidad en sus registros y no tenían relevancia para la red neuronal, por lo que se eliminaron del dataset*

![Distribución variables categóricas](https://github.com/jeremiaspabon/taller1_DL/blob/main/graphics/distributionvarcat.png)


3. **Preprocesamiento aplicado**:

   **Análisis y transformación de la variable `Rent` (Target)**
   
   Durante el análisis exploratorio, observamos que la variable `Rent` (precio de alquiler) presentaba una fuerte asimetría positiva, con la mayoría de las propiedades concentradas en valores bajos y algunas propiedades de lujo creando una larga cola hacia la derecha. Esta distribución no gaussiana podía afectar negativamente el rendimiento de nuestro modelo de red neuronal.
   
   ![Distribución de variable Rent](https://github.com/jeremiaspabon/taller1_DL/blob/main/graphics/rent.png)
   
   Para abordar este problema, aplicamos una transformación logarítmica natural (ln(x+1)) que logró una distribución mucho más cercana a la normal, facilitando el aprendizaje del modelo y mejorando la estabilidad numérica durante el entrenamiento.
   
   **Análisis y transformación de la variable `Size`**
   
   De manera similar, la variable `Size` (tamaño de la propiedad) también presentaba una distribución asimétrica con valores atípicos que podían distorsionar el modelo.
   
   ![Distribución de variable Size](https://github.com/jeremiaspabon/taller1_DL/blob/main/graphics/size.png)
   
   Aplicamos la misma transformación logarítmica natural para normalizar su distribución, lo que mejoró significativamente el ajuste del modelo.
   
   **Otras transformaciones aplicadas:**
   
   - **Codificación One-Hot**: Para variables categóricas como `City` y `Furnishing Status`.
   - **Normalización**: Uso de StandardScaler para todas las variables numéricas.
   - **Limpieza de datos**: Eliminación de registros con valores faltantes o inconsistentes.


## ⚡️ Implementación de la red neuronal

Para abordar el problema de predicción de precios de alquiler, implementamos una red neuronal feed-forward con propagación hacia atrás (backpropagation). Este tipo de arquitectura es ideal para capturar relaciones complejas entre múltiples variables en problemas de regresión.

## Arquitectura del modelo

Después de múltiples experimentos, se determinó que la mejor arquitectura para este problema de regresión es:

```python
def crear_modelo_optimo(input_shape):
    """
    Construye una red neuronal mejorada para predecir el precio de alquiler.
    """
    model = keras.Sequential([
        layers.Dense(256, input_shape=(input_shape,)),
        layers.LeakyReLU(alpha=0.1),  # Permite que los gradientes fluyan mejor
        layers.Dropout(0.3),
        layers.Dense(128),
        layers.LeakyReLU(alpha=0.1),
        layers.Dropout(0.3),
        layers.Dense(64),
        layers.LeakyReLU(alpha=0.1),
        layers.Dense(1, activation="linear")  # Regresión, salida sin activación
    ])
    
    # Compilar el modelo con una tasa de aprendizaje mayor
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005),  
                  loss="mse",
                  metrics=["mae", "mse"])
    
    return model
```

### Características clave del modelo:

1. **Activación LeakyReLU**: 
   - A diferencia de ReLU tradicional, LeakyReLU permite un pequeño gradiente cuando la unidad no está activa (input negativo), lo que ayuda a evitar el problema de "neuronas muertas".
   - El parámetro alpha=0.1 determina la pendiente de la función cuando la entrada es negativa.

2. **Dropout (0.3)**:
   - Se utiliza para reducir el sobreajuste desactivando aleatoriamente el 30% de las neuronas durante cada paso de entrenamiento.
   - Mejora la generalización del modelo.

3. **Arquitectura decreciente (256 → 128 → 64 → 1)**:
   - El modelo comienza con un gran número de neuronas para capturar patrones complejos.
   - Reduce progresivamente el número de neuronas para condensar la información aprendida.

4. **Optimizador Adam**:
   - Se utiliza con una tasa de aprendizaje de 0.005, que mostró mejor rendimiento que el valor predeterminado de 0.001.

5. **Entrenamiento**:
   - Épocas: 100 (con early stopping)
   - Batch size: 32
   - Callbacks:
     - EarlyStopping: detiene el entrenamiento cuando no hay mejora en la validación
     - ReduceLROnPlateau: reduce la tasa de aprendizaje cuando el rendimiento se estanca



## ⚡️ Estructuración del repositorio

El proyecto está organizado de manera modular. A continuación se detalla la estructura de carpetas y archivos:
```
taller1_DL/
├── notebooks/                # Jupyter Notebooks para experimentos y análisis
│   ├── 01_exploracion.ipynb  # Análisis exploratorio de datos
│   ├── 02_entrenamiento.ipynb # Pruebas de entrenamiento del modelo
│   └── 03_evaluacion.ipynb   # Evaluación y visualización de resultados
│
├── src/                      # Código fuente del proyecto
│   ├── data_loader.py        # Carga y preprocesamiento de datos
│   ├── model.py              # Definición del modelo de DL
│   ├── train.py              # Script para entrenar el modelo
│   ├── evaluate.py           # Evaluación del modelo
│   ├── predict.py            # Predicción con el modelo entrenado
│   └── utils.py              # Funciones auxiliares
│
├── models/                   # Modelos entrenados y checkpoints
│   ├── model_v1.h5           # Modelo en TensorFlow/Keras
│   └── model_best.h5         # Mejor modelo encontrado
│
├── data/                     # Carpeta para almacenar datos (no incluida en Git)
│   └── README.md             # Instrucciones para obtener los datos
│
├── graphics/                 # Visualizaciones y gráficos generados
│   ├── distributionvarcat.png # Distribución de variables categóricas
│   ├── distributionvarnuum.png # Distribución de variables numéricas
│   ├── rent.png               # Análisis de la variable objetivo
│   └── size.png               # Análisis de tamaño de propiedades
│
├── requirements.txt          # Dependencias del proyecto
├── .gitignore                # Archivos a ignorar por Git
└── README.md                 # Documentación principal del proyecto
```

## 🛠Instrucciones de uso de la red neuronal y rendimiento obtenido

### Requisitos previos

- Python 3.9 o superior
- Git instalado
- IDE (recomendado: Visual Studio Code)
- Cuenta de GitHub

### Instalación

1. **Crear carpeta local para el proyecto**

   ```bash
   # Crear carpeta para el proyecto (Windows)
   mkdir C:\Users\{tu_usuario}\Documents\proyectos\prediccion_alquiler

   # Acceder a la carpeta
   cd C:\Users\{tu_usuario}\Documents\proyectos\prediccion_alquiler
   ```

2. **Configurar Git (si es la primera vez)**

   ```bash
   # Configurar nombre de usuario
   git config --global user.name "Tu Nombre"

   # Configurar correo electrónico con el que tienes la cuenta de GitHub
   git config --global user.email "tu_correo@ejemplo.com"
   ```

3. **Verificar estado de Git**

   ```bash
   git status
   ```

4. **Clonar el repositorio**

   ```bash
   git clone https://github.com/jeremiaspabon/taller1_DL.git
   cd taller1_DL
   ```

5. **Instalar dependencias**

   ```bash
   pip install -r requirements.txt
   ```

### 🤔 Cómo entrenar el modelo desde train.py

Para entrenar la Red Neuronal, ejecuta el siguiente comando:

```bash
python src/train.py
```

> **Nota**: Asegúrese de tener instaladas las librerías que se encuentran en el archivo requirements.txt:
> - pandas
> - openpyxl
> - numpy
> - scikit-learn
> - tabulate
> - statsmodels
> - seaborn
> - plotly
> - tensorflow


Este script realizará las siguientes acciones:
1. Le pedirá que seleccione el archivo csv "House_Rent_Dataset" `data/House_Rent_Dataset.csv`, para cargar la información del dataset.
2. Aplicación del preprocesamiento (transformación logarítmica, codificación one-hot, etc.)
3. División de datos (80% entrenamiento, 20% prueba)
4. Entrenamiento de la red neuronal con configuración optimizada
5. Evaluación del rendimiento
6. Guardado del mejor modelo en `best_model_checkpoint.h5`



### 🤔 Cómo evaluar el modelo usando evaluate.py

Para evaluar la Red Neuronal, ejecuta el siguiente comando:

```bash
python src/evaluate.py 
```

Este script:
1. Le pedirá que seleccione el archivo del modelo entrenado (best_model_checkpoint.h5) `models/best_model_checkpoint.h5`
2. Le solicitará seleccionar el archivo de datos de prueba (House_Rent_Dataset.csv) para evaluar el modelo  `data/House_Rent_Dataset.csv`
3. Le pedirá seleccionar el archivo del preprocesador (.joblib) que se utilizó durante el entrenamiento `models/preprocessor.joblib`
4. Aplicará el preprocesamiento correspondiente a los datos de prueba
5. Evaluará el modelo y generará múltiples métricas de rendimiento:
- En escala logarítmica: MSE, MAE, RMSE, R²
- En escala original: MSE, MAE, RMSE, R², MAPE

5. Realizará un análisis estadístico de los errores de predicción:

- Error medio y mediano
- Desviación estándar
- Errores máximos y mínimos
- Error porcentual medio

*Nota: Asegúrese de tener los siguientes archivos preparados antes de ejecutar la evaluación:
El modelo guardado (.h5) best_model_checkpoint.h5
El archivo de datos para prueba (House_Rent_Dataset.csv)
El archivo del preprocesador (.joblib) preprocessor.joblib*



### 🤔Cómo hacer predicciones con predict.py

Para realizar predicciones con nuevos datos, ejecute el siguiente comando:

```bash
python src/predict.py
```

Este script funciona en modo interactivo y realizará las siguientes acciones:

1. Le pedirá seleccionar el archivo CSV que contiene los datos para los cuales desea realizar predicciones (`data/ejemplo_prediccion.csv`)
2. Cargará automáticamente el modelo entrenado (`models/best_model_checkpoint.h5`) y el preprocesador (`models/preprocessor.joblib`)
3. Procesará los datos de entrada aplicando las mismas transformaciones utilizadas durante el entrenamiento
4. Realizará las predicciones y mostrará los resultados
5. Si el archivo de entrada contiene la columna `Rent`, el script también calculará y mostrará métricas de error (MAE, MAPE)

#### Formato de datos de entrada

Para que las predicciones funcionen correctamente, su archivo CSV  debe estar separado por comas y contener las siguientes columnas conservando el mismo orden. 
Ejemplo:

| BHK | Rent | Size | Floor | Area Type | City | Furnishing Status | Bathroom | Point of Contact | Posted On | Area Locality | Tenant Preferred |
|-----|------|------|-------|-----------|------|------------------|----------|-----------------|-----------|---------------|------------------|
| 2 | 10000 | 1100 | Ground out of 2 | Super Area | Kolkata | Unfurnished | 2 | Contact Agent | 2022-05-15 | Andheri East | Bachelors/Family |

> **Nota importante**: 

> - Asegúrese de que el formato de los datos sea consistente con el conjunto de datos de entrenamiento.

#### Ejemplo de archivo de predicción

Hemos proporcionado un archivo de ejemplo `data/ejemplo_prediccion.csv` que puede utilizar como plantilla para sus propios datos:

1. Descargue la plantilla desde la carpeta `data/`
2. Complete las columnas con los datos de las propiedades que desea predecir
3. Guarde el archivo con un nuevo nombre
4. Utilice este archivo al ejecutar `predict.py`

## ⚡️ Rendimiento obtenido

El modelo entrenado alcanzó los siguientes resultados en la evaluación:

### Métricas en escala logarítmica

| Métrica | Valor |
|---------|-------|
| MSE     | 0.2545 |
| MAE     | 0.3749 |
| RMSE    | 0.5044 |
| R²      | 0.7099 |

### Métricas en escala original

| Métrica | Valor |
|---------|-------|
| MSE     | 4,232,047,360.00 |
| MAE     | 15,156.02 |
| RMSE    | 65,054.19 |
| R²      | 0.3065 |
| MAPE    | 40.30% |

### Análisis de errores

El análisis detallado de los errores de predicción muestra:

* **Error medio**: 7,064.26
* **Error mediano**: -636.93
* **Desviación estándar**: 64,669.50
* **Error máximo**: 3,430,828.37
* **Error mínimo**: -127,801.39
* **Error porcentual medio**: -13.21%

El coeficiente de determinación (R²) en escala logarítmica de aproximadamente 0.71 indica que nuestro modelo explica el 71% de la varianza en los precios de alquiler transformados logarítmicamente. Sin embargo, el R² en escala original es menor (0.31), lo que refleja la dificultad de predecir valores absolutos en datos con alta variabilidad como los precios inmobiliarios.

El error porcentual medio absoluto (MAPE) de 40.30% sugiere que, en promedio, las predicciones tienen un margen de error significativo, lo cual es esperable en mercados inmobiliarios donde numerosos factores no capturados (como características específicas de la propiedad, condiciones del vecindario o tendencias del mercado local) pueden influir en los precios de alquiler.


## 🧠Conclusiones

1. La transformación logarítmica aplicada a las variables `Rent` y `Size` mejoró significativamente el rendimiento del modelo al normalizar sus distribuciones.

2. La arquitectura de capas densas con tamaños decrecientes (256→128→64→1) demostró ser efectiva para este problema.

3. El uso de LeakyReLU como función de activación, con un parámetro alpha de 0.1, superó al ReLU tradicional al permitir la propagación de gradientes incluso con entradas negativas.

4. La técnica de Dropout (30%) fue crucial para prevenir el sobreajuste y mejorar la generalización del modelo.

5. El modelo final consigue un R² de 0.71, lo que representa un buen equilibrio entre la complejidad del modelo y su capacidad predictiva.

6. Futuras mejoras podrían incluir:
   - Incorporación de variables geoespaciales más detalladas
   - Técnicas avanzadas de feature engineering
   - Arquitecturas más complejas como redes residuales o modelos de ensamble




## Autores del Proyecto 🤓

Este proyecto fue desarrollado por:

* [Jeremías Pabón](https://github.com/jeremiaspabon) 
* [Gersón Julián Rincón](https://github.com/Julk-ui) 
* [Andrés Bravo](https://github.com/pipebravo10) 
* [Carolina Tobaria](https://github.com/Carot2) 