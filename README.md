# Proyecto D: Sistema de Predicción de Calidad de Vino
Proyecto D: Sistema de Predicción de Calidad de Vino mediante Redes Neuronales Artificiales (ANN).

## Inteligencia Artificial Aplicada

**Curso:** BD-151 Inteligencia Artificial Aplicada  
**Profesor:** Osvaldo Gonzalez Chaves  
**Institución:** Colegio Universitario de Cartago  
**Área:** Big Data (BD)  
**Proyecto:** Proyecto D - Sistema de Predicción de Calidad de Vino  

---

## 1. Introducción

Este proyecto forma parte del curso **BD-151 Inteligencia Artificial Aplicada** y tiene como propósito desarrollar una solución completa de Inteligencia Artificial aplicada a un problema del mundo real.

El proyecto se enfoca en el análisis y predicción de la **calidad del vino**, utilizando datos relacionados con sus propiedades fisicoquímicas.

El desarrollo contempla el ciclo completo de un proyecto de Machine Learning e Inteligencia Artificial:

- Exploración y comprensión de los datos.
- Limpieza y preprocesamiento.
- Ingeniería de características.
- Preparación de conjuntos de entrenamiento y prueba.
- Desarrollo de modelos mediante Redes Neuronales Artificiales (ANN).
- Evaluación y comparación de modelos.
- Selección y almacenamiento del mejor modelo.
- Exposición del modelo mediante una API REST.
- Desarrollo de una interfaz web para realizar predicciones y visualizar resultados.

El proyecto utiliza Python como lenguaje principal y mantiene una arquitectura modular que separa los datos, notebooks, código reutilizable, modelos, API y frontend.

---

# 2. Objetivos

## 2.1 Objetivo general

Desarrollar un sistema basado en técnicas de Inteligencia Artificial capaz de analizar características fisicoquímicas de diferentes vinos y utilizar esta información para desarrollar modelos predictivos de calidad.

---

## 2.2 Objetivos específicos

Los principales objetivos del proyecto son:

1. Cargar y validar los datasets originales correspondientes a vinos tintos y blancos.

2. Unificar ambos conjuntos de datos manteniendo información sobre el tipo de vino.

3. Realizar un Análisis Exploratorio de Datos (EDA) utilizando estadísticas y visualizaciones.

4. Identificar posibles valores faltantes, registros duplicados, valores atípicos y patrones relevantes.

5. Preparar los datos para su utilización en modelos de Machine Learning.

6. Transformar variables categóricas a representaciones numéricas.

7. Crear variables objetivo para problemas de regresión y clasificación.

8. Separar los datos en conjuntos de entrenamiento y prueba.

9. Normalizar las variables predictoras antes del entrenamiento.

10. Diseñar y entrenar diferentes Redes Neuronales Artificiales (ANN).

11. Evaluar y comparar el rendimiento de los modelos desarrollados.

12. Seleccionar el modelo con mejores resultados según las métricas definidas.

13. Persistir los modelos entrenados y los objetos necesarios para el preprocesamiento.

14. Implementar una API REST mediante FastAPI.

15. Desarrollar una interfaz web mediante Streamlit.

16. Mantener documentación técnica y control de versiones del proyecto.

---

# 3. Problema

La calidad del vino puede relacionarse con diferentes propiedades fisicoquímicas presentes en cada muestra.

Entre las variables disponibles en los datasets se encuentran:

- Acidez fija.
- Acidez volátil.
- Ácido cítrico.
- Azúcar residual.
- Cloruros.
- Dióxido de azufre libre.
- Dióxido de azufre total.
- Densidad.
- pH.
- Sulfatos.
- Alcohol.
- Calidad.

El proyecto utilizará estas variables para estudiar la relación entre las características fisicoquímicas y la calidad registrada del vino.

---

# 4. Enfoque predictivo

El proyecto está diseñado para permitir dos perspectivas de análisis supervisado.

## 4.1 Regresión

Para regresión se mantiene como variable objetivo:

`quality`

El objetivo será desarrollar un modelo capaz de estimar el valor de calidad del vino utilizando diferentes características predictoras.

El problema puede representarse conceptualmente como:

`Características fisicoquímicas → quality`

---

## 4.2 Clasificación

También se genera una variable objetivo binaria llamada:

`high_quality`

Actualmente se utiliza el siguiente criterio:

`quality >= 7 → high_quality = 1`

`quality < 7 → high_quality = 0`

Por lo tanto:

- `0` representa vinos que no cumplen el criterio definido de calidad alta.
- `1` representa vinos clasificados como de calidad alta.

Esta transformación permite analizar el problema también como clasificación:

`Características fisicoquímicas → high_quality`

---

# 5. Tecnologías utilizadas

El proyecto considera las siguientes tecnologías y librerías:

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
- Joblib
- Jupyter Notebook
- FastAPI
- Pydantic
- Streamlit
- Git
- GitHub

---

# 6. Estructura del proyecto

La organización general del repositorio sigue la estructura solicitada para los proyectos del curso.

```text
Proyecto_D_Calidad_Vino/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── winequality-red.csv
│   │   └── winequality-white.csv
│   │
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train_regression.csv
│       ├── y_test_regression.csv
│       ├── y_train_classification.csv
│       └── y_test_classification.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocesamiento.ipynb
│   ├── 03_ANN_Modelo1.ipynb
│   ├── 04_ANN_Modelo2.ipynb
│   └── 05_Comparacion_Modelos.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_prep.py
│   ├── eda.py
│   │
│   └── train/
│       ├── __init__.py
│       ├── model1.py
│       ├── model2.py
│       └── utils.py
│
├── models/
│   ├── model1.keras
│   ├── model2.keras
│   ├── scaler.pkl
│   └── columnas.pkl
│
├── api/
│   ├── main.py
│   ├── schemas.py
│   └── predict.py
│
└── app/
    ├── Home.py
    │
    └── pages/
        ├── 1_Prediccion.py
        ├── 2_Analisis.py
        └── 3_Metricas.py
```

> Algunos archivos representan la estructura objetivo del proyecto y serán desarrollados progresivamente durante las diferentes etapas.

---

# 7. Descripción de carpetas

## `data/`

Contiene los datos utilizados durante el proyecto.

### `data/raw/`

Contiene los datasets originales sin procesamiento.

Los datos originales no deben ser modificados directamente. Las transformaciones se realizan mediante el pipeline de preprocesamiento.

### `data/processed/`

Contiene los datos resultantes después de aplicar las diferentes etapas de limpieza, transformación, división y normalización.

---

## `notebooks/`

Contiene los Jupyter Notebooks utilizados para experimentación, análisis y desarrollo de los modelos.

### `01_EDA.ipynb`

Notebook destinado al **Análisis Exploratorio de Datos (EDA)**.

Incluye progresivamente:

- Carga de datos.
- Inspección inicial.
- Dimensiones del dataset.
- Tipos de datos.
- Valores faltantes.
- Registros duplicados.
- Estadísticas descriptivas.
- Distribución de `quality`.
- Distribución de variables fisicoquímicas.
- Análisis de correlaciones.
- Análisis de posibles valores atípicos.
- Comparaciones entre vino tinto y blanco.
- Interpretación de resultados.

### `02_Preprocesamiento.ipynb`

Contiene el proceso relacionado con:

- Limpieza de datos.
- Transformación de columnas.
- Creación de variables objetivo.
- Encoding.
- División Train/Test.
- Normalización.
- Validación de los datos preparados.

### `03_ANN_Modelo1.ipynb`



### `04_ANN_Modelo2.ipynb`



### `05_Comparacion_Modelos.ipynb`



## `src/`

Contiene código modular y reutilizable.

El objetivo es evitar concentrar toda la lógica del proyecto directamente dentro de los notebooks.

---

## `src/config.py`

Centraliza parámetros y configuraciones utilizados en diferentes partes del proyecto.

Entre estos parámetros pueden encontrarse:

- Rutas de datos.
- Rutas de modelos.
- Directorios de salida.
- Umbral utilizado para `high_quality`.
- Porcentaje de datos utilizado para prueba.
- Semilla aleatoria.
- Configuraciones utilizadas por los modelos.

---

## `src/data_prep.py`

Contiene las clases utilizadas actualmente para carga y preprocesamiento.

Las clases principales son:

- `WineDataLoader`
- `WinePreprocessor`

---

## `src/eda.py`

Contiene funcionalidad reutilizable relacionada con el Análisis Exploratorio de Datos.

Esto permite separar la lógica de procesamiento de las visualizaciones e interpretaciones realizadas desde `01_EDA.ipynb`.

---

## `src/train/`

Contendrá los scripts relacionados con el entrenamiento de las Redes Neuronales Artificiales.

La estructura propuesta incluye:

- `model1.py`
- `model2.py`
- `utils.py`

`utils.py` permitirá mantener funciones compartidas entre diferentes modelos, por ejemplo para métricas, evaluación o generación de gráficas.

---

## `models/`

Está destinado a almacenar modelos entrenados y objetos necesarios durante inferencia.

La estructura contempla elementos como:

- Modelos Keras.
- Scaler utilizado durante entrenamiento.
- Información sobre las columnas utilizadas por los modelos.

---

## `api/`

Contendrá la implementación de una API REST mediante FastAPI.

La estructura propuesta incluye:

### `main.py`

Aplicación principal de FastAPI y definición de endpoints.

### `schemas.py`

Modelos utilizados para validar entradas y respuestas de la API.

### `predict.py`

Lógica necesaria para cargar el modelo y realizar inferencias.

---

## `app/`

Contendrá el frontend desarrollado mediante Streamlit.

La aplicación se plantea como una interfaz multipágina.

### `Home.py`

Página principal de la aplicación.

### `pages/1_Prediccion.py`

Página destinada a predicciones individuales.

### `pages/2_Analisis.py`

Página destinada al análisis de datos o lotes.

### `pages/3_Metricas.py`

Página destinada a mostrar información relacionada con métricas y rendimiento de los modelos.

---

# 8. Flujo general del proyecto

El pipeline conceptual del proyecto sigue las siguientes etapas:

1. Obtención de los datasets.
2. Validación de archivos.
3. Validación de columnas.
4. Carga de datos.
5. Identificación del tipo de vino.
6. Unificación de datasets.
7. Análisis Exploratorio de Datos.
8. Limpieza de datos.
9. Creación de variables objetivo.
10. Codificación de variables categóricas.
11. Separación entre variables predictoras y objetivos.
12. División Train/Test.
13. Normalización.
14. Entrenamiento de modelos ANN.
15. Evaluación de modelos.
16. Comparación de arquitecturas.
17. Selección del modelo.
18. Persistencia del modelo y objetos de preprocesamiento.
19. Desarrollo de API REST.
20. Desarrollo del frontend.
21. Pruebas y documentación.

---

# 9. Carga y unificación de datos

La clase:

`WineDataLoader`

es responsable de cargar y unificar los datasets correspondientes a vinos tintos y blancos.

---

## 9.1 Validación de archivos

Antes de realizar la lectura se comprueba que cada archivo exista.

Si un archivo requerido no puede encontrarse, el proceso genera un error para evitar continuar con datos incompletos.

---

## 9.2 Validación de columnas

Se mantiene una lista de columnas esperadas:

- `fixed acidity`
- `volatile acidity`
- `citric acid`
- `residual sugar`
- `chlorides`
- `free sulfur dioxide`
- `total sulfur dioxide`
- `density`
- `pH`
- `sulphates`
- `alcohol`
- `quality`

Antes de continuar se verifica que el dataset contenga estas columnas.

Si alguna no se encuentra, el proceso reporta cuáles columnas requeridas están ausentes.

---

## 9.3 Identificación del tipo de vino

Antes de realizar la combinación se agrega una nueva variable:

`wine_type`

Para vino tinto:

`wine_type = red`

Para vino blanco:

`wine_type = white`

Esto permite conservar el origen de cada observación después de unir ambos datasets.

---

## 9.4 Combinación

Los datasets se combinan utilizando Pandas para crear una única estructura sobre la cual realizar las etapas posteriores del proyecto.

---

# 10. Preprocesamiento

La clase:

`WinePreprocessor`

se encarga de preparar los datos para los modelos.

Actualmente incluye funcionalidades relacionadas con:

- Limpieza.
- Creación de targets.
- Encoding.
- División Train/Test.
- Normalización.
- Persistencia de datos procesados.

---

# 11. Limpieza de datos

La función de limpieza realiza inicialmente diferentes operaciones.

---

## 11.1 Normalización de nombres de columnas

Los nombres son:

- Convertidos a minúsculas.
- Limpiados de espacios al inicio o final.
- Transformados para reemplazar espacios por `_`.

Por ejemplo:

`fixed acidity`

se transforma a:

`fixed_acidity`

Esto facilita el acceso programático a las variables durante las siguientes etapas.

---

## 11.2 Duplicados

El pipeline contempla la eliminación de registros duplicados mediante Pandas.

Esta funcionalidad puede habilitarse o deshabilitarse mediante el parámetro correspondiente del método de limpieza.

---

## 11.3 Valores faltantes

La implementación actual elimina registros que presenten valores faltantes antes de continuar con el procesamiento.

---

## 11.4 Índice

Después de las operaciones de limpieza, el índice del DataFrame es reconstruido para mantener una secuencia ordenada.

---

# 12. Creación de la variable objetivo

Para permitir un problema adicional de clasificación se genera:

`high_quality`

utilizando el valor existente de:

`quality`

Con un umbral actualmente configurado en `7`:

`quality >= 7 → high_quality = 1`

`quality < 7 → high_quality = 0`

La variable original `quality` se conserva para permitir posteriormente experimentos de regresión.

---

# 13. Encoding del tipo de vino

Originalmente:

`wine_type`

es una variable categórica que contiene los valores:

- `red`
- `white`

Durante el preprocesamiento se transforma mediante variables dummy.

La implementación actual utiliza una representación binaria donde se genera una variable como:

`wine_type_white`

permitiendo representar numéricamente el tipo de vino.

---

# 14. Preparación de variables

El dataset es separado conceptualmente entre:

## Variables predictoras

`X`

Contiene las características utilizadas para realizar predicciones.

## Target para regresión

`y_regression = quality`

## Target para clasificación

`y_classification = high_quality`

Las variables objetivo se eliminan de `X` para evitar utilizarlas directamente como predictores.

---

# 15. División de entrenamiento y prueba

La configuración actual utiliza:

`test_size = 0.20`

Por lo tanto, el objetivo del pipeline es utilizar aproximadamente:

- 80 % para entrenamiento.
- 20 % para prueba.

También se utiliza:

`random_state = 42`

para mantener reproducibilidad durante las pruebas realizadas con la misma configuración.

En clasificación, la variable `high_quality` se utiliza para estratificar la separación.

---

# 16. Normalización

La implementación utiliza:

`StandardScaler`

El scaler se ajusta utilizando únicamente el conjunto de entrenamiento:

`X_train`

Posteriormente se utiliza el scaler ya ajustado para transformar:

- `X_train`
- `X_test`

Esto permite mantener la misma transformación para ambos conjuntos.

---

# 17. Persistencia de datos procesados

El pipeline está preparado para generar archivos como:

```text
X_train.csv
X_test.csv
y_train_regression.csv
y_test_regression.csv
y_train_classification.csv
y_test_classification.csv
```

Estos archivos se almacenan dentro de:

`data/processed/`

También puede persistirse el scaler utilizado durante el proceso mediante Joblib.

---

# 18. Redes Neuronales Artificiales

Una etapa posterior del proyecto contempla el diseño y entrenamiento de diferentes Redes Neuronales Artificiales.

Los modelos serán desarrollados de manera independiente para permitir experimentar con diferentes arquitecturas y configuraciones.

La estructura contempla:

`03_ANN_Modelo1.ipynb`

y:

`04_ANN_Modelo2.ipynb`

junto con los scripts reutilizables correspondientes dentro de:

`src/train/`

---

# 19. Comparación de modelos

El notebook:

`05_Comparacion_Modelos.ipynb`

será utilizado para comparar los resultados obtenidos por los modelos.

La selección del modelo final se realizará según las métricas definidas durante la etapa de modelado y de acuerdo con el tipo de problema abordado.

Esta sección será actualizada con los resultados reales obtenidos durante los experimentos.

---

# 20. API REST

Una vez seleccionado el modelo, el proyecto contempla su integración mediante una API REST desarrollada con FastAPI.

La API permitirá separar la lógica del modelo de las aplicaciones que consuman sus predicciones.

La implementación se mantendrá dentro de:

`api/`

Esta sección será documentada con sus endpoints, formatos de entrada y respuestas una vez completada la implementación.

---

# 21. Frontend

El frontend será desarrollado utilizando Streamlit.

La estructura definida permite trabajar con una aplicación multipágina que incluirá progresivamente:

- Página principal.
- Predicciones individuales.
- Análisis.
- Métricas del modelo.

La implementación se mantendrá dentro de:

`app/`

---

# 22. Instalación

## 22.1 Clonar el repositorio

```text
git clone https://github.com/MauOrtizC/Proyecto-D-Sistema-de-Prediccion-de-Calidad-de-Vino.git
```

Ingresar al directorio:

```text
cd Proyecto-D-Sistema-de-Prediccion-de-Calidad-de-Vino
```

---

## 22.2 Crear un entorno virtual

En Windows:

```text
python -m venv .venv
```

---

## 22.3 Activar el entorno virtual

Desde PowerShell:

```text
.venv\Scripts\activate
```

Cuando esté activo se debería observar el nombre del entorno virtual en la terminal.

---

## 22.4 Instalar dependencias

Las dependencias del proyecto se mantienen en:

`requirements.txt`

Para instalarlas:

```text
python -m pip install -r requirements.txt
```

---

# 23. Dependencias principales

El proyecto contempla dependencias como:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
joblib
jupyter
ipykernel
fastapi
uvicorn
pydantic
streamlit
```

Las versiones específicas utilizadas durante el desarrollo deben mantenerse en `requirements.txt` conforme avance el proyecto.

---

# 24. Ejecución de los notebooks

Los notebooks deben ejecutarse siguiendo el orden lógico de desarrollo:

1. `01_EDA.ipynb`
2. `02_Preprocesamiento.ipynb`
3. `03_ANN_Modelo1.ipynb`
4. `04_ANN_Modelo2.ipynb`
5. `05_Comparacion_Modelos.ipynb`

Este orden permite que cada etapa documente progresivamente la exploración, transformación, modelado y evaluación.

---

# 25. Uso de Inteligencia Artificial durante el desarrollo

Durante el desarrollo del proyecto se utilizó **Inteligencia Artificial como una herramienta de apoyo para definir, organizar y revisar el flujo lógico del proyecto y del código**.

La asistencia de IA se utilizó principalmente durante la planificación del pipeline.

A partir de los objetivos académicos del proyecto se definió una secuencia de trabajo que incluye:

1. Carga.
2. Validación.
3. Unificación.
4. EDA.
5. Limpieza.
6. Creación de variables objetivo.
7. Encoding.
8. Train/Test.
9. Normalización.
10. Modelado.
11. Evaluación.
12. Despliegue.

La IA también fue utilizada como apoyo para separar las responsabilidades entre diferentes componentes.

Por ejemplo:

### `WineDataLoader`

Se definió para concentrar responsabilidades relacionadas con:

- Validación de archivos.
- Carga de datos.
- Validación de columnas.
- Identificación del tipo de vino.
- Unificación de datasets.
- Almacenamiento del dataset combinado.

### `WinePreprocessor`

Se definió para concentrar:

- Limpieza.
- Creación de variables objetivo.
- Encoding.
- Separación de los datos.
- Normalización.
- Persistencia de los datasets procesados.

---

## 25.1 Propósito del uso de IA

La Inteligencia Artificial no se utilizó únicamente como generador de código.

También fue utilizada como herramienta de asistencia para:

- Organizar el problema.
- Definir el flujo de desarrollo.
- Analizar alternativas de implementación.
- Dividir responsabilidades entre componentes.
- Comprender fragmentos de código.
- Revisar la lógica implementada.
- Identificar posibles problemas.
- Mejorar documentación.
- Explicar conceptos relacionados con Machine Learning.
- Mantener una estructura modular durante el desarrollo.

Las propuestas generadas mediante asistencia de IA fueron revisadas y adaptadas a las necesidades particulares del proyecto.

El objetivo de utilizar esta herramienta fue complementar el proceso de aprendizaje y desarrollo, manteniendo como parte fundamental la comprensión de la lógica implementada.

---

# 26. Control de versiones

El proyecto utiliza Git para mantener un historial de cambios y GitHub como repositorio remoto.

El desarrollo utiliza ramas para trabajar funcionalidades antes de incorporarlas a la rama principal.

Por ejemplo:

`main`

y:

`feature/eda-preprocessing`

Durante el desarrollo, los cambios relacionados con EDA y preprocesamiento pueden mantenerse en la rama correspondiente antes de integrarse a `main`.

---

## Flujo básico de trabajo

Verificar cambios:

```text
git status
```

Agregar cambios:

```text
git add .
```

Crear commit:

```text
git commit -m "Descripción del cambio"
```

Enviar cambios:

```text
git push
```

---

# 27. Principios de organización del código

El proyecto intenta mantener diferentes responsabilidades separadas.

Los notebooks están destinados principalmente a:

- Experimentación.
- Visualización.
- Interpretación.
- Comparación de resultados.

Mientras que `src/` mantiene:

- Clases.
- Funciones reutilizables.
- Configuraciones.
- Lógica de entrenamiento.

Esta separación facilita la reutilización del código durante las diferentes etapas del proyecto.

---

# 28. Estado actual del proyecto

El proyecto se encuentra actualmente en desarrollo.

## Estructura y datos

- [x] Creación del repositorio.
- [x] Creación de la estructura inicial.
- [x] Incorporación de datasets de vino tinto y blanco.
- [x] Configuración inicial del entorno Python.

## Carga y preprocesamiento

- [x] Implementación inicial de `WineDataLoader`.
- [x] Validación de archivos.
- [x] Validación de columnas.
- [x] Unificación de datasets.
- [x] Implementación inicial de `WinePreprocessor`.
- [x] Creación de `high_quality`.
- [x] Encoding de `wine_type`.
- [x] Implementación inicial de Train/Test split.
- [x] Implementación inicial de normalización.
- [ ] Validación final del pipeline de preprocesamiento.

## EDA

- [x] Creación de `01_EDA.ipynb`.
- [ ] Análisis exploratorio completo.
- [ ] Visualizaciones finales.
- [ ] Análisis de correlaciones.
- [ ] Análisis de valores atípicos.
- [ ] Documentación de conclusiones del EDA.

## Modelado

- [ ] Desarrollo ANN Modelo 1.
- [ ] Entrenamiento ANN Modelo 1.
- [ ] Desarrollo ANN Modelo 2.
- [ ] Entrenamiento ANN Modelo 2.
- [ ] Evaluación de modelos.
- [ ] Comparación de modelos.
- [ ] Selección del modelo final.
- [ ] Persistencia del modelo.

## Despliegue

- [ ] Desarrollo API REST con FastAPI.
- [ ] Validación de entradas mediante Pydantic.
- [ ] Endpoint de predicción.
- [ ] Desarrollo frontend con Streamlit.
- [ ] Página de predicciones.
- [ ] Página de análisis.
- [ ] Página de métricas.
- [ ] Integración frontend/API/modelo.

## Documentación

- [x] README inicial.
- [x] Documentación del uso de IA.
- [ ] Documentación de resultados del EDA.
- [ ] Documentación de modelos.
- [ ] Documentación de API.
- [ ] Documentación final del proyecto.

---

# 29. Próximas etapas

Las siguientes etapas contempladas son:

1. Completar el Análisis Exploratorio de Datos.
2. Validar el proceso completo de preprocesamiento.
3. Generar los conjuntos de entrenamiento y prueba definitivos.
4. Diseñar el primer modelo ANN.
5. Diseñar una segunda arquitectura ANN.
6. Entrenar y evaluar ambos modelos.
7. Comparar resultados.
8. Seleccionar y persistir el modelo final.
9. Crear la API con FastAPI.
10. Crear la interfaz con Streamlit.
11. Integrar los diferentes componentes.
12. Completar la documentación y conclusiones.

---

# 30. Repositorio

El código fuente del proyecto está alojado en GitHub:

**Repositorio:**  
`https://github.com/MauOrtizC/Proyecto-D-Sistema-de-Prediccion-de-Calidad-de-Vino`

---

# 31. Autores

Proyecto desarrollado como parte del curso:

**BD-151 Inteligencia Artificial Aplicada**

**Profesor:** Osvaldo Gonzalez Chaves  
**Colegio Universitario de Cartago**  
**Área de Big Data**

Integrantes del proyecto:

- Caleb
- Mariana
- Andrey
- Mauricio

---

# 32. Nota académica

Este repositorio corresponde a un proyecto académico desarrollado para aplicar de forma integrada conocimientos relacionados con:

- Análisis de datos.
- Machine Learning.
- Redes Neuronales Artificiales.
- Preparación y transformación de datos.
- Evaluación de modelos.
- Desarrollo de APIs.
- Desarrollo de aplicaciones.
- Control de versiones.
- Documentación técnica.
- Uso responsable de herramientas de Inteligencia Artificial como asistencia durante el proceso de desarrollo.

El proyecto se encuentra en desarrollo, por lo que la estructura, modelos, resultados y documentación continuarán evolucionando conforme se completen las diferentes etapas del curso.
