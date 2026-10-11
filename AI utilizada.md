# Uso responsable de la IA
## En esta parte se pretende documentar el uso de la IA en el proyecto 

### data-prep.py

Para el desarrollo de esta sección del proyecto se utilizó Inteligencia Artificial como herramienta de apoyo para definir y organizar el flujo que debía seguir el procesamiento de los datos.

La IA ayudó principalmente a estructurar el problema en diferentes etapas, permitiendo separar las responsabilidades del código y mantener una secuencia lógica de trabajo.

A partir de este análisis se definió el siguiente flujo:

1. Carga de los datasets: leer de forma independiente los archivos correspondientes a vino tinto y vino blanco.
2. Validación de los archivos: comprobar que los archivos existan antes de intentar procesarlos.
3. Validación de la estructura: verificar que ambos datasets contengan las columnas necesarias para continuar con el análisis.
4. Identificación del tipo de vino: agregar la variable wine_type para conservar la información sobre si cada registro corresponde a vino tinto o blanco.
5. Unificación de los datasets: combinar ambos conjuntos de datos en un único DataFrame que pueda utilizarse durante las siguientes etapas del proyecto.
6. Limpieza de los datos: estandarizar los nombres de las columnas y revisar valores faltantes y registros duplicados.
7. Creación de variables objetivo: mantener quality para el problema de regresión y crear high_quality como una variable binaria para abordar también el problema desde clasificación.
8. Transformación de variables categóricas: convertir wine_type a una representación numérica que posteriormente pueda ser utilizada por los modelos.
9. Separación de los datos: dividir las variables predictoras y las variables objetivo en conjuntos de entrenamiento y prueba.
10. Normalización: preparar las variables numéricas utilizando StandardScaler, ajustándolo con los datos de entrenamiento y utilizando posteriormente esa transformación con los datos de prueba.
11. Persistencia de los resultados: guardar los datasets procesados y el scaler para poder reutilizarlos posteriormente durante el entrenamiento y evaluación de los modelos.
A partir de este flujo se organizaron dos clases principales:

- WineDataLoader se encarga de la carga, validación y unificación de los datasets.
- WinePreprocessor se encarga de la limpieza, transformación, preparación, división y normalización de los datos.

El uso de IA en esta etapa consistió en apoyar la definición de una estructura lógica y modular para el pipeline de datos. Cada etapa propuesta fue posteriormente revisada para comprender su propósito dentro del proyecto y realizar los ajustes necesarios según los objetivos del análisis de calidad del vino.

De esta manera, la IA funcionó como una herramienta de asistencia para organizar el desarrollo, mientras que la implementación final se integró y revisó de acuerdo con los requerimientos específicos del proyecto.

