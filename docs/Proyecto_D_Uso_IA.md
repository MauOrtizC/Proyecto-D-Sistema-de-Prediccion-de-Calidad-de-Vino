# Uso responsable de la inteligencia artificial

## Proyecto D: Sistema de Predicción de Calidad de Vino mediante Redes Neuronales Artificiales

### 1. Introducción

Durante el desarrollo del Proyecto D se utilizaron herramientas de inteligencia artificial generativa como recursos de apoyo académico y técnico en diferentes etapas del trabajo.

Estas herramientas facilitaron la organización de procesos, el análisis de información, la comprensión de conceptos relacionados con redes neuronales artificiales, la resolución de problemas técnicos y la elaboración de documentación.

Su utilización estuvo orientada a complementar el aprendizaje y apoyar las actividades de programación, experimentación e integración del sistema.

A continuación, se describen los usos realizados por cada integrante del equipo, diferenciando las actividades desarrolladas y el tipo de asistencia recibida.

---

## 2. Ortiz Coronado, Mauricio

### 2.1. Organización del procesamiento de datos (`data-prep.py`)

Se utilizó inteligencia artificial como herramienta de apoyo para definir y organizar el flujo de procesamiento de los conjuntos de datos correspondientes a vinos tintos y blancos.

La asistencia permitió estructurar el problema en etapas independientes, mantener una secuencia lógica de ejecución y distribuir responsabilidades entre los diferentes componentes del código.

El procesamiento se organizó mediante las siguientes etapas:

1. **Carga de los conjuntos de datos:** lectura independiente de los archivos de vinos tintos y blancos.
2. **Validación de archivos:** comprobación de la existencia de los archivos requeridos.
3. **Validación de estructura:** revisión de las columnas necesarias para continuar con el procesamiento.
4. **Identificación del tipo de vino:** incorporación de la variable `wine_type` para distinguir vinos tintos y blancos.
5. **Unificación de datos:** integración de ambos conjuntos en un único DataFrame.
6. **Limpieza de datos:** estandarización de nombres de columnas y revisión de valores faltantes y registros duplicados.
7. **Creación de variables objetivo:** utilización de `quality` para regresión y creación de `high_quality` para clasificación binaria.
8. **Transformación de variables categóricas:** conversión del tipo de vino a una representación numérica.
9. **Separación de datos:** división entre variables predictoras, variables objetivo y conjuntos de entrenamiento y prueba.
10. **Normalización:** aplicación de `StandardScaler`, ajustándolo con los datos de entrenamiento y reutilizando la transformación con los datos de prueba.
11. **Persistencia:** almacenamiento de los datos procesados y del escalador para su utilización posterior.

A partir de esta organización se definieron dos clases principales:

- **`WineDataLoader`:** responsable de cargar, validar y unificar los conjuntos de datos.
- **`WinePreprocessor`:** encargado de limpiar, transformar, dividir, normalizar y preparar los datos.

La inteligencia artificial funcionó como apoyo en el diseño de una estructura lógica y modular. Las etapas propuestas se revisaron posteriormente para comprender su funcionamiento y adaptarlas a las necesidades específicas del proyecto.

### 2.2. Elaboración inicial del archivo `README.md`

Se utilizó inteligencia artificial para apoyar la elaboración de una versión inicial del archivo `README.md`, con el objetivo de organizar la descripción del proyecto y facilitar su comprensión.

El documento se consideró una versión preliminar, pendiente de revisión y actualización para reflejar correctamente la estructura, las dependencias y las instrucciones definitivas del repositorio.

### 2.3. Resolución de problemas de compatibilidad con TensorFlow

Se recurrió a la inteligencia artificial para investigar un problema de compatibilidad relacionado con TensorFlow.

Durante el diagnóstico se identificó que se estaba utilizando Python 3.14 y que, para la configuración de TensorFlow empleada, era necesario trabajar con Python 3.12.

La asistencia permitió orientar la búsqueda de una solución y comprender la importancia de verificar la compatibilidad entre las versiones de Python y las bibliotecas utilizadas para el aprendizaje automático.

---

## 3. Calderón Vega, Andrey

### 3.1. Interpretación del análisis exploratorio de datos (EDA)

Se utilizó inteligencia artificial como apoyo para comprender, revisar y mejorar la interpretación de los resultados obtenidos durante el análisis exploratorio de datos.

Las consultas estuvieron orientadas a examinar resúmenes estadísticos, distribuciones de las características fisicoquímicas y relaciones entre las variables y la puntuación de calidad del vino.

También se solicitó asistencia para interpretar correlaciones, comprender el comportamiento de las variables y mejorar la explicación de los resultados obtenidos.

Este apoyo permitió organizar las observaciones del análisis en explicaciones técnicas más claras para la documentación del proyecto.

### 3.2. Investigación de estrategias para mejorar las métricas

Durante el entrenamiento y evaluación de los modelos neuronales, se utilizó inteligencia artificial para investigar alternativas orientadas a mejorar el desempeño de las configuraciones desarrolladas.

Las consultas se complementaron con búsquedas de información en Internet relacionadas con técnicas de entrenamiento, regularización y evaluación de modelos.

Entre los temas estudiados se incluyeron:

- Arquitecturas de redes neuronales y cantidades de neuronas.
- Regularización mediante Dropout.
- Utilización de EarlyStopping.
- Tratamiento del desbalance de clases.
- Interpretación de métricas como accuracy, precision, recall y F1-score.
- Evaluación del efecto de los umbrales de clasificación.

La inteligencia artificial contribuyó a comprender las ventajas y limitaciones de diferentes estrategias. Los resultados de los experimentos se analizaron a partir de las métricas generadas mediante la ejecución de los notebooks.

### 3.3. Apoyo en las pruebas funcionales del sistema

Se utilizó inteligencia artificial como herramienta de asistencia para preparar y revisar pruebas funcionales de la API REST y de la interfaz web.

El apoyo incluyó la preparación de datos de entrada, la interpretación de respuestas JSON y la comparación entre los resultados obtenidos mediante FastAPI y Streamlit.

Durante las pruebas se revisó que ambas aplicaciones recibieran las mismas características fisicoquímicas al comparar predicciones.

Asimismo, se analizó la diferencia entre las salidas de los dos modelos: la puntuación continua de regresión y la probabilidad de clasificación binaria.

La asistencia permitió identificar aspectos de la lógica de presentación que podían provocar diferencias entre las etiquetas mostradas por el dashboard y las respuestas de la API.

Las recomendaciones se contrastaron mediante la revisión del código y la ejecución de pruebas locales.

### 3.4. Elaboración y revisión de documentación técnica

Se utilizó inteligencia artificial para apoyar la estructuración, ampliación y redacción de diferentes apartados de la documentación técnica.

La asistencia se orientó a explicar procedimientos de entrenamiento, interpretar métricas, organizar los resultados de los experimentos y describir los componentes que integran el sistema.

La documentación se preparó tomando como referencia el código, los notebooks y los resultados registrados, procurando mantener la correspondencia entre las explicaciones y el desarrollo realizado.

---

## 4. Figueroa Gómez, Caleb

### 4.1. Diseño y desarrollo del modelo neuronal de regresión

Se utilizó inteligencia artificial como herramienta de asistencia para estructurar, configurar y desarrollar el Modelo ANN 1, encargado de estimar la puntuación numérica de calidad del vino.

La asistencia estuvo relacionada con la organización del código, la comprensión de las capas neuronales y la revisión de distintas configuraciones de arquitectura.

También se utilizó como recurso de consulta para comprender aspectos del entrenamiento, como las funciones de activación, los optimizadores, la función de pérdida y las técnicas de regularización.

Estas consultas contribuyeron al aprendizaje de los conceptos necesarios para implementar una red neuronal orientada a regresión.

### 4.2. Evaluación y comparación de modelos

Se utilizó inteligencia artificial como apoyo durante el análisis comparativo de las diferentes configuraciones y enfoques implementados.

La asistencia permitió facilitar la interpretación de las métricas de evaluación y comprender las diferencias observadas en el rendimiento de las arquitecturas.

Además, contribuyó a identificar aspectos relevantes para comparar los resultados experimentales y reconocer las configuraciones con mejor desempeño según las métricas utilizadas.

Las conclusiones sobre el rendimiento se fundamentaron en los resultados generados durante las pruebas y evaluaciones realizadas.

### 4.3. Depuración de errores y compatibilidad de dependencias

Durante el proceso de desarrollo se recurrió a la inteligencia artificial para investigar errores de programación, dificultades de ejecución y problemas de compatibilidad entre bibliotecas.

Las consultas facilitaron la interpretación de mensajes de error, la identificación de posibles causas y la búsqueda de alternativas para resolver las incidencias técnicas.

Las soluciones propuestas se utilizaron como orientación y se verificaron mediante su implementación y ejecución en el entorno de trabajo.

---

## 5. Víquez Guzmán, Mariana

### 5.1. Gestión de ramas e integración del trabajo mediante GitHub

Se utilizó inteligencia artificial como recurso de consulta para comprender los procedimientos necesarios para integrar el trabajo desarrollado en diferentes ramas de Git y GitHub.

En particular, se solicitó orientación sobre cómo vincular o integrar una rama con otra, con el objetivo de disponer de los modelos neuronales almacenados en el repositorio dentro de la rama utilizada para desarrollar la interfaz web.

La asistencia permitió comprender el procedimiento general de integración de cambios entre ramas y la importancia de mantener actualizados los archivos compartidos por el equipo.

Este apoyo estuvo relacionado con la organización del trabajo colaborativo y la disponibilidad de los artefactos necesarios para ejecutar las predicciones desde la aplicación.

### 5.2. Organización visual de la interfaz web con Streamlit

Se utilizó inteligencia artificial para consultar alternativas de organización y presentación de los componentes visuales del dashboard desarrollado mediante Streamlit.

Las consultas estuvieron orientadas a comprender cómo distribuir los campos de entrada utilizando columnas y controles deslizantes (*sliders*), con el propósito de presentar de manera ordenada los parámetros fisicoquímicos del vino.

El objetivo principal fue mejorar la distribución de los controles, facilitar la lectura de las características y ofrecer una interfaz más clara para el usuario.

Esta asistencia contribuyó a organizar visualmente las entradas relacionadas con la acidez, el contenido de alcohol, el pH, la densidad y las demás propiedades requeridas por los modelos.

El uso de inteligencia artificial en esta etapa se concentró en la estructura y presentación de la interfaz, sin sustituir la ejecución de los modelos neuronales utilizados para generar las predicciones.

---

## 6. Revisión humana y responsabilidad académica

Las herramientas de inteligencia artificial generativa se utilizaron como recursos complementarios de apoyo durante distintas etapas del proyecto.

Su participación estuvo relacionada principalmente con consultas conceptuales, organización de procesos, asistencia en programación, comprensión de errores y elaboración de explicaciones técnicas.

La responsabilidad académica y técnica sobre el proyecto corresponde a los integrantes del equipo, quienes deben comprender y revisar los componentes desarrollados y verificar los resultados presentados.

Las sugerencias proporcionadas por herramientas de inteligencia artificial no se consideran automáticamente correctas. Su utilización requiere revisión, adaptación al contexto y comprobación mediante el código y las pruebas correspondientes.

Los resultados estadísticos y las predicciones reportadas deben provenir de la ejecución de los modelos y de los procedimientos experimentales implementados.

## 7. Declaración final

El uso de inteligencia artificial en el Proyecto D tuvo como finalidad apoyar el aprendizaje, facilitar la investigación de soluciones y mejorar la organización del proceso de desarrollo.

Las herramientas utilizadas no sustituyeron los modelos neuronales entrenados, la evaluación experimental ni las responsabilidades individuales de los integrantes.

Esta declaración documenta los principales usos reportados por el equipo y refleja el compromiso de mantener transparencia sobre la utilización de herramientas de inteligencia artificial en el contexto académico.