# Documentación de la API: Predicción de Calidad de Vino

Proyecto D, BD-151 Inteligencia Artificial Aplicada, Colegio Universitario de Cartago.

## 1. Qué hace la API

Esta API recibe las propiedades fisicoquímicas de un vino (tinto o blanco) y devuelve dos resultados, cada uno calculado por una red neuronal distinta:

- **Modelo 1 (regresión):** estima la puntuación de calidad del vino, en una escala de 0 a 10.
- **Modelo 2 (clasificación binaria):** indica si el vino es de calidad alta, es decir, si su calidad es 7 o más.

Está hecha con FastAPI y carga los modelos entrenados (Keras/TensorFlow) una sola vez al arrancar, para que cada petición responda rápido. El dashboard de Streamlit consume estos mismos endpoints.

## 2. Cómo ejecutarla

Requisitos: Python 3.10 a 3.13 (TensorFlow todavía no se instala en Python 3.14) y las librerías del proyecto instaladas.

```bash
pip install tensorflow pandas numpy scikit-learn joblib fastapi "uvicorn[standard]" requests
```

El servidor se levanta siempre desde la **raíz del proyecto**, la carpeta que contiene `api/`, `models/` y `app/`:

```bash
uvicorn api.main:app --reload
```

Si todo va bien, la terminal muestra `Modelos cargados correctamente`. La documentación interactiva queda en:

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

> Si el comando se ejecuta desde dentro de la carpeta `api/`, Python no encuentra el paquete `api` y aparece `ModuleNotFoundError: No module named 'api'`.

## 3. Archivos que usa

| Archivo | Para qué sirve |
|---|---|
| `models/model1.keras` | Red neuronal de regresión (calidad 0 a 10) |
| `models/model2.keras` | Red neuronal de clasificación (calidad alta sí/no) |
| `models/standard_scaler.joblib` | `StandardScaler` ajustado con los datos de entrenamiento |

Código de la API:

| Archivo | Contenido |
|---|---|
| `api/main.py` | La aplicación FastAPI y sus endpoints |
| `api/schemas.py` | Modelos Pydantic: validación de entradas y formato de respuestas |
| `api/predict.py` | Carga de modelos y lógica de predicción |

## 4. Datos de entrada

Los tres endpoints de predicción reciben el mismo objeto, con las 11 variables fisicoquímicas más el tipo de vino. Si algún valor está fuera de rango o falta un campo, la API responde con error 422 y no llega a ejecutar el modelo.

| Campo | Descripción | Mínimo | Máximo |
|---|---|---|---|
| `fixed_acidity` | Acidez fija (g/dm³) | 3 | 16 |
| `volatile_acidity` | Acidez volátil (g/dm³) | 0 | 2 |
| `citric_acid` | Ácido cítrico (g/dm³) | 0 | 2 |
| `residual_sugar` | Azúcar residual (g/dm³) | 0 | 70 |
| `chlorides` | Cloruros (g/dm³) | 0 | 0.7 |
| `free_sulfur_dioxide` | Dióxido de azufre libre (mg/dm³) | 0 | 300 |
| `total_sulfur_dioxide` | Dióxido de azufre total (mg/dm³) | 0 | 450 |
| `density` | Densidad (g/cm³) | 0.98 | 1.05 |
| `ph` | pH | 2.5 | 4.5 |
| `sulphates` | Sulfatos (g/dm³) | 0 | 2.5 |
| `alcohol` | Alcohol (% vol.) | 5 | 16 |
| `tipo` | Tipo de vino | `"tinto"` o `"blanco"` | |

Ejemplo de entrada:

```json
{
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.70,
  "citric_acid": 0.0,
  "residual_sugar": 1.9,
  "chlorides": 0.076,
  "free_sulfur_dioxide": 11.0,
  "total_sulfur_dioxide": 34.0,
  "density": 0.9978,
  "ph": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4,
  "tipo": "tinto"
}
```

## 5. Endpoints

### GET `/`

Ruta raíz. Sirve para comprobar rápidamente que el servidor está activo.

Respuesta (200):

```json
{
  "mensaje": "API de Calidad de Vino activa",
  "documentacion": "/docs"
}
```

### GET `/health`

Indica si los modelos y el escalador se cargaron correctamente. Es lo primero que conviene consultar si una predicción falla.

Respuesta (200) con los modelos cargados:

```json
{
  "estado": "ok",
  "umbral_calidad_alta": 0.5,
  "detalle": null
}
```

Si los modelos no se pudieron cargar, `estado` vale `"modelos_no_cargados"` y `detalle` contiene el motivo (por ejemplo, un archivo faltante en `models/`).

### POST `/predict/quality`

Usa el Modelo 1. Recibe un vino (ver sección 4) y devuelve la puntuación de calidad estimada.

| Campo de respuesta | Tipo | Significado |
|---|---|---|
| `calidad_predicha` | decimal | Puntuación estimada entre 0 y 10, con 2 decimales |
| `calidad_redondeada` | entero | La misma puntuación redondeada al entero más cercano |

Ejemplo de respuesta (los valores son ilustrativos):

```json
{
  "calidad_predicha": 5.21,
  "calidad_redondeada": 5
}
```

### POST `/predict/high_quality`

Usa el Modelo 2. Recibe el mismo objeto de entrada y responde si el vino se considera de calidad alta (7 o más).

| Campo de respuesta | Tipo | Significado |
|---|---|---|
| `es_calidad_alta` | booleano | `true` si la probabilidad es 0.5 o mayor |
| `probabilidad` | decimal | Probabilidad de calidad alta, entre 0 y 1 |
| `etiqueta` | texto | `"Calidad alta"` o `"Calidad no alta"` |

Ejemplo de respuesta (valores ilustrativos):

```json
{
  "es_calidad_alta": false,
  "probabilidad": 0.0312,
  "etiqueta": "Calidad no alta"
}
```

### POST `/predict/batch`

Evalúa varios vinos en una sola petición con los dos modelos a la vez. Lo usa la página de análisis de lotes del dashboard. Acepta entre 1 y 1000 vinos.

Entrada:

```json
{
  "vinos": [
    { "...": "vino 1, mismos campos de la sección 4" },
    { "...": "vino 2" }
  ]
}
```

Respuesta: el total de vinos procesados y un resultado por cada uno, en el mismo orden en que se enviaron.

```json
{
  "total": 2,
  "resultados": [
    {
      "calidad_predicha": 5.21,
      "calidad_redondeada": 5,
      "es_calidad_alta": false,
      "probabilidad_alta": 0.0312
    },
    {
      "calidad_predicha": 6.48,
      "calidad_redondeada": 6,
      "es_calidad_alta": false,
      "probabilidad_alta": 0.2874
    }
  ]
}
```

## 6. Códigos de error

| Código | Cuándo ocurre |
|---|---|
| 422 | Un dato está fuera de rango, falta un campo o `tipo` no es `"tinto"` ni `"blanco"`. La respuesta indica qué campo falló. |
| 503 | Los modelos no están cargados (por ejemplo, falta un archivo en `models/`). |
| 500 | Falló algo inesperado durante la predicción. El mensaje viene en el campo `detail`. |

Ejemplo de error 422 al enviar un pH de 20:

```json
{
  "detail": [
    {
      "type": "less_than_equal",
      "loc": ["body", "ph"],
      "msg": "Input should be less than or equal to 4.5"
    }
  ]
}
```

## 7. Qué pasa por dentro en cada predicción

1. Pydantic valida los datos de entrada contra los rangos de la sección 4.
2. Se arma una fila de 12 columnas en el mismo orden que se usó al entrenar. Las 11 primeras son las variables fisicoquímicas y la última es `wine_type_white`, que vale 1 si el vino es blanco y 0 si es tinto.
3. El `StandardScaler` guardado en `standard_scaler.joblib` estandariza esa fila. El orden y los nombres de las columnas se leen del propio escalador, de modo que la API no puede desordenarlas.
4. La fila escalada entra a la red neuronal correspondiente.
5. Se devuelve el resultado. En la regresión, la salida se limita al rango de 0 a 10. En la clasificación, el vino se marca como calidad alta si la probabilidad es 0.5 o más.

## 8. Ejemplo de uso desde Python

```python
import requests

vino = {
    "fixed_acidity": 7.4, "volatile_acidity": 0.70, "citric_acid": 0.0,
    "residual_sugar": 1.9, "chlorides": 0.076, "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0, "density": 0.9978, "ph": 3.51,
    "sulphates": 0.56, "alcohol": 9.4, "tipo": "tinto",
}

r = requests.post("http://127.0.0.1:8000/predict/quality", json=vino)
print(r.json())
```

El archivo `probar_api.py` en la raíz del proyecto prueba todos los endpoints de una vez, incluyendo un caso con datos inválidos.

## 9. Limitaciones

- Los rangos de validación se definieron a partir de los valores típicos de los datasets de vino tinto y blanco. Un vino con valores fuera de esos rangos es rechazado aunque sea real.
- El umbral de calidad alta es fijo (0.5). Si se quiere cambiar, está en la constante `UMBRAL_CALIDAD_ALTA` de `api/predict.py`.
- La API depende de que `models/` contenga los tres archivos de la sección 3. Si se reentrenan los modelos, hay que reemplazar también el escalador.