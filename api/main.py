"""API REST (FastAPI) del Sistema de Prediccion de Calidad de Vino.

Ejecutar desde la RAIZ del proyecto (la carpeta que contiene api/, models/, app/ ...):
    uvicorn api.main:app --reload
Documentacion automatica: http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.predict import UMBRAL_CALIDAD_ALTA, predictor
from api.schemas import (
    CalidadAltaRespuesta,
    CalidadRespuesta,
    LoteEntrada,
    LoteRespuesta,
    ResultadoLote,
    VinoEntrada,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Los modelos se cargan una sola vez al iniciar el servidor
    try:
        predictor.cargar()
        print("Modelos cargados correctamente")
    except Exception as e:  # el servidor arranca igual; /health informa el problema
        print(f"No se pudieron cargar los modelos: {e}")
        app.state.error_carga = str(e)
    yield


app = FastAPI(
    title="API Calidad de Vino",
    description=(
        "Predice la calidad de un vino (tinto o blanco) a partir de sus propiedades "
        "fisicoquimicas. Modelo 1: regresion de la puntuacion (0-10). "
        "Modelo 2: clasificacion binaria de calidad alta (>= 7)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _exigir_modelos() -> None:
    if not predictor.listo:
        detalle = getattr(app.state, "error_carga", "Los modelos no estan cargados")
        raise HTTPException(status_code=503, detail=f"Servicio no disponible: {detalle}")


@app.get("/", tags=["General"])
def raiz():
    return {"mensaje": "API de Calidad de Vino activa", "documentacion": "/docs"}


@app.get("/health", tags=["General"])
def health():
    """Indica si los modelos estan cargados y listos."""
    return {
        "estado": "ok" if predictor.listo else "modelos_no_cargados",
        "umbral_calidad_alta": UMBRAL_CALIDAD_ALTA,
        "detalle": getattr(app.state, "error_carga", None),
    }


@app.post("/predict/quality", response_model=CalidadRespuesta, tags=["Predicciones"])
def predecir_calidad(vino: VinoEntrada):
    """Modelo 1 (regresion): puntuacion de calidad estimada (0-10)."""
    _exigir_modelos()
    try:
        calidad = predictor.predecir_calidad([vino.model_dump()])[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {e}")
    return CalidadRespuesta(
        calidad_predicha=round(calidad, 2),
        calidad_redondeada=int(round(calidad)),
    )


@app.post("/predict/high_quality", response_model=CalidadAltaRespuesta, tags=["Predicciones"])
def predecir_calidad_alta(vino: VinoEntrada):
    """Modelo 2 (clasificacion binaria): si el vino es de calidad alta (>= 7)."""
    _exigir_modelos()
    try:
        prob = predictor.predecir_calidad_alta([vino.model_dump()])[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {e}")
    es_alta = prob >= UMBRAL_CALIDAD_ALTA
    return CalidadAltaRespuesta(
        es_calidad_alta=es_alta,
        probabilidad=round(prob, 4),
        etiqueta="Calidad alta" if es_alta else "Calidad no alta",
    )


@app.post("/predict/batch", response_model=LoteRespuesta, tags=["Predicciones"])
def predecir_lote(lote: LoteEntrada):
    """Evalua varios vinos a la vez con ambos modelos (lo usa el dashboard para lotes)."""
    _exigir_modelos()
    vinos = [v.model_dump() for v in lote.vinos]
    try:
        calidades = predictor.predecir_calidad(vinos)
        probs = predictor.predecir_calidad_alta(vinos)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir el lote: {e}")
    resultados = [
        ResultadoLote(
            calidad_predicha=round(c, 2),
            calidad_redondeada=int(round(c)),
            es_calidad_alta=p >= UMBRAL_CALIDAD_ALTA,
            probabilidad_alta=round(p, 4),
        )
        for c, p in zip(calidades, probs)
    ]
    return LoteRespuesta(total=len(resultados), resultados=resultados)