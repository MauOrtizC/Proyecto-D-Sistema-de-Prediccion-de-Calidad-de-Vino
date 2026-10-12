"""Modelos Pydantic: validan los datos de entrada y definen las respuestas de la API."""
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field


class VinoEntrada(BaseModel):
    """Propiedades fisicoquimicas de un vino (11 variables) + su tipo."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
                "tipo": "tinto",
            }
        }
    )

    fixed_acidity: float = Field(..., ge=3.0, le=16.0, description="Acidez fija (g/dm3)")
    volatile_acidity: float = Field(..., ge=0.0, le=2.0, description="Acidez volatil (g/dm3)")
    citric_acid: float = Field(..., ge=0.0, le=2.0, description="Acido citrico (g/dm3)")
    residual_sugar: float = Field(..., ge=0.0, le=70.0, description="Azucar residual (g/dm3)")
    chlorides: float = Field(..., ge=0.0, le=0.7, description="Cloruros (g/dm3)")
    free_sulfur_dioxide: float = Field(..., ge=0.0, le=300.0, description="Dioxido de azufre libre (mg/dm3)")
    total_sulfur_dioxide: float = Field(..., ge=0.0, le=450.0, description="Dioxido de azufre total (mg/dm3)")
    density: float = Field(..., ge=0.98, le=1.05, description="Densidad (g/cm3)")
    ph: float = Field(..., ge=2.5, le=4.5, description="pH")
    sulphates: float = Field(..., ge=0.0, le=2.5, description="Sulfatos (g/dm3)")
    alcohol: float = Field(..., ge=5.0, le=16.0, description="Alcohol (% vol.)")
    tipo: Literal["tinto", "blanco"] = Field(..., description="Tipo de vino: 'tinto' o 'blanco'")


class LoteEntrada(BaseModel):
    """Lista de vinos para evaluar varios a la vez (lo usa el dashboard)."""

    vinos: List[VinoEntrada] = Field(..., min_length=1, max_length=1000)


class CalidadRespuesta(BaseModel):
    """Respuesta del Modelo 1 (regresion)."""

    calidad_predicha: float = Field(..., description="Puntuacion de calidad estimada (0-10)")
    calidad_redondeada: int = Field(..., description="Puntuacion redondeada al entero mas cercano")


class CalidadAltaRespuesta(BaseModel):
    """Respuesta del Modelo 2 (clasificacion binaria)."""

    es_calidad_alta: bool = Field(..., description="True si el vino se clasifica como calidad alta (>= 7)")
    probabilidad: float = Field(..., description="Probabilidad de calidad alta (0.0 a 1.0)")
    etiqueta: str = Field(..., description="'Calidad alta' o 'Calidad no alta'")


class ResultadoLote(BaseModel):
    """Resultado combinado de ambos modelos para un vino del lote."""

    calidad_predicha: float
    calidad_redondeada: int
    es_calidad_alta: bool
    probabilidad_alta: float


class LoteRespuesta(BaseModel):
    total: int
    resultados: List[ResultadoLote]