"""Logica de inferencia: carga los artefactos entrenados y hace las predicciones.

Artefactos en la carpeta models/ (generados por el equipo de modelado):
    model1.keras           -> regresion (calidad 0-10), entrada de 12 variables
    model2.keras           -> clasificacion binaria (calidad alta >= 7), salida sigmoid
    standard_scaler.joblib -> StandardScaler entrenado con 12 columnas
                              (11 fisicoquimicas + wine_type_white)

El orden y los nombres de las columnas se leen del propio scaler (feature_names_in_),
asi la API siempre construye la entrada igual que en el entrenamiento.
"""
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

UMBRAL_CALIDAD_ALTA = 0.5  # probabilidad minima para clasificar como calidad alta

# Nombres de las 11 variables numericas (los mismos que usa el scaler)
CAMPOS_NUMERICOS = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "ph",
    "sulphates",
    "alcohol",
]


def _normalizar(nombre: str) -> str:
    """'fixed acidity' / 'Fixed_Acidity' / 'pH' -> 'fixed_acidity' / 'fixed_acidity' / 'ph'."""
    return str(nombre).strip().lower().replace(" ", "_")


class Predictor:
    """Carga los modelos una sola vez y expone metodos de prediccion."""

    def __init__(self) -> None:
        self.modelo_calidad = None
        self.modelo_alta = None
        self.scaler = None
        self.columnas: List[str] = []

    # ------------------------------------------------------------------ carga
    def cargar(self) -> None:
        archivos = {
            "model1.keras": MODELS_DIR / "model1.keras",
            "model2.keras": MODELS_DIR / "model2.keras",
            "standard_scaler.joblib": MODELS_DIR / "standard_scaler.joblib",
        }
        faltantes = [nombre for nombre, ruta in archivos.items() if not ruta.exists()]
        if faltantes:
            raise FileNotFoundError(
                f"Faltan archivos en {MODELS_DIR}: {', '.join(faltantes)}"
            )

        self.modelo_calidad = load_model(archivos["model1.keras"])
        self.modelo_alta = load_model(archivos["model2.keras"])
        self.scaler = joblib.load(archivos["standard_scaler.joblib"])

        if not hasattr(self.scaler, "feature_names_in_"):
            raise ValueError(
                "El scaler no guarda los nombres de las columnas (feature_names_in_). "
                "Debe haberse entrenado con un DataFrame de pandas."
            )
        self.columnas = list(self.scaler.feature_names_in_)
        self._verificar_columnas()

    def _verificar_columnas(self) -> None:
        """Falla al arrancar (no al predecir) si alguna columna no se puede construir."""
        no_reconocidas = []
        for col in self.columnas:
            c = _normalizar(col)
            if c in CAMPOS_NUMERICOS:
                continue
            if self._es_dummy_tinto(c) or self._es_dummy_blanco(c):
                continue
            no_reconocidas.append(col)
        if no_reconocidas:
            raise ValueError(
                "Columnas del scaler que la API no sabe construir: "
                f"{no_reconocidas}. Revisa CAMPOS_NUMERICOS / dummies en api/predict.py"
            )

    @property
    def listo(self) -> bool:
        return all(
            x is not None
            for x in (self.modelo_calidad, self.modelo_alta, self.scaler)
        ) and bool(self.columnas)

    # ----------------------------------------------------------- preprocesado
    @staticmethod
    def _es_dummy_tinto(c: str) -> bool:
        return "tinto" in c or "red" in c

    @staticmethod
    def _es_dummy_blanco(c: str) -> bool:
        return "blanco" in c or "white" in c

    def _construir_matriz(self, vinos: List[dict]) -> np.ndarray:
        """Replica el preprocesamiento del entrenamiento: dummy del tipo + orden de columnas + escalado."""
        filas = []
        for vino in vinos:
            fila = []
            for col in self.columnas:
                c = _normalizar(col)
                if c in CAMPOS_NUMERICOS:
                    fila.append(float(vino[c]))
                elif self._es_dummy_tinto(c):
                    fila.append(1.0 if vino["tipo"] == "tinto" else 0.0)
                else:  # dummy de blanco (wine_type_white): 1 si es blanco, 0 si es tinto
                    fila.append(1.0 if vino["tipo"] == "blanco" else 0.0)
            filas.append(fila)

        X = pd.DataFrame(filas, columns=self.columnas)
        return self.scaler.transform(X)

    # ------------------------------------------------------------ predicciones
    def predecir_calidad(self, vinos: List[dict]) -> List[float]:
        X = self._construir_matriz(vinos)
        pred = self.modelo_calidad.predict(X, verbose=0).reshape(-1)
        return [float(np.clip(p, 0.0, 10.0)) for p in pred]

    def predecir_calidad_alta(self, vinos: List[dict]) -> List[float]:
        X = self._construir_matriz(vinos)
        prob = self.modelo_alta.predict(X, verbose=0).reshape(-1)
        return [float(p) for p in prob]


predictor = Predictor()