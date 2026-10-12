"""Segunda parte - Clase para cargar y unificar los datasets """

from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import ProjectConfig


class WineDataLoader:
    """Carga y unifica los datasets de vino tinto y blanco."""

    EXPECTED_COLUMNS = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
        "quality",
    ]

    def __init__(
        self,
        red_wine_path: Path,
        white_wine_path: Path,
        separator: str = ";",
    ) -> None:
        self.red_wine_path = Path(red_wine_path)
        self.white_wine_path = Path(white_wine_path)
        self.separator = separator

        self.red_wine: pd.DataFrame | None = None
        self.white_wine: pd.DataFrame | None = None
        self.combined_data: pd.DataFrame | None = None

    def _validate_file(self, file_path: Path) -> None:
        """Acá se valida que el archivo exista, y si no, entonces ... """

        if not file_path.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo: {file_path}"
            )

    def _validate_columns(self, dataframe: pd.DataFrame) -> None:
        """En esta parte se valida que el dataset tenga las columnas esperadas."""

        missing_columns = set(self.EXPECTED_COLUMNS) - set(
            dataframe.columns
        )
        """sino va a ... """

        if missing_columns:
            raise ValueError(
                "Faltan columnas requeridas: "
                f"{sorted(missing_columns)}"
            )

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Carga los archivos de vino tinto y blanco."""

        self._validate_file(self.red_wine_path)
        self._validate_file(self.white_wine_path)

        self.red_wine = pd.read_csv(
            self.red_wine_path,
            sep=self.separator,
        )

        self.white_wine = pd.read_csv(
            self.white_wine_path,
            sep=self.separator,
        )

        self._validate_columns(self.red_wine)
        self._validate_columns(self.white_wine)

        return self.red_wine, self.white_wine

    def combine_data(self) -> pd.DataFrame:
        """
        Agrega la variable wine_type y unifica ambos datasets.
        """

        if self.red_wine is None or self.white_wine is None:
            self.load_data()

        red_data = self.red_wine.copy()
        white_data = self.white_wine.copy()

        red_data["wine_type"] = "red"
        white_data["wine_type"] = "white"

        self.combined_data = pd.concat(
            [red_data, white_data],
            ignore_index=True,
        )

        return self.combined_data

    def get_dataset_summary(self) -> Dict[str, int]:
        """Devuelve la cantidad de registros por dataset."""

        if self.combined_data is None:
            self.combine_data()

        return {
            "red_wines": len(self.red_wine),
            "white_wines": len(self.white_wine),
            "total_wines": len(self.combined_data),
            "total_columns": len(self.combined_data.columns),
        }

    def save_combined_data(self, output_path: Path) -> None:
        """Guarda el dataset unificado en la carpeta processed."""

        if self.combined_data is None:
            self.combine_data()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.combined_data.to_csv(
            output_path,
            index=False,
        )


class WinePreprocessor:
    """Limpia, transforma, divide y normaliza los datos ( aún en progreso )"""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        high_quality_threshold: int = 7,
        test_size: float = 0.20,
        random_state: int = 42,
    ) -> None:
        self.dataframe = dataframe.copy()
        self.high_quality_threshold = high_quality_threshold
        self.test_size = test_size
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.feature_columns: list[str] = []

    def clean_data(self, remove_duplicates: bool = True) -> pd.DataFrame:
        """
        Limpia nombres de columnas, elimina valores faltantes
        y opcionalmente elimina duplicados ( en progreso , revisar que sirva bien en futuro commits, ojo remover este comentario .
        """

        self.dataframe.columns = [
            column.strip().lower().replace(" ", "_")
            for column in self.dataframe.columns
        ]

        if remove_duplicates:
            self.dataframe = self.dataframe.drop_duplicates().copy()

        self.dataframe = self.dataframe.dropna().copy()
        self.dataframe = self.dataframe.reset_index(drop=True)

        return self.dataframe

    def create_target_variables(self) -> pd.DataFrame:
        """Crea la variable binaria high_quality."""

        if "quality" not in self.dataframe.columns:
            raise ValueError(
                "No se encontró la columna objetivo 'quality'."
            )

        self.dataframe["high_quality"] = (
            self.dataframe["quality"]
            >= self.high_quality_threshold
        ).astype(int)

        return self.dataframe

    def encode_wine_type(self) -> pd.DataFrame:
        """
        Codifica wine_type usando una variable dummy - como vimos en clase que se vuelve binaria para identificar .

        wine_type_white:
            0 = vino tinto
            1 = vino blanco
        """

        if "wine_type" not in self.dataframe.columns:
            raise ValueError(
                "No se encontró la columna 'wine_type'."
            )

        self.dataframe = pd.get_dummies(
            self.dataframe,
            columns=["wine_type"],
            drop_first=True,
            dtype=int,
        )

        return self.dataframe

    def prepare_data(self) -> pd.DataFrame:
        """Ejecuta limpieza, creación de objetivos y codificación."""

        self.clean_data()
        self.create_target_variables()
        self.encode_wine_type()

        return self.dataframe

    def split_data(
        self,
    ) -> Dict[str, pd.DataFrame | pd.Series]:
        """
        Divide los datos para regresión y clasificación.

        Se emplea estratificación para conservar la proporción
        de vinos de calidad alta en entrenamiento y prueba.
        """

        required_columns = {"quality", "high_quality"}

        if not required_columns.issubset(self.dataframe.columns):
            self.prepare_data()

        X = self.dataframe.drop(
            columns=["quality", "high_quality"]
        )

        y_regression = self.dataframe["quality"]
        y_classification = self.dataframe["high_quality"]

        (
            X_train,
            X_test,
            y_train_classification,
            y_test_classification,
        ) = train_test_split(
            X,
            y_classification,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_classification,
        )

        y_train_regression = y_regression.loc[X_train.index]
        y_test_regression = y_regression.loc[X_test.index]

        self.feature_columns = X.columns.tolist()

        """Usando como ejemplo ejercicio en clase / revisar"""

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train_regression": y_train_regression,
            "y_test_regression": y_test_regression,
            "y_train_classification": y_train_classification,
            "y_test_classification": y_test_classification,
        }

    def normalize_data(
        self,
        split_data: Dict[str, pd.DataFrame | pd.Series],
    ) -> Dict[str, pd.DataFrame | pd.Series]:
        """
        Ajusta StandardScaler solamente con entrenamiento y
        transforma entrenamiento y prueba.
        """

        X_train = split_data["X_train"]
        X_test = split_data["X_test"]

        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )

        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )

        normalized_data = split_data.copy()
        normalized_data["X_train"] = X_train_scaled
        normalized_data["X_test"] = X_test_scaled

        return normalized_data

    def save_processed_data(
        self,
        split_data: Dict[str, pd.DataFrame | pd.Series],
        output_directory: Path,
        scaler_path: Path,
    ) -> None:
        """Guarda datasets procesados y el escalador.Revisar esta parte ya que segun instrucciones se debe guardar en modelos para su uso posterior"""

        output_directory = Path(output_directory)
        scaler_path = Path(scaler_path)

        output_directory.mkdir(parents=True, exist_ok=True)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)

        file_names = {
            "X_train": "X_train.csv",
            "X_test": "X_test.csv",
            "y_train_regression": "y_train_regression.csv",
            "y_test_regression": "y_test_regression.csv",
            "y_train_classification":
                "y_train_classification.csv",
            "y_test_classification":
                "y_test_classification.csv",
        }

        for dataset_name, file_name in file_names.items():
            split_data[dataset_name].to_csv(
                output_directory / file_name,
                index=False,
            )

        joblib.dump(self.scaler, scaler_path)

        """Se crea el joblib del salvado, pero revisar comparando con ejercicio en clase """