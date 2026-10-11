"""Primera  parte EDA"""

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class WineEDA:
    """Realiza análisis exploratorio del dataset de vinos (en progreso(puede cambiar segun resultados))."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        figures_directory: Path | None = None,
    ) -> None:
        self.dataframe = dataframe.copy()
        self.figures_directory = (
            Path(figures_directory)
            if figures_directory is not None
            else None
        )

        sns.set_theme(style="whitegrid")

    def _save_figure(self, file_name: str) -> None:
        """Guarda la figura cuando se proporciona una carpeta."""

        if self.figures_directory is not None:
            self.figures_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            plt.savefig(
                self.figures_directory / file_name,
                dpi=300,
                bbox_inches="tight",
            )

    def general_summary(self) -> Dict[str, object]:
        """Genera un resumen general del dataset."""

        return {
            "shape": self.dataframe.shape,
            "columns": self.dataframe.columns.tolist(),
            "data_types": self.dataframe.dtypes,
            "missing_values": self.dataframe.isna().sum(),
            "duplicated_rows": int(
                self.dataframe.duplicated().sum()
            ),
            "descriptive_statistics":
                self.dataframe.describe().T,
        }

    def missing_values_table(self) -> pd.DataFrame:
        """Crea una tabla de valores faltantes."""

        missing_count = self.dataframe.isna().sum()
        missing_percentage = (
            missing_count / len(self.dataframe)
        ) * 100

        result = pd.DataFrame(
            {
                "missing_count": missing_count,
                "missing_percentage": missing_percentage,
            }
        )

        return result.sort_values(
            by="missing_count",
            ascending=False,
        )

    def duplicate_summary(self) -> Dict[str, float]:
        """Calcula cantidad y porcentaje de duplicados."""
        """Revisar si es esperado eliminar los duplicados ( Investigando )"""

        duplicate_count = int(
            self.dataframe.duplicated().sum()
        )

        duplicate_percentage = (
            duplicate_count / len(self.dataframe)
        ) * 100

        return {
            "duplicate_count": duplicate_count,
            "duplicate_percentage": duplicate_percentage,
        }

    def quality_distribution(self) -> None:
        """Grafico de la distribución de la puntuación de calidad."""
        """Revisar colores ( m[as adelante ) """

        plt.figure(figsize=(9, 5))

        sns.countplot(
            data=self.dataframe,
            x="quality",
            hue="wine_type",
            palette={
                "red": "#8B1E3F",
                "white": "#D4A72C",
            },
        )

        plt.title("Distribución de calidad por tipo de vino")
        plt.xlabel("Puntuación de calidad")
        plt.ylabel("Cantidad de vinos")
        plt.legend(title="Tipo de vino")

        self._save_figure("quality_distribution.png")
        plt.show()

    def high_quality_distribution(
        self,
        threshold: int = 7,
    ) -> pd.DataFrame:
        """Analiza el desbalance de la calidad alta."""

        analysis_data = self.dataframe.copy()

        analysis_data["high_quality"] = (
            analysis_data["quality"] >= threshold
        ).map(
            {
                True: "Calidad alta",
                False: "Calidad no alta",
            }
        )

        distribution = (
            analysis_data["high_quality"]
            .value_counts()
            .rename_axis("category")
            .reset_index(name="count")
        )

        distribution["percentage"] = (
            distribution["count"]
            / distribution["count"].sum()
        ) * 100

        plt.figure(figsize=(7, 5))

        sns.countplot(
            data=analysis_data,
            x="high_quality",
            hue="high_quality",
            palette=["#4C72B0", "#DD8452"],
            legend=False,
        )

        plt.title(
            f"Distribución de calidad alta, calidad ≥ {threshold}"
        )
        plt.xlabel("Categoría")
        plt.ylabel("Cantidad de vinos")

        self._save_figure("high_quality_distribution.png")
        plt.show()

        return distribution

    def numerical_distributions(self) -> None:
        """Graficos de histogramas de todas las variables numéricas."""

        numerical_columns = (
            self.dataframe
            .select_dtypes(include="number")
            .columns
        )

        self.dataframe[numerical_columns].hist(
            bins=30,
            figsize=(18, 14),
            color="#4C72B0",
            edgecolor="black",
        )

        plt.suptitle(
            "Distribución de variables numéricas",
            fontsize=16,
            y=1.02,
        )
        plt.tight_layout()

        self._save_figure("numerical_distributions.png")
        plt.show()

    def boxplots_by_wine_type(self) -> None:
        """Crea boxplots para identificar posibles atípicos."""

        numerical_columns = [
            column
            for column in self.dataframe.select_dtypes(
                include="number"
            ).columns
            if column != "quality"
        ]

        number_of_columns = 3
        number_of_rows = (
            len(numerical_columns) + number_of_columns - 1
        ) // number_of_columns

        figure, axes = plt.subplots(
            number_of_rows,
            number_of_columns,
            figsize=(18, number_of_rows * 4),
        )

        axes = axes.flatten()

        for index, column in enumerate(numerical_columns):
            sns.boxplot(
                data=self.dataframe,
                x="wine_type",
                y=column,
                hue="wine_type",
                palette={
                    "red": "#8B1E3F",
                    "white": "#D4A72C",
                },
                legend=False,
                ax=axes[index],
            )

            axes[index].set_title(column)
            axes[index].set_xlabel("Tipo de vino")
            axes[index].set_ylabel("Valor")

        for index in range(
            len(numerical_columns),
            len(axes),
        ):
            figure.delaxes(axes[index])

        plt.tight_layout()

        self._save_figure("boxplots_by_wine_type.png")
        plt.show()

    def correlation_matrix(self) -> pd.DataFrame:
        """Calcula y grafica la correlación ."""

        numerical_data = self.dataframe.select_dtypes(
            include="number"
        )

        correlation = numerical_data.corr()

        plt.figure(figsize=(14, 10))

        sns.heatmap(
            correlation,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
        )

        plt.title("Matriz de correlación de Pearson")
        plt.tight_layout()

        self._save_figure("correlation_matrix.png")
        plt.show()

        return correlation

    def quality_correlations(self) -> pd.Series:
        """Ordena las correlaciones respecto a quality."""

        numerical_data = self.dataframe.select_dtypes(
            include="number"
        )

        correlations = (
            numerical_data
            .corr()["quality"]
            .drop("quality")
            .sort_values(ascending=False)
        )

        plt.figure(figsize=(10, 6))

        sns.barplot(
            x=correlations.values,
            y=correlations.index,
            hue=correlations.index,
            palette="vlag",
            legend=False,
        )

        plt.axvline(0, color="black", linewidth=1)
        plt.title("Correlación de variables con la calidad")
        plt.xlabel("Correlación de Pearson")
        plt.ylabel("Variable")

        self._save_figure("quality_correlations.png")
        plt.show()

        return correlations

    def compare_wine_types(self) -> pd.DataFrame:
        """Compara los promedios entre vinos tintos y blancos."""

        numerical_columns = self.dataframe.select_dtypes(
            include="number"
        ).columns

        return (
            self.dataframe
            .groupby("wine_type")[numerical_columns]
            .mean()
            .T
        )

    def detect_outliers_iqr(self) -> pd.DataFrame:
        """
        Identifica posibles atípicos utilizando el rango
        intercuartílico, sin eliminarlos automáticamente. Formulas de Miner[ia de datos 1
        """

        numerical_columns = [
            column
            for column in self.dataframe.select_dtypes(
                include="number"
            ).columns
            if column != "quality"
        ]

        results = []

        for column in numerical_columns:
            q1 = self.dataframe[column].quantile(0.25)
            q3 = self.dataframe[column].quantile(0.75)
            iqr = q3 - q1

            lower_limit = q1 - 1.5 * iqr
            upper_limit = q3 + 1.5 * iqr

            outlier_mask = (
                (self.dataframe[column] < lower_limit)
                | (self.dataframe[column] > upper_limit)
            )

            outlier_count = int(outlier_mask.sum())

            results.append(
                {
                    "variable": column,
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "lower_limit": lower_limit,
                    "upper_limit": upper_limit,
                    "outlier_count": outlier_count,
                    "outlier_percentage": (
                        outlier_count
                        / len(self.dataframe)
                    ) * 100,
                }
            )

        return pd.DataFrame(results).sort_values(
            by="outlier_percentage",
            ascending=False,
        )