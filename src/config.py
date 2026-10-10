from pathlib import Path


class ProjectConfig:
    """Configuración central de rutas y parámetros del proyecto del Vino."""

    BASE_DIR = Path(__file__).resolve().parent.parent

    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"

    MODELS_DIR = BASE_DIR / "models"
    REPORTS_DIR = BASE_DIR / "reports"
    FIGURES_DIR = REPORTS_DIR / "figures"

    RED_WINE_FILE = RAW_DATA_DIR / "winequality-red.csv"
    WHITE_WINE_FILE = RAW_DATA_DIR / "winequality-white.csv"

    COMBINED_DATA_FILE = PROCESSED_DATA_DIR / "winequality-combined.csv"
    CLEAN_DATA_FILE = PROCESSED_DATA_DIR / "winequality-clean.csv"

    X_TRAIN_FILE = PROCESSED_DATA_DIR / "X_train.csv"
    X_TEST_FILE = PROCESSED_DATA_DIR / "X_test.csv"

    Y_TRAIN_REGRESSION_FILE = (
        PROCESSED_DATA_DIR / "y_train_regression.csv"
    )
    Y_TEST_REGRESSION_FILE = (
        PROCESSED_DATA_DIR / "y_test_regression.csv"
    )

    Y_TRAIN_CLASSIFICATION_FILE = (
        PROCESSED_DATA_DIR / "y_train_classification.csv"
    )
    Y_TEST_CLASSIFICATION_FILE = (
        PROCESSED_DATA_DIR / "y_test_classification.csv"
    )

    SCALER_FILE = MODELS_DIR / "standard_scaler.joblib"

    TEST_SIZE = 0.20
    RANDOM_STATE = 42
    HIGH_QUALITY_THRESHOLD = 7

    @classmethod
    def create_directories(cls) -> None:
        """Crea las carpetas utilizadas por el proyecto."""

        directories = [
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.MODELS_DIR,
            cls.REPORTS_DIR,
            cls.FIGURES_DIR,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)