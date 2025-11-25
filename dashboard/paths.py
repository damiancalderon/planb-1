from pathlib import Path

# Root directories
REPO_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"

# Frequently used artifacts
DUCKDB_FILE = DASHBOARD_DIR / "cdmx_insights.db"
MODEL_RF_PATH = DASHBOARD_DIR / "violence_rf_optimizado_v3.joblib"
MODEL_XGB_PATH = DASHBOARD_DIR / "violence_xgb_optimizado_v3.joblib"
MODEL_KMEANS_PATH = DASHBOARD_DIR / "kmeans_zonas.joblib"
MODEL_FORECAST_PATH = DASHBOARD_DIR / "crime_forecaster.pkl"
CLUSTER_INFO_PATH = DASHBOARD_DIR / "cluster_info.csv"
ALCALDIAS_GEOJSON_PATH = DASHBOARD_DIR / "alcaldias.geojson"
ARTIFACTS_DIR = DASHBOARD_DIR / "artifacts"
