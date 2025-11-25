import duckdb
import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm
import gzip
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans 
from sklearn.metrics import classification_report
from scipy.stats import randint, uniform 
import warnings

warnings.filterwarnings('ignore')

DB_FILE = "cdmx_insights.db"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def snapshot_artifact(source_path: Path, archive_name: str) -> None:
    """
    Compress artefacts so they can be committed (<100 MB) and restored on deploy.
    """
    if not source_path.exists():
        return
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = ARTIFACTS_DIR / archive_name
    with open(source_path, "rb") as src, gzip.open(archive_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    print(f"Copia comprimida actualizada en '{archive_path.name}'.")

def load_data_for_classification():
    print("Cargando datos para clasificación (v3)...")
    con = duckdb.connect(DB_FILE, read_only=True)
    query = """
    SELECT 
        HOUR(hora_hecho) AS hora_hecho,
        mes_hecho, 
        CAST(fecha_hecho AS DATE) AS fecha_hecho, 
        alcaldia_hecho, 
        categoria_delito, 
        latitud, 
        longitud, 
        violence_type 
    FROM 
        crimes
    WHERE
        violence_type IS NOT NULL AND
        alcaldia_hecho IS NOT NULL AND
        categoria_delito IS NOT NULL AND
        latitud IS NOT NULL AND
        longitud IS NOT NULL
    LIMIT 200000 
    """
    df = con.execute(query).fetchdf()
    con.close()
    df = df.dropna(subset=['latitud', 'longitud', 'alcaldia_hecho', 'categoria_delito'])
    print(f"Datos cargados: {len(df)} filas.")
    return df

def load_data_for_timeseries():
    print("Cargando datos para series de tiempo...")
    con = duckdb.connect(DB_FILE, read_only=True)
    query = "SELECT CAST(fecha_hecho AS DATE) AS fecha, COUNT(*) AS total_delitos FROM crimes GROUP BY fecha ORDER BY fecha ASC"
    df = con.execute(query).fetchdf()
    con.close()
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.set_index('fecha')
    df_resampled = df.asfreq('D').fillna(0)
    print(f"Datos de series de tiempo listos: {len(df_resampled)} días.")
    return df_resampled['total_delitos']

def map_to_time_slot(hour):
    if 0 <= hour <= 5: return 'Madrugada'
    elif 6 <= hour <= 11: return 'Mañana'
    elif 12 <= hour <= 18: return 'Tarde'
    return 'Noche'

def train_classification_models():
    df = load_data_for_classification()
    
    X = df.drop('violence_type', axis=1)
    y = df['violence_type'].map({'Non-Violent': 0, 'Violent': 1})
    target_names = ['Non-Violent', 'Violent']

    print("Iniciando Feature Engineering (Fechas)...")
    X['fecha_hecho'] = pd.to_datetime(X['fecha_hecho'])
    X['dia_de_la_semana'] = X['fecha_hecho'].dt.dayofweek
    X['es_fin_de_semana'] = (X['dia_de_la_semana'] >= 5).astype(int)
    X['es_quincena'] = X['fecha_hecho'].dt.day.isin([14,15,16, 28,29,30,31,1,2]).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Iniciando Feature Engineering (Clustering de Ubicación)...")
    kmeans = KMeans(n_clusters=50, random_state=42, n_init=10) 
    coords_train = X_train[['latitud', 'longitud']]
    X_train['zona_cluster'] = kmeans.fit_predict(coords_train)
    coords_test = X_test[['latitud', 'longitud']]
    X_test['zona_cluster'] = kmeans.predict(coords_test)
    joblib.dump(kmeans, 'kmeans_zonas.joblib')
    print("Modelo de Clustering guardado.")

    print("Iniciando Feature Engineering (v3 - Interacciones)...")
    
    for df_split in [X_train, X_test]:
        df_split['franja_horaria'] = df_split['hora_hecho'].apply(map_to_time_slot)
        df_split['zona_hora'] = df_split['zona_cluster'].astype(str) + "_" + df_split['franja_horaria']
        df_split['mes_sin'] = np.sin(2 * np.pi * df_split['mes_hecho'] / 12)
        df_split['mes_cos'] = np.cos(2 * np.pi * df_split['mes_hecho'] / 12)

    
    # --- Definición Final del Preprocesador (v3) ---
    categorical_features = ['alcaldia_hecho', 'categoria_delito', 'zona_hora'] 
    numeric_features = [
        'dia_de_la_semana', 
        'es_fin_de_semana', 
        'es_quincena',
        'mes_sin', 
        'mes_cos'
    ]
    
    cols_to_drop = ['fecha_hecho', 'latitud', 'longitud', 'hora_hecho', 'mes_hecho', 'zona_cluster', 'franja_horaria']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop' 
    )

    # --- Hyperparameter Tuning ---

    # --- Modelo A: RandomForest ---
    print("\n===== BUSCANDO MEJORES PARÁMETROS: RANDOM FOREST (v3) =====")
    pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', RandomForestClassifier(random_state=42, 
                                                                       class_weight='balanced'))])
    
    param_dist_rf = {
        'classifier__n_estimators': randint(100, 300),
        'classifier__max_depth': [15, 20, 30, None],
        'classifier__min_samples_leaf': randint(1, 5)
    }
    
    random_search_rf = RandomizedSearchCV(pipeline_rf, 
                                          param_distributions=param_dist_rf, 
                                          n_iter=10, # Número de combinaciones a probar
                                          cv=3, 
                                          verbose=2, 
                                          random_state=42, 
                                          n_jobs=4, # Evitar crasheos
                                          scoring='f1_weighted')
    
    random_search_rf.fit(X_train, y_train)
    print(f"Mejores parámetros para RF (v3): {random_search_rf.best_params_}")
    best_model_rf = random_search_rf.best_estimator_
    
    y_pred_rf = best_model_rf.predict(X_test)
    print("\n--- Reporte de Clasificación: RandomForest (Optimizado v3) ---")
    print(classification_report(y_test, y_pred_rf, target_names=target_names))
    joblib.dump(best_model_rf, 'violence_rf_optimizado_v3.joblib') # Nuevo nombre v3
    print("Modelo v3 (RF) guardado en 'violence_rf_optimizado_v3.joblib'")

    # --- Modelo B: XGBoost ---
    print("\n===== BUSCANDO MEJORES PARÁMETROS: XGBOOST (v3) =====")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    pipeline_xgb = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', XGBClassifier(random_state=42,
                                                               n_jobs=-1,
                                                               eval_metric='logloss',
                                                               use_label_encoder=False,
                                                               objective='binary:logistic',
                                                               max_delta_step=1,
                                                               scale_pos_weight=scale_pos_weight))])
    
    param_dist_xgb = {
        'classifier__n_estimators': randint(120, 280),
        'classifier__max_depth': [3, 4, 5, 6],
        'classifier__learning_rate': [0.01, 0.03, 0.05, 0.08],
        'classifier__subsample': [0.6, 0.75, 0.9],
        'classifier__colsample_bytree': [0.6, 0.75, 0.9],
        'classifier__min_child_weight': randint(3, 10),
        'classifier__gamma': uniform(0.0, 2.0),
        'classifier__reg_lambda': uniform(0.5, 1.5),
        'classifier__reg_alpha': uniform(0.0, 1.0)
    }

    random_search_xgb = RandomizedSearchCV(pipeline_xgb, 
                                           param_distributions=param_dist_xgb, 
                                           n_iter=10, 
                                           cv=3, 
                                           verbose=2, 
                                           random_state=42, 
                                           n_jobs=4, # Evitar crasheos
                                           scoring='f1_weighted')
    
    random_search_xgb.fit(X_train, y_train)
    print(f"Mejores parámetros para XGB (v3): {random_search_xgb.best_params_}")
    best_model_xgb = random_search_xgb.best_estimator_
    
    y_pred_xgb = best_model_xgb.predict(X_test)
    print("\n--- Reporte de Clasificación: XGBoost (Optimizado v3) ---")
    print(classification_report(y_test, y_pred_xgb, target_names=target_names))
    joblib.dump(best_model_xgb, 'violence_xgb_optimizado_v3.joblib') # Nuevo nombre v3 seguir incrementando cada reentrenamiento
    print("Modelo v3 (XGB) guardado en 'violence_xgb_optimizado_v3.joblib'")

# --- Función de Series de Tiempo ---
def train_timeseries_model():
    y = load_data_for_timeseries()
    print("\n===== ENTRENANDO MODELO DE SERIES DE TIEMPO =====")
    print("Esto puede tardar varios minutos.")
    model = sm.tsa.statespace.SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    model_fit = model.fit(disp=False)
    model_fit.save('crime_forecaster.pkl')
    print("Entrenamiento SARIMA completado.")
    snapshot_artifact(Path('crime_forecaster.pkl'), 'crime_forecaster.pkl.gz')

if __name__ == "__main__":
    print("===== INICIANDO ENTRENAMIENTO DE MODELOS (v3 con Interacciones) =====")
    train_classification_models()
    train_timeseries_model()
    print("\n===== ENTRENAMIENTO FINALIZADO =====")
