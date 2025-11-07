import duckdb
import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans # <-- MEJORA 1 (Clustering)
from sklearn.metrics import classification_report
from scipy.stats import randint # <-- MEJORA 3 (Tuning)
import warnings

warnings.filterwarnings('ignore')

DB_FILE = "cdmx_insights.db"

def load_data_for_classification():
    """
    Carga de datos MEJORADA.
    Ahora incluye lat, lon y fecha_hecho para feature engineering.
    """
    print("Cargando datos para clasificación (con features extendidos)...")
    con = duckdb.connect(DB_FILE, read_only=True)
    query = """
    SELECT 
        HOUR(hora_hecho) AS hora_hecho,
        mes_hecho, 
        CAST(fecha_hecho AS DATE) AS fecha_hecho, -- MEJORA 2 (Fechas)
        alcaldia_hecho, 
        categoria_delito, 
        latitud, -- MEJORA 1 (Ubicación)
        longitud, -- MEJORA 1 (Ubicación)
        violence_type 
    FROM 
        crimes
    WHERE
        violence_type IS NOT NULL AND
        alcaldia_hecho IS NOT NULL AND
        categoria_delito IS NOT NULL AND
        latitud IS NOT NULL AND  -- Nos aseguramos de tener datos de ubicación
        longitud IS NOT NULL
    LIMIT 200000 
    """
    df = con.execute(query).fetchdf()
    con.close()
    
    # Manejo de nulos por si acaso (aunque la query ya filtra)
    df = df.dropna(subset=['latitud', 'longitud', 'alcaldia_hecho', 'categoria_delito'])
    
    print(f"Datos cargados: {len(df)} filas.")
    return df

def load_data_for_timeseries():
    """
    Carga datos para el modelo de series de tiempo (Sin cambios).
    """
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

def train_classification_models():
    """
    Entrena y optimiza (tuning) RandomForest y XGBoost
    usando feature engineering avanzado.
    """
    df = load_data_for_classification()
    
    # --- 1. Definición de Target ---
    X = df.drop('violence_type', axis=1)
    y = df['violence_type'].map({'Non-Violent': 0, 'Violent': 1})
    target_names = ['Non-Violent', 'Violent']

    # --- 2. MEJORA 2: Feature Engineering (Fechas) ---
    print("Iniciando Feature Engineering (Fechas)...")
    X['fecha_hecho'] = pd.to_datetime(X['fecha_hecho'])
    X['dia_de_la_semana'] = X['fecha_hecho'].dt.dayofweek # 0=Lunes, 6=Domingo
    X['es_fin_de_semana'] = (X['dia_de_la_semana'] >= 5).astype(int)
    # Quincena (del día 14 al 16, y del 28 al 2 del mes siguiente)
    X['es_quincena'] = X['fecha_hecho'].dt.day.isin([14,15,16, 28,29,30,31,1,2]).astype(int)

    # --- 3. División de Datos (Train/Test Split) ---
    # ¡Importante! Dividimos ANTES de hacer clustering para evitar data leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 4. MEJORA 1: Feature Engineering (Ubicación/Clustering) ---
    print("Iniciando Feature Engineering (Clustering de Ubicación)...")
    # Usaremos 50 "zonas calientes" (puedes ajustar este número)
    kmeans = KMeans(n_clusters=50, random_state=42, n_init=10) 
    
    # Entrenamos KMeans SÓLO con los datos de entrenamiento
    coords_train = X_train[['latitud', 'longitud']]
    X_train['zona_cluster'] = kmeans.fit_predict(coords_train)
    
    # Transformamos los datos de prueba (no volvemos a entrenar)
    coords_test = X_test[['latitud', 'longitud']]
    X_test['zona_cluster'] = kmeans.predict(coords_test)

    # Guardamos el modelo de KMeans para usarlo en la app de predicción
    kmeans_filename = 'kmeans_zonas.joblib'
    joblib.dump(kmeans, kmeans_filename)
    print(f"Modelo de Clustering guardado en: {kmeans_filename}")

    # --- 5. Feature Engineering (Cíclicos) ---
    print("Iniciando Feature Engineering (Datos Cíclicos)...")
    # Hacemos esto para Train y Test por separado
    for df_split in [X_train, X_test]:
        df_split['hora_sin'] = np.sin(2 * np.pi * df_split['hora_hecho'] / 24)
        df_split['hora_cos'] = np.cos(2 * np.pi * df_split['hora_hecho'] / 24)
        df_split['mes_sin'] = np.sin(2 * np.pi * df_split['mes_hecho'] / 12)
        df_split['mes_cos'] = np.cos(2 * np.pi * df_split['mes_hecho'] / 12)

    # --- 6. Definición Final del Preprocesador ---
    
    # Lista de features que usará el modelo
    # Nota: Ya no usamos 'latitud' y 'longitud' directamente
    categorical_features = ['alcaldia_hecho', 'categoria_delito', 'zona_cluster'] 
    numeric_features = [
        'dia_de_la_semana', 
        'es_fin_de_semana', 
        'es_quincena',
        'hora_sin', 
        'hora_cos', 
        'mes_sin', 
        'mes_cos'
    ]
    
    # Columnas que ya no necesitamos
    cols_to_drop = ['fecha_hecho', 'latitud', 'longitud', 'hora_hecho', 'mes_hecho']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop' # Ignora las columnas en cols_to_drop
    )

    # --- 7. MEJORA 3: Hyperparameter Tuning ---

    # --- Modelo A: RandomForest ---
    print("\n===== BUSCANDO MEJORES PARÁMETROS: RANDOM FOREST =====")
    pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', RandomForestClassifier(random_state=42, 
                                                                       class_weight='balanced'))])
    
    # Parámetros para buscar (versión reducida para ir más rápido)
    param_dist_rf = {
        'classifier__n_estimators': randint(100, 300),
        'classifier__max_depth': [15, 20, 30, None],
        'classifier__min_samples_leaf': randint(1, 5)
    }
    
    random_search_rf = RandomizedSearchCV(pipeline_rf, 
                                          param_distributions=param_dist_rf, 
                                          n_iter=10, # Número de combinaciones a probar (sube esto si tienes tiempo)
                                          cv=3, # 3-fold cross-validation
                                          verbose=2, 
                                          random_state=42, 
                                          n_jobs=4,
                                          scoring='f1_weighted') # Optimizar por F1-score
    
    random_search_rf.fit(X_train, y_train)
    print(f"Mejores parámetros para RF: {random_search_rf.best_params_}")
    best_model_rf = random_search_rf.best_estimator_
    
    # Evaluación
    y_pred_rf = best_model_rf.predict(X_test)
    print("\n--- Reporte de Clasificación: RandomForest (Optimizado) ---")
    print(classification_report(y_test, y_pred_rf, target_names=target_names))
    joblib.dump(best_model_rf, 'violence_rf_optimizado.joblib')

    # --- Modelo B: XGBoost ---
    print("\n===== BUSCANDO MEJORES PARÁMETROS: XGBOOST =====")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    pipeline_xgb = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', XGBClassifier(random_state=42,
                                                               n_jobs=-1,
                                                               eval_metric='logloss',
                                                               use_label_encoder=False,
                                                               scale_pos_weight=scale_pos_weight))])
    
    # Parámetros para buscar
    param_dist_xgb = {
        'classifier__n_estimators': randint(100, 300),
        'classifier__max_depth': [5, 6, 8, 10],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__subsample': [0.7, 0.8, 1.0]
    }

    random_search_xgb = RandomizedSearchCV(pipeline_xgb, 
                                           param_distributions=param_dist_xgb, 
                                           n_iter=10, 
                                           cv=3, 
                                           verbose=2, 
                                           random_state=42, 
                                           n_jobs=-1,
                                           scoring='f1_weighted')
    
    random_search_xgb.fit(X_train, y_train)
    print(f"Mejores parámetros para XGB: {random_search_xgb.best_params_}")
    best_model_xgb = random_search_xgb.best_estimator_
    
    # Evaluación
    y_pred_xgb = best_model_xgb.predict(X_test)
    print("\n--- Reporte de Clasificación: XGBoost (Optimizado) ---")
    print(classification_report(y_test, y_pred_xgb, target_names=target_names))
    joblib.dump(best_model_xgb, 'violence_xgb_optimizado.joblib')

# --- Función de Series de Tiempo (Sin cambios) ---
def train_timeseries_model():
    y = load_data_for_timeseries()
    print("\n===== ENTRENANDO MODELO DE SERIES DE TIEMPO =====")
    print("Esto puede tardar varios minutos.")
    model = sm.tsa.statespace.SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    model_fit = model.fit(disp=False)
    model_fit.save('crime_forecaster.pkl')
    print("Entrenamiento SARIMA completado y guardado en 'crime_forecaster.pkl'.")

# --- Ejecución Principal ---
if __name__ == "__main__":
    print("===== INICIANDO ENTRENAMIENTO DE MODELOS =====")
    train_classification_models()
    train_timeseries_model()
    print("\n===== ENTRENAMIENTO FINALIZADO =====")