import json
from datetime import datetime
from functools import lru_cache
from typing import List, Optional

import folium
import joblib
import mapa
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from unidecode import unidecode

from paths import (
    ALCALDIAS_GEOJSON_PATH,
    CLUSTER_INFO_PATH,
    DASHBOARD_DIR,
    MODEL_KMEANS_PATH,
    MODEL_XGB_PATH,
)
from .data import get_all_alcaldias, get_all_crime_categories, run_query
from .location_utils import enrich_cluster_locations

# =========================
# RUTAS GLOBALES (ajusta si hace falta)
# =========================
BASE_PATH = DASHBOARD_DIR

ALL_YEARS_OPTION = "Todo el histórico"
DEFAULT_MAP_YEAR = 2024
HOTSPOT_CACHE_VERSION = "1.0"


# --- 1. Carga de Modelos y Datos (V3) ---
@st.cache_resource
def load_models_and_data():
    """
    Carga todos los modelos (XGB v3, KMeans) y datos (Clusters, GeoJSON) necesarios.
    """
    # Modelo XGB
    try:
        model = joblib.load(MODEL_XGB_PATH)
    except FileNotFoundError:
        st.error("Error: 'violence_xgb_optimizado_v3.joblib' no encontrado. Asegúrate de que esté en el directorio raíz.")
        model = None

    # Modelo KMeans
    try:
        kmeans = joblib.load(MODEL_KMEANS_PATH)
    except FileNotFoundError:
        st.error("Error: 'kmeans_zonas.joblib' no encontrado. Asegúrate de que esté en el directorio raíz.")
        kmeans = None

    # Info de clusters pre-calculados
    try:
        df_clusters = pd.read_csv(CLUSTER_INFO_PATH)
        df_clusters = enrich_cluster_locations(df_clusters)
    except FileNotFoundError:
        st.error("Error: 'cluster_info.csv' no encontrado. Ejecuta 'crear_cluster_info.py' primero.")
        df_clusters = None

    # GeoJSON de alcaldías
    GEOJSON_PATH = ALCALDIAS_GEOJSON_PATH
    try:
        with open(GEOJSON_PATH, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
    except FileNotFoundError:
        st.error(f"Error: No se encontró 'alcaldias.geojson'. Se buscó en: {GEOJSON_PATH}")
        geojson_data = None

    # Tablas desde la BD
    try:
        df_alcaldias = get_all_alcaldias()
        df_categorias = get_all_crime_categories()
    except Exception as e:
        st.error(f"Error al cargar datos de la base de datos: {e}")
        df_alcaldias = pd.DataFrame()
        df_categorias = pd.DataFrame()

    return model, kmeans, df_clusters, df_alcaldias, df_categorias, geojson_data


@lru_cache(maxsize=20000)
def _normalize_value(value: str) -> str:
    """Cacheable text normalization to reduce repeated unidecode calls."""
    if value is None:
        return 'UNKNOWN'
    normalized = unidecode(value).upper().strip()
    return normalized if normalized else 'UNKNOWN'


def _normalize_series(series: pd.Series) -> pd.Series:
    """Normaliza texto reutilizando los valores cacheados para acelerar grandes cargas."""
    base = series.fillna('UNKNOWN').astype(str)
    normalized = base.map(_normalize_value)
    return normalized.replace('', 'UNKNOWN')


@st.cache_data(show_spinner=False, ttl=3600)
def get_available_years() -> List[int]:
    """Obtiene las opciones de años disponibles en la base de delitos."""
    df_years = run_query(
        """
        SELECT DISTINCT EXTRACT(YEAR FROM fecha_hecho) AS year
        FROM crimes
        WHERE fecha_hecho IS NOT NULL
        ORDER BY year DESC
        """
    )

    if df_years.empty:
        return []

    years = (
        df_years['year']
        .dropna()
        .astype(int)
        .sort_values(ascending=False)
        .tolist()
    )
    return years

@st.cache_data(show_spinner=False, ttl=3600)
def load_duckdb_map_data(year_filter: Optional[int] = None):
    """
    Carga y normaliza los datos de mapa directamente desde DuckDB con mínimo post-procesamiento.
    Permite limitar el histórico para acelerar la carga inicial.
    """
    params = []
    date_filter_sql = ""

    if year_filter is not None:
        date_filter_sql = "AND EXTRACT(YEAR FROM fecha_hecho) = ?"
        params.append(int(year_filter))

    query = f"""
    SELECT
        fecha_hecho,
        hora_hecho,
        delito,
        alcaldia_hecho,
        colonia_hecho,
        latitud,
        longitud,
        (fecha_hecho + hora_hecho) AS datetime
    FROM crimes
    WHERE
        fecha_hecho IS NOT NULL
        AND hora_hecho IS NOT NULL
        AND latitud BETWEEN 19.0 AND 19.6
        AND longitud BETWEEN -99.4 AND -98.9
        {date_filter_sql}
    """
    df = run_query(query, params if params else None)
    if df.empty:
        return df

    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df.dropna(subset=['datetime'], inplace=True)

    df.dropna(subset=['latitud', 'longitud'], inplace=True)

    for col in ['delito', 'alcaldia_hecho', 'colonia_hecho']:
        df[f"{col}_N"] = _normalize_series(df[col])

    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=3600)
def load_cached_historical_dataset():
    """
    Recupera el dataset tradicional usado por mapa.py cuando DuckDB no está disponible.
    Se cachea para evitar volver a leer/limpiar los CSVs pesados en cada rerun.
    """
    try:
        return mapa.load_and_clean_raw_data()
    except AttributeError:
        st.error("El módulo 'mapa' no expone load_and_clean_raw_data().")
    except Exception as exc:
        st.error(f"No se pudo recuperar el dataset histórico tradicional: {exc}")
    return pd.DataFrame()
def _get_hotspot_cache():
    """Ensure a session-level cache for predictive hotspots exists."""
    if 'ui_map_hotspot_cache' not in st.session_state:
        st.session_state.ui_map_hotspot_cache = {}
    return st.session_state.ui_map_hotspot_cache


def _build_hotspot_cache_key(alcaldia: str, categoria: str, fecha, hora: int) -> str:
    """Normalized cache key to reuse hotspot predictions."""
    date_str = pd.to_datetime(fecha).strftime("%Y-%m-%d")
    al_norm = unidecode(str(alcaldia or "ALL")).upper()
    cat_norm = unidecode(str(categoria or "ALL")).upper()
    return f"{HOTSPOT_CACHE_VERSION}|{al_norm}|{cat_norm}|{date_str}|{int(hora)}"


# --- 2. Funciones de Preprocessing (V3) ---
def map_to_time_slot(hour):
    """Convierte una hora (0-23) en una franja horaria categórica."""
    if 0 <= hour <= 5:
        return 'Madrugada'
    elif 6 <= hour <= 11:
        return 'Mañana'
    elif 12 <= hour <= 18:
        return 'Tarde'
    return 'Noche'  # 19-23


def preprocess_inputs_mapa_v3(fecha, hora, lat, lon, alcaldia, categoria, kmeans_model):
    """
    Toma los inputs crudos y los transforma para el pipeline V3 (con zona_hora).
    """
    fecha_dt = pd.to_datetime(fecha)
    dia_de_la_semana = fecha_dt.dayofweek
    es_fin_de_semana = int(dia_de_la_semana >= 5)
    mes = fecha_dt.month
    dia_del_mes = fecha_dt.day
    es_quincena = int(dia_del_mes in [14, 15, 16, 28, 29, 30, 31, 1, 2])

    coords = pd.DataFrame({'latitud': [lat], 'longitud': [lon]})
    zona_cluster = kmeans_model.predict(coords)[0]

    franja_horaria = map_to_time_slot(hora)
    zona_hora = f"{zona_cluster}_{franja_horaria}"
    mes_sin = np.sin(2 * np.pi * mes / 12)
    mes_cos = np.cos(2 * np.pi * mes / 12)

    input_data = {
        'alcaldia_hecho': [alcaldia],
        'categoria_delito': [categoria],
        'dia_de_la_semana': [dia_de_la_semana],
        'es_fin_de_semana': [es_fin_de_semana],
        'es_quincena': [es_quincena],
        'zona_hora': [zona_hora],
        'mes_sin': [mes_sin],
        'mes_cos': [mes_cos],
        'latitud': [lat],
        'longitud': [lon],
        'hora_hecho': [hora],
        'mes_hecho': [mes],
        'zona_cluster': [zona_cluster],
        'franja_horaria': [franja_horaria]
    }

    input_df = pd.DataFrame(input_data)
    return input_df


# --- Función de Ayuda para Color (adaptada a HEX para Folium) ---
def get_color_from_probability(prob):
    """Genera un color HEX (verde → amarillo → rojo) basado en la probabilidad."""
    if prob < 0.65:
        r, g = 0, 180
    elif prob < 0.75:
        g = 255
        r = int(255 * ((prob - 0.65) / 0.10))
    elif prob < 0.85:
        r = 255
        g = int(255 * (1 - ((prob - 0.75) / 0.10)))
    else:
        r, g = 255, 0

    return f'#{r:02x}{g:02x}00'


def render():
    """
    Función principal de renderizado para el módulo del Mapa Interactivo (Folium).
    """
    st.markdown("---")
    st.title("Mapa Interactivo de Incidencia Delictiva")

    available_years = get_available_years()
    history_labels = [ALL_YEARS_OPTION] + [str(year) for year in available_years]
    default_history_index = 0
    if str(DEFAULT_MAP_YEAR) in history_labels:
        default_history_index = history_labels.index(str(DEFAULT_MAP_YEAR))

    with st.sidebar:
        st.subheader("Rango histórico")
        selected_history_label = st.selectbox(
            "Rango histórico a cargar para el mapa:",
            history_labels,
            index=default_history_index,
            help="Filtra los datos por año para acelerar la carga inicial o consulta todo el histórico."
        )
        st.divider()

    selected_year = None
    if selected_history_label != ALL_YEARS_OPTION:
        try:
            selected_year = int(selected_history_label)
        except ValueError:
            selected_year = None

    # Coordenadas base para CDMX
    CENTER_LAT = 19.4326
    CENTER_LON = -99.1332

    # --- 3. MAPA 1: Histórico ---
    st.header("Mapa Histórico de Incidencia (Filtrado)")
    st.markdown(
        "Interactúa con los filtros en la barra lateral para explorar el histórico de delitos."
    )

    df_mapa = load_duckdb_map_data(year_filter=selected_year)  # Cargar datos históricos desde DuckDB
    label_caption = f"año {selected_history_label}" if selected_year else selected_history_label.lower()
    st.caption(f"Registros cargados: {len(df_mapa):,} ({label_caption}).")
    if df_mapa.empty:
        st.warning("No se pudieron cargar los datos del mapa desde DuckDB. Se utilizará el método tradicional.")
        fallback_df = load_cached_historical_dataset()
        fallback_caption = f"{len(fallback_df):,} registros cacheados" if not fallback_df.empty else "sin datos disponibles"
        st.caption(f"Dataset histórico (cacheado): {fallback_caption}.")
        mapa.render_interactive_map(embed=True, df_crime=fallback_df if not fallback_df.empty else None)
    else:
        mapa.render_interactive_map(embed=True, df_crime=df_mapa)

    # ==============================
    # 4. MAPA 2: Predicción Hotspots
    # ==============================
    st.divider()
    st.header("Predicción de Hotspots Futuros")
    st.markdown("Selecciona una fecha, hora, categoría y alcaldía para predecir zonas de riesgo.")

    enable_predictions = st.toggle(
        "Activar módulo de predicción",
        value=False,
        help="Desactiva esta sección si solo necesitas el mapa histórico; al activarla se cargan los modelos de ML."
    )
    if not enable_predictions:
        st.info("Activa el módulo para cargar los modelos y generar hotspots.")
    else:
        model_xgb, model_kmeans, df_clusters, df_alcaldias, df_categorias, geojson_data = load_models_and_data()
        if (
            model_xgb is None or
            model_kmeans is None or
            df_clusters is None or
            df_alcaldias.empty or
            df_categorias.empty or
            geojson_data is None
        ):
            st.error("La sección de predicción no se pudo cargar. Faltan componentes.")
            return

        df_categorias = df_categorias.copy()
        df_categorias['categoria_delito'] = df_categorias['categoria_delito'].astype(str).str.upper()
        all_categories = df_categorias['categoria_delito'].tolist()

        # Controles de fecha y hora
        col_map_1, col_map_2 = st.columns(2)
        with col_map_1:
            map_fecha = st.date_input("Fecha para el Mapa:", datetime.now().date(), key="pred_fecha")
        with col_map_2:
            map_hora = st.slider("Hora para el Mapa (24h):", 0, 23, 22, format="%d:00", key="pred_hora")

        # Controles de alcaldía y categoría
        col_map_3, col_map_4 = st.columns(2)
        with col_map_3:
            map_alcaldia = st.selectbox(
                "Alcaldía a Predecir:",
                options=df_alcaldias['alcaldia_hecho'].tolist() if not df_alcaldias.empty else ["Cargando..."]
            )
        with col_map_4:
            map_categoria = st.selectbox(
                "Categoría de Delito a Predecir:",
                options=all_categories if all_categories else ["Cargando..."]
            )

        # --- Lógica de Predicción en Tiempo Real (V3) ---
        df_hotspots = pd.DataFrame()
        cache_hit = False

        if (
            df_clusters is not None and
            not df_clusters.empty and
            map_alcaldia and
            map_categoria and
            model_xgb is not None and
            model_kmeans is not None
        ):
            cache_store = _get_hotspot_cache()
            cache_key = _build_hotspot_cache_key(map_alcaldia, map_categoria, map_fecha, map_hora)
            cached_result = cache_store.get(cache_key)

            if cached_result is not None:
                df_hotspots = cached_result.copy()
                cache_hit = True
            else:
                hotspots = []
                clusters_filtrados = df_clusters[df_clusters['alcaldia_comun'].str.upper() == map_alcaldia.upper()]

                if clusters_filtrados.empty:
                    st.warning(f"No se encontraron zonas de clúster pre-calculadas para **{map_alcaldia}**.")
                else:
                    for _, cluster in clusters_filtrados.iterrows():
                        try:
                            location_label = (
                                cluster.get('cluster_label')
                                or cluster.get('calle_cercana')
                                or "Ubicación sin referencia"
                            )
                            input_df = preprocess_inputs_mapa_v3(
                                map_fecha,
                                map_hora,
                                cluster['latitud'],
                                cluster['longitud'],
                                cluster['alcaldia_comun'],
                                map_categoria,
                                model_kmeans
                            )

                            probability = model_xgb.predict_proba(input_df)
                            prob_violento = probability[0][1]

                            if prob_violento >= 0.65:
                                hotspots.append({
                                    'lat': cluster['latitud'],
                                    'lon': cluster['longitud'],
                                    'probabilidad_val': prob_violento,
                                    'probabilidad': f"{prob_violento * 100:.1f}%",
                                    'calle': location_label,
                                    'radius': 200 + (prob_violento * 800)
                                })
                        except Exception:
                            # Silenciar errores por cluster individual
                            pass

                df_hotspots = pd.DataFrame(hotspots)
                cache_store[cache_key] = df_hotspots.copy()

        # Crear mapa base Folium para predicción
        m_pred = folium.Map(
            location=[CENTER_LAT, CENTER_LON],
            zoom_start=11,
            tiles='CartoDB dark_matter'
        )

        # Capa GeoJSON de alcaldías (también con NOMGEO y tooltip)
        folium.GeoJson(
            geojson_data,
            name="Alcaldías",
            style_function=lambda feature: {
                "fillColor": "#000000",
                "color": "#FFFFFF",
                "weight": 1,
                "fillOpacity": 0.1,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["NOMGEO"],
                aliases=["Alcaldía:"],
                localize=True
            )
        ).add_to(m_pred)

        # Añadir hotspots como círculos
        if cache_hit:
            st.caption("Predicciones recuperadas del caché local para agilizar la visualización.")

        if not df_hotspots.empty:
            for _, row in df_hotspots.iterrows():
                color = get_color_from_probability(row['probabilidad_val'])

                folium.Circle(
                    location=[row['lat'], row['lon']],
                    radius=row['radius'],  # en metros
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.6,
                    popup=folium.Popup(
                        f"<b>Probabilidad:</b> {row['probabilidad']}<br>"
                        f"<b>Cerca de:</b> {row['calle']}",
                        max_width=250
                    ),
                    tooltip=f"{row['probabilidad']} - {row['calle']}"
                ).add_to(m_pred)

            st.success(f"Mostrando **{len(df_hotspots)} hotspots** (zonas con >= 65% prob. de violencia)")
            with st.expander("Ver detalles de los hotspots"):
                st.dataframe(df_hotspots[['probabilidad', 'calle', 'lat', 'lon']])
        else:
            if map_alcaldia and map_categoria:
                st.info(f"No se encontraron hotspots con >= 65% de probabilidad para {map_categoria} en **{map_alcaldia}**.")

        # Mostrar mapa de predicción
        st_folium(m_pred, width="100%", height=600)
