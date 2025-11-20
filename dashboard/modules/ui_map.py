# modules/ui_map.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import database
import json
from datetime import datetime
from pathlib import Path

import folium
from streamlit_folium import st_folium

import mapa

# =========================
# RUTAS GLOBALES (ajusta si hace falta)
# =========================
BASE_PATH = Path(__file__).parent.parent


# --- 1. Carga de Modelos y Datos (V3) ---
@st.cache_resource
def load_models_and_data():
    """
    Carga todos los modelos (XGB v3, KMeans) y datos (Clusters, GeoJSON) necesarios.
    """
    base_path = Path(__file__).parent.parent

    # Modelo XGB
    try:
        model = joblib.load(base_path / 'violence_xgb_optimizado_v3.joblib')
    except FileNotFoundError:
        st.error("Error: 'violence_xgb_optimizado_v3.joblib' no encontrado. Asegúrate de que esté en el directorio raíz.")
        model = None

    # Modelo KMeans
    try:
        kmeans = joblib.load(base_path / 'kmeans_zonas.joblib')
    except FileNotFoundError:
        st.error("Error: 'kmeans_zonas.joblib' no encontrado. Asegúrate de que esté en el directorio raíz.")
        kmeans = None

    # Info de clusters pre-calculados
    try:
        df_clusters = pd.read_csv(base_path / 'cluster_info.csv')
    except FileNotFoundError:
        st.error("Error: 'cluster_info.csv' no encontrado. Ejecuta 'crear_cluster_info.py' primero.")
        df_clusters = None

    # GeoJSON de alcaldías
    GEOJSON_PATH = base_path / "alcaldias.geojson"
    try:
        with open(GEOJSON_PATH, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
    except FileNotFoundError:
        st.error(f"Error: No se encontró 'alcaldias.geojson'. Se buscó en: {GEOJSON_PATH}")
        geojson_data = None

    # Tablas desde la BD
    try:
        df_alcaldias = database.get_all_alcaldias()
        df_categorias = database.get_all_crime_categories()
    except Exception as e:
        st.error(f"Error al cargar datos de la base de datos: {e}")
        df_alcaldias = pd.DataFrame()
        df_categorias = pd.DataFrame()

    return model, kmeans, df_clusters, df_alcaldias, df_categorias, geojson_data


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
    st.title("🗺️ Mapa Interactivo de Incidencia Delictiva")

    # --- Cargar todos los modelos y datos ---
    model_xgb, model_kmeans, df_clusters, df_alcaldias, df_categorias, geojson_data = load_models_and_data()

    # --- Bloque de Validación ---
    if (
        model_xgb is None or
        model_kmeans is None or
        df_clusters is None or
        df_alcaldias.empty or
        df_categorias.empty or
        geojson_data is None
    ):
        st.error("La aplicación de mapas no se pudo cargar. Faltan componentes.")
        return

    # Coordenadas base para CDMX
    CENTER_LAT = 19.4326
    CENTER_LON = -99.1332

    if not df_categorias.empty:
        df_categorias = df_categorias.copy()
        df_categorias['categoria_delito'] = df_categorias['categoria_delito'].astype(str).str.upper()
        all_categories = df_categorias['categoria_delito'].tolist()
    else:
        all_categories = []

    # --- 3. MAPA 1: Histórico ---
    st.header("Mapa Histórico de Incidencia (Filtrado)")
    st.markdown(
        "Interactúa con los filtros en la barra lateral para explorar el histórico de delitos."
    )

    mapa.render_interactive_map(embed=True)

    # ==============================
    # 4. MAPA 2: Predicción Hotspots
    # ==============================
    st.divider()
    st.header("Predicción de Hotspots Futuros")
    st.markdown("Selecciona una fecha, hora, categoría y alcaldía para predecir zonas de riesgo.")

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
    hotspots = []

    if (
        df_clusters is not None and
        not df_clusters.empty and
        map_alcaldia and
        map_categoria and
        model_xgb is not None and
        model_kmeans is not None
    ):
        clusters_filtrados = df_clusters[df_clusters['alcaldia_comun'].str.upper() == map_alcaldia.upper()]

        if clusters_filtrados.empty:
            st.warning(f"No se encontraron zonas de clúster pre-calculadas para **{map_alcaldia}**.")
        else:
            for _, cluster in clusters_filtrados.iterrows():
                try:
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
                            'calle': cluster['calle_cercana'],
                            'radius': 200 + (prob_violento * 800)
                        })
                except Exception:
                    # Silenciar errores por cluster individual
                    pass

    df_hotspots = pd.DataFrame(hotspots)

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
