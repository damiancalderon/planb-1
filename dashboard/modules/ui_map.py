# modules/ui_map.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
# Asegúrate de que este módulo 'database' esté accesible.
# Si no está en el mismo directorio, ajusta el import (ej: from . import database)
import database  
import pydeck as pdk
import json
from datetime import datetime
from pathlib import Path

# --- 1. Carga de Modelos y Datos (V3) ---
@st.cache_resource
def load_models_and_data():
    """
    Carga todos los modelos (XGB v3, KMeans) y datos (Clusters, GeoJSON) necesarios.
    """
    # Define la ruta base para los archivos estáticos. 
    # Es crucial que estas rutas sean correctas en el entorno de ejecución.
    BASE_PATH = Path(__file__).parent.parent 
    
    try:
        model = joblib.load(BASE_PATH / 'violence_xgb_optimizado_v3.joblib') # Carga el modelo V3
    except FileNotFoundError:
        st.error("Error: 'violence_xgb_optimizado_v3.joblib' no encontrado. Asegúrate de que esté en el directorio raíz.")
        model = None
    
    try:
        kmeans = joblib.load(BASE_PATH / 'kmeans_zonas.joblib')
    except FileNotFoundError:
        st.error("Error: 'kmeans_zonas.joblib' no encontrado. Asegúrate de que esté en el directorio raíz.")
        kmeans = None
    
    try:
        df_clusters = pd.read_csv(BASE_PATH / 'cluster_info.csv')
    except FileNotFoundError:
        st.error("Error: 'cluster_info.csv' no encontrado. Ejecuta 'crear_cluster_info.py' primero.")
        df_clusters = None
    
    # Ruta al archivo GeoJSON
    GEOJSON_PATH = BASE_PATH / "alcaldias.geojson"
    try:
        with open(GEOJSON_PATH, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
    except FileNotFoundError:
        st.error(f"Error: No se encontró 'alcaldias.geojson'. Se buscó en: {GEOJSON_PATH}")
        geojson_data = None
            
    # Asume que 'database' está importado y es funcional
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
    if 0 <= hour <= 5: return 'Madrugada'
    elif 6 <= hour <= 11: return 'Mañana'
    elif 12 <= hour <= 18: return 'Tarde'
    return 'Noche' # 19-23

def preprocess_inputs_mapa_v3(fecha, hora, lat, lon, alcaldia, categoria, kmeans_model):
    """
    Toma los inputs crudos y los transforma para el pipeline V3 (con zona_hora).
    """
    fecha_dt = pd.to_datetime(fecha)
    dia_de_la_semana = fecha_dt.dayofweek
    es_fin_de_semana = int(dia_de_la_semana >= 5)
    mes = fecha_dt.month
    dia_del_mes = fecha_dt.day
    es_quincena = int(dia_del_mes in [14,15,16, 28,29,30,31,1,2])
    
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
        'latitud': [lat], 'longitud': [lon], 'hora_hecho': [hora], 'mes_hecho': [mes],
        'zona_cluster': [zona_cluster], 'franja_horaria': [franja_horaria] 
    }
    
    input_df = pd.DataFrame(input_data)
    
    return input_df

# --- Función de Ayuda para Color ---
def get_color_from_probability(prob):
    """Genera un color RGB basado en la probabilidad (Verde a Rojo)."""
    if prob < 0.75:
        # De Verde (0.65) a Amarillo (0.75)
        g = 255
        # Mapea prob de [0.65, 0.75] a [0, 255]
        r = int(255 * ((prob - 0.65) / 0.10)) if prob >= 0.65 else 0
        return [r, g, 0, 180]
    elif prob < 0.85:
        # De Amarillo (0.75) a Rojo (0.85)
        r = 255
        # Mapea prob de [0.75, 0.85] a [255, 0] para el componente G
        g = int(255 * (1 - ((prob - 0.75) / 0.10)))
        return [r, g, 0, 200]
    else:
        # Rojo (>= 0.85)
        return [255, 0, 0, 220]


def render():
    """
    Función principal de renderizado para el módulo del Mapa Interactivo.
    """
    st.markdown("---")
    st.title("🗺️ Mapa Interactivo de Incidencia Delictiva")
    
    # --- Cargar todos los modelos y datos ---
    model_xgb, model_kmeans, df_clusters, df_alcaldias, df_categorias, geojson_data = load_models_and_data()

    # --- Bloque de Validación ---
    if model_xgb is None or \
       model_kmeans is None or \
       df_clusters is None or \
       df_alcaldias.empty or \
       df_categorias.empty or \
       geojson_data is None:
        
        st.error("La aplicación de mapas no se pudo cargar. Faltan componentes.")
        return # Sale de la función si hay un error de carga

    # --- Configuración del ViewState PyDeck ---
    view_state = pdk.ViewState(
        latitude=19.4326,
        longitude=-99.1332,
        zoom=9.5,
        pitch=45
    )

    # --- 3. MAPA 1: Histórico ---
    st.header("Mapa Histórico de Incidencia (Filtrado)")
    st.markdown("Usa los filtros de la barra lateral **exclusiva del Mapa** para explorar los datos históricos.")

    # Crear una barra lateral dedicada para los filtros del mapa
    # Esto aísla los filtros del mapa de la barra lateral principal de navegación
    st.sidebar.markdown("---")
    st.sidebar.header("Filtros del Mapa Histórico")
    
    # --- Filtros en la Barra Lateral del Mapa ---
    if not df_categorias.empty:
        # Aseguramos que las categorías por defecto existan
        default_cats = [c for c in df_categorias['categoria_delito'].tolist() if c in df_categorias['categoria_delito'].iloc[0:2].tolist()]
        if not default_cats:
            default_cats = df_categorias['categoria_delito'].iloc[0:1].tolist() if not df_categorias.empty else []
            
        crime_type = st.sidebar.multiselect(
            "Selecciona tipo de crimen:",
            options=df_categorias['categoria_delito'].tolist(),
            default=default_cats
        )
    else:
        crime_type = []

    hour_slider = st.sidebar.slider(
        "Selecciona hora del día:",
        min_value=0, max_value=23, value=(0, 23), format="%d:00", key="hist_slider"
    )
    crime_classification = st.sidebar.radio(
        "Selecciona clasificación:",
        ('Violent', 'Non-Violent', 'Ambos'), index=2, key="hist_radio"
    )

    # --- Lógica del Mapa Histórico ---
    # Solo llamar a la BD si hay algo seleccionado
    if crime_type:
        df_mapa = database.get_filtered_map_data(
            crime_types=crime_type,
            hour_range=hour_slider,
            classification=crime_classification
        )
    else:
        df_mapa = pd.DataFrame({'longitud': [], 'latitud': []}) # DataFrame vacío

    alcaldias_layer = pdk.Layer(
        'GeoJsonLayer',
        data=geojson_data,
        get_fill_color='[255, 255, 255, 5]',
        get_line_color='[255, 255, 255, 100]',
        get_line_width=100,
        pickable=True,
        auto_highlight=True,
        tooltip={
           "html": "<b>Alcaldía:</b> {nomgeo}", 
           "style": {
                "backgroundColor": "steelblue",
                "color": "white"
           }
        }
    )

    # Solo renderizar Heatmap si hay datos
    if not df_mapa.empty:
        heatmap_layer = pdk.Layer(
            'HeatmapLayer',
            data=df_mapa,
            get_position='[longitud, latitud]',
            opacity=0.8,
            get_weight=1
        )
    else:
        heatmap_layer = pdk.Layer(
            'HeatmapLayer',
            data=pd.DataFrame({'longitud': [0], 'latitud': [0]}), # Layer vacío para evitar error
            get_position='[longitud, latitud]',
            opacity=0,
            get_weight=0
        )
        st.info("Selecciona al menos un tipo de crimen para ver el mapa histórico.")

    st.pydeck_chart(pdk.Deck(
        layers=[heatmap_layer, alcaldias_layer],
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/dark-v9',
    ))

# Llama a la función render() si este módulo se ejecuta directamente (opcional)
if __name__ == '__main__':
    st.set_page_config(page_title="Mapa Interactivo", page_icon="🗺️", layout="wide")
    render()