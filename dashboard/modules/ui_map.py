import streamlit as st
import pandas as pd
import numpy as np
import joblib
import database  
import pydeck as pdk
import json
from datetime import datetime
from pathlib import Path

st.set_page_config(page_title="Mapa Interactivo", page_icon="🗺️", layout="wide")
st.title("🗺️ Mapa Interactivo de Incidencia Delictiva")

# --- 1. Carga de Modelos y Datos (V3) ---
@st.cache_resource
def load_models_and_data():
    """
    Carga todos los modelos (XGB v3, KMeans) y datos (Clusters, GeoJSON) necesarios.
    """
    try:
        model = joblib.load('violence_xgb_optimizado_v3.joblib') # Carga el modelo V3
    except FileNotFoundError:
        st.error("Error: 'violence_xgb_optimizado_v3.joblib' no encontrado.")
        model = None
    
    try:
        kmeans = joblib.load('kmeans_zonas.joblib')
    except FileNotFoundError:
        st.error("Error: 'kmeans_zonas.joblib' no encontrado.")
        kmeans = None
    
    try:
        df_clusters = pd.read_csv('cluster_info.csv')
    except FileNotFoundError:
        st.error("Error: 'cluster_info.csv' no encontrado.")
        st.warning("Ejecuta 'crear_cluster_info.py' primero.")
        df_clusters = None
    
    # Ruta absoluta al archivo GeoJSON
    GEOJSON_PATH = Path(__file__).parent.parent / "alcaldias.geojson"
    try:
        with open(GEOJSON_PATH, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
    except FileNotFoundError:
        st.error(f"Error: No se encontró 'alcaldias.geojson'. Se buscó en: {GEOJSON_PATH}")
        geojson_data = None
            
    df_alcaldias = database.get_all_alcaldias()
    df_categorias = database.get_all_crime_categories()
    
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

# --- Cargar todos los modelos y datos ---
model_xgb, model_kmeans, df_clusters, df_alcaldias, df_categorias, geojson_data = load_models_and_data()

# --- Bloque de Validación ---
if model_xgb is None or \
   model_kmeans is None or \
   df_clusters is None or \
   df_alcaldias.empty or \
   df_categorias.empty or \
   geojson_data is None:
    
    st.error("La aplicación no se pudo cargar. Faltan componentes (modelos, CSV, GeoJSON o datos de la BD).")
    st.stop()

# --- 3. MAPA 1: Histórico ---
st.header("Mapa Histórico de Incidencia (Filtrado)")
st.markdown("Usa los filtros de la barra lateral para explorar los datos históricos.")

# --- Filtros en la Barra Lateral ---
st.sidebar.header("Filtros del Mapa Histórico")
if not df_categorias.empty:
    default_categories = df_categorias['categoria_delito'].iloc[0:2].tolist() 
    crime_type = st.sidebar.multiselect(
        "Selecciona tipo de crimen:",
        options=df_categorias['categoria_delito'].tolist(),
        default=default_categories
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
df_mapa = database.get_filtered_map_data(
    crime_types=crime_type,
    hour_range=hour_slider,
    classification=crime_classification
)

view_state = pdk.ViewState(
    latitude=19.4326,
    longitude=-99.1332,
    zoom=9.5,
    pitch=45
)

alcaldias_layer = pdk.Layer(
    'GeoJsonLayer',
    data=geojson_data,
    get_fill_color='[255, 255, 255, 5]',
    get_line_color='[255, 255, 255, 100]',
    get_line_width=100,
    pickable=True,
    auto_highlight=True,
    tooltip={
       "html": "<b>Alcaldía:</b> {nomgeo}", # TODO: Ajustar según el campo correcto en el GeoJSON
       "style": {
            "backgroundColor": "steelblue",
            "color": "white"
       }
    }
)

heatmap_layer = pdk.Layer(
    'HeatmapLayer',
    data=df_mapa,
    get_position='[longitud, latitud]',
    opacity=0.8,
    get_weight=1
)
st.pydeck_chart(pdk.Deck(
    layers=[heatmap_layer, alcaldias_layer],
    initial_view_state=view_state,
    map_style='mapbox://styles/mapbox/dark-v9',
))

st.divider()
st.header("Mapa 2: Explorador Histórico por Fecha (Animado)")
st.markdown("Selecciona un año y luego desliza el *slider* de 'Mes' para animar el heatmap de crímenes.")

# --- Controles de Filtro para Mapa 2 ---
col1, col2 = st.columns([1, 3])
with col1:
    # 1. Selector de Año
    lista_anios = database.run_query("SELECT DISTINCT anio_hecho FROM crimes ORDER BY anio_hecho DESC")['anio_hecho'].tolist()
    if not lista_anios:
        lista_anios = [2024, 2023, 2022] # Fallback
    
    selected_anio = st.selectbox(
        "Selecciona el Año:",
        options=lista_anios,
        key="map2_anio"
    )

with col2:
    meses_dict = {
        "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4, "Mayo": 5, "Junio": 6,
        "Julio": 7, "Agosto": 8, "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12
    }
    
    # st.select_slider usa los nombres (keys) como opciones
    selected_mes_nombre = st.select_slider(
        "Selecciona el Mes:",
        options=meses_dict.keys(),
        key="map2_mes_slider"
    )
    # Convertimos el nombre del mes de vuelta a un número para la BD
    selected_mes_num = meses_dict[selected_mes_nombre]

# --- Lógica del Mapa 2 ---
df_mapa_2 = database.get_map_data_by_date(selected_anio, selected_mes_num)

heatmap_layer_2 = pdk.Layer(
    'HeatmapLayer',
    data=df_mapa_2,
    get_position='[longitud, latitud]',
    opacity=0.8,
    pickable=False,
    get_weight=1
)

st.pydeck_chart(pdk.Deck(
    layers=[heatmap_layer_2, alcaldias_layer],
    initial_view_state=view_state,
    map_style='mapbox://styles/mapbox/dark-v9',
))

st.info(f"Mostrando {len(df_mapa_2)} puntos para {selected_mes_nombre} de {selected_anio}.")

# --- 4. MAPA 2: Predicción Animada (V3) ---
st.divider()
st.header("Predicción de Hotspots Futuros (Animada)")
st.markdown("Selecciona una fecha, hora, categoría y alcaldía. El mapa mostrará dinámicamente las zonas de clúster y su riesgo.")

# --- Controles para el mapa futuro ---
col_map_1, col_map_2 = st.columns(2)
with col_map_1:
    map_fecha = st.date_input("Fecha para el Mapa:", datetime.now().date(), key="pred_fecha")
with col_map_2:
    map_hora = st.slider("Hora para el Mapa (24h):", 0, 23, 22, format="%d:00", key="pred_hora")

col_map_3, col_map_4 = st.columns(2)
with col_map_3:
    map_alcaldia = st.selectbox(
        "Alcaldía a Predecir:",
        options=df_alcaldias['alcaldia_hecho'].tolist()
    )
with col_map_4:
    map_categoria = st.selectbox(
        "Categoría de Delito a Predecir:",
        options=df_categorias['categoria_delito'].tolist()
    )

# --- Función de Ayuda para Color ---
def get_color_from_probability(prob):
    if prob < 0.75:
        g = 255
        r = int(255 * ((prob - 0.65) / 0.10))
        return [r, g, 0, 180]
    elif prob < 0.85:
        r = 255
        g = int(255 * (1 - ((prob - 0.75) / 0.10)))
        return [r, g, 0, 200]
    else:
        return [255, 0, 0, 220]

# --- Lógica de Predicción en Tiempo Real (V3) ---
hotspots = []

clusters_filtrados = df_clusters[df_clusters['alcaldia_comun'].str.upper() == map_alcaldia.upper()]

if clusters_filtrados.empty and map_alcaldia:
     st.warning(f"No se encontraron zonas de clúster pre-calculadas para {map_alcaldia}.")

for index, cluster in clusters_filtrados.iterrows():
    try:
        # Usa la función de preprocessing V3 
        #TODO mejorar con datos del crowling
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
                'probabilidad': f"{prob_violento*100:.1f}%",
                'calle': cluster['calle_cercana'],
                'radius': 200 + (prob_violento * 800),
                'color_rgb': get_color_from_probability(prob_violento)
            })
    except Exception as e:
        pass 

df_hotspots = pd.DataFrame(hotspots)

# --- Renderizar el Mapa de Predicción ---
alcaldias_layer_pred = pdk.Layer(
    'GeoJsonLayer',
    data=geojson_data,
    get_fill_color='[255, 255, 255, 20]',
    get_line_color='[255, 255, 255, 80]',
    get_line_width=100,
)

hotspots_layer = pdk.Layer(
    'ScatterplotLayer',
    data=df_hotspots,
    get_position='[lon, lat]',
    get_fill_color='color_rgb',
    get_radius='radius',
    pickable=True,
)

tooltip = {
    "html": "<b>Probabilidad: {probabilidad}</b><br/>Cerca de: {calle}",
    "style": { "backgroundColor": "steelblue", "color": "white" }
}

st.pydeck_chart(pdk.Deck(
    layers=[alcaldias_layer_pred, hotspots_layer],
    initial_view_state=view_state,
    map_style='mapbox://styles/mapbox/dark-v9',
    tooltip=tooltip
))

if df_hotspots.empty:
    st.info("No se encontraron hotspots con >= 65% de probabilidad para esta combinación de filtros.")
else:
    st.success(f"Mostrando {len(df_hotspots)} hotspots (zonas con >= 65% prob. de violencia)")
    with st.expander("Ver detalles de los hotspots"):
        st.dataframe(df_hotspots[['probabilidad', 'calle', 'lat', 'lon']])