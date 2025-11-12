import streamlit as st
import pandas as pd
import numpy as np
import joblib
import database  # Módulo de base de datos
import requests  # Para llamar a n8n
import json      # Para el chat de IA
from datetime import datetime
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import pydeck as pdk 
from pathlib import Path # <-- FALTABA ESTA IMPORTACIÓN

st.set_page_config(page_title="Alerta Policial", page_icon="🛡️", layout="wide")

# --- 1. URL del Webhook de n8n ---
N8N_WEBHOOK_URL = "<URL_DE_PRODUCCIÓN_DE_N8N>" # Pega la URL de producción de n8n aquí

# --- 2. Carga de Modelos y Datos ---
@st.cache_resource
def load_models_and_data():
    """
    Carga todos los modelos (XGB, KMeans) y datos (Clusters) necesarios al iniciar.
    """
    try:
        model = joblib.load('violence_xgb_optimizado_v3.joblib')
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
    
    df_alcaldias = database.get_all_alcaldias()
    df_categorias = database.get_all_crime_categories()
    
    GEOJSON_PATH = Path(__file__).parent.parent / "alcaldias.geojson"
    try:
        with open(GEOJSON_PATH, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
    except FileNotFoundError:
        st.error(f"Error: No se encontró 'alcaldias.geojson'. Se buscó en: {GEOJSON_PATH}")
        geojson_data = None
    
    return model, kmeans, df_clusters, df_alcaldias, df_categorias, geojson_data 

# --- 3. Funciones de Geocoding y Helpers ---
@st.cache_resource
def get_geolocator():
    geolocator = Nominatim(user_agent="cdmx-insights-project")
    return geolocator

@st.cache_data
def get_coords_from_address(address):
    if not address: return None
    try:
        geolocator = get_geolocator()
        location = geolocator.geocode(f"{address}, Mexico City", timeout=10)
        return (location.latitude, location.longitude) if location else None
    except Exception:
        return None

def map_to_time_slot(hour):
    if 0 <= hour <= 5: return 'Madrugada'
    elif 6 <= hour <= 11: return 'Mañana'
    elif 12 <= hour <= 18: return 'Tarde'
    return 'Noche' # 19-23

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

# --- 4. Funciones de Preprocessing (V2) ---
def preprocess_inputs(fecha, hora, lat, lon, alcaldia, categoria, kmeans_model):
    fecha_dt = pd.to_datetime(fecha)
    dia_de_la_semana = fecha_dt.dayofweek
    es_fin_de_semana = int(dia_de_la_semana >= 5)
    mes = fecha_dt.month
    dia_del_mes = fecha_dt.day
    es_quincena = int(dia_del_mes in [14,15,16, 28,29,30,31,1,2])
    
    coords = pd.DataFrame({'latitud': [lat], 'longitud': [lon]})
    zona_cluster = kmeans_model.predict(coords)[0]
    
    franja_horaria = map_to_time_slot(hora)
    mes_sin = np.sin(2 * np.pi * mes / 12)
    mes_cos = np.cos(2 * np.pi * mes / 12)
    
    input_data = {
        'hora_hecho': [hora], 'mes_hecho': [mes], 'alcaldia_hecho': [alcaldia],
        'categoria_delito': [categoria], 'latitud': [lat], 'longitud': [lon],
        'dia_de_la_semana': [dia_de_la_semana], 'es_fin_de_semana': [es_fin_de_semana],
        'es_quincena': [es_quincena], 'zona_cluster': [zona_cluster],
        'franja_horaria': [franja_horaria],
        'mes_sin': [mes_sin], 'mes_cos': [mes_cos],
    }
    
    input_df = pd.DataFrame(input_data)
    
    contexto = {
        "zona_cluster": int(zona_cluster), "alcaldia": alcaldia, "categoria": categoria,
        "hora": hora, "es_fin_de_semana": es_fin_de_semana, "es_quincena": es_quincena
    }
    
    return input_df, contexto

def preprocess_inputs_mapa(fecha, hora, lat, lon, alcaldia, categoria, kmeans_model):
    fecha_dt = pd.to_datetime(fecha)
    dia_de_la_semana = fecha_dt.dayofweek
    es_fin_de_semana = int(dia_de_la_semana >= 5)
    mes = fecha_dt.month
    dia_del_mes = fecha_dt.day
    es_quincena = int(dia_del_mes in [14,15,16, 28,29,30,31,1,2])
    
    coords = pd.DataFrame({'latitud': [lat], 'longitud': [lon]})
    zona_cluster = kmeans_model.predict(coords)[0]
    
    franja_horaria = map_to_time_slot(hora)
    mes_sin = np.sin(2 * np.pi * mes / 12)
    mes_cos = np.cos(2 * np.pi * mes / 12)
    
    input_data = {
        'hora_hecho': [hora], 'mes_hecho': [mes], 'alcaldia_hecho': [alcaldia],
        'categoria_delito': [categoria], 'latitud': [lat], 'longitud': [lon],
        'dia_de_la_semana': [dia_de_la_semana], 'es_fin_de_semana': [es_fin_de_semana],
        'es_quincena': [es_quincena], 'zona_cluster': [zona_cluster],
        'franja_horaria': [franja_horaria],
        'mes_sin': [mes_sin], 'mes_cos': [mes_cos],
    }
    
    input_df = pd.DataFrame(input_data)
    return input_df

# --- 5. Función de Chat ---
def call_gemini_analyst(pregunta_usuario, contexto_modelo):
    if not N8N_WEBHOOK_URL.startswith("https"):
        return "Error: La URL del Webhook de n8n no está configurada en el script."

    payload = {
        "pregunta_usuario": pregunta_usuario,
        "contexto": contexto_modelo
    }
    try:
        response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=120)
        response.raise_for_status()
        try:
            respuesta_json = response.json()
            texto_respuesta = respuesta_json['content']['parts'][0]['text']
            return texto_respuesta
        except Exception as e:
            return f"Error parseando JSON de n8n: {e}. Respuesta cruda: {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Error de conexión con n8n: {e}"

# --- 6. Página de Alerta (Función Principal) ---
def show_alert_page():
    
    st.title("🛡️ Sistema de Alerta Predictiva y Análisis de IA")
    
    model_xgb, model_kmeans, df_clusters, df_alcaldias, df_categorias, geojson_data = load_models_and_data()
    
    if model_xgb is None or \
       model_kmeans is None or \
       df_clusters is None or \
       df_alcaldias.empty or \
       df_categorias.empty or \
       geojson_data is None:
        
        st.error("La aplicación no se pudo cargar. Faltan componentes (modelos, CSV, GeoJSON o datos de la BD).")
        st.stop()
    # --- FIN DE LA CORRECCIÓN ---

    if "latitud" not in st.session_state:
        st.session_state.latitud = 19.432608 
    if "longitud" not in st.session_state:
        st.session_state.longitud = -99.133209
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_context" not in st.session_state:
        st.session_state.current_context = None

    # --- CARACTERÍSTICA 1: BÚSQUEDA DE CALLES ---
    st.subheader("1. Predicción Individual")
    st.markdown("Busca una dirección o ingresa coordenadas para una predicción específica.")
    
    address_query = st.text_input("Buscar dirección (ej. 'Angel de la Independencia'):")
    if st.button("Buscar Dirección"):
        with st.spinner("Geocodificando..."):
            coords = get_coords_from_address(address_query)
            if coords:
                st.session_state.latitud = coords[0]
                st.session_state.longitud = coords[1]
                st.success(f"Dirección encontrada: {coords[0]:.6f}, {coords[1]:.6f}")
            else:
                st.error("No se pudo encontrar la dirección.")

    # --- Formulario de Predicción ---
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            selected_fecha = st.date_input("Fecha:", datetime.now())
        with col2:
            selected_hora = st.slider("Hora (24h):", 0, 23, datetime.now().hour, format="%d:00")
        
        col3, col4 = st.columns(2)
        with col3:
            selected_lat = st.number_input("Latitud:", value=st.session_state.latitud, format="%.6f", key="lat_input")
        with col4:
            selected_lon = st.number_input("Longitud:", value=st.session_state.longitud, format="%.6f", key="lon_input")

        col5, col6 = st.columns(2)
        with col5:
            selected_alcaldia = st.selectbox("Alcaldía:", options=df_alcaldias['alcaldia_hecho'].tolist())
        with col6:
            selected_categoria = st.selectbox("Categoría:", options=df_categorias['categoria_delito'].tolist())
        
        submit_button = st.form_submit_button(label="Generar Predicción")

    # --- Lógica de Predicción ---
    if submit_button:
        try:
            input_df, contexto = preprocess_inputs(
                selected_fecha, selected_hora, selected_lat, selected_lon,
                selected_alcaldia, selected_categoria, model_kmeans
            )
            prediction = model_xgb.predict(input_df)
            probability = model_xgb.predict_proba(input_df)
            
            pred_index = prediction[0]
            pred_name = 'Violento' if pred_index == 1 else 'No-Violento'
            confidence = probability[0][pred_index] * 100
            
            st.divider()
            if pred_name == 'Violento':
                st.error(f"ALERTA: Predicción de Crimen VIOLENTO (Confianza: {confidence:.1f}%)")
            else:
                st.success(f"Predicción de Crimen NO-VIOLENTO (Confianza: {confidence:.1f}%)")

            st.session_state.current_context = contexto
            st.session_state.current_context["prediccion"] = pred_name
            st.session_state.current_context["confianza"] = f"{confidence:.1f}"
            st.session_state.chat_history = []

        except Exception as e:
            st.error(f"Error al procesar la predicción: {e}")
            st.session_state.current_context = None

    # --- CARACTERÍSTICA 2: CHAT CON CONTEXTO ---
    if st.session_state.current_context:
        st.subheader("2. Analista de IA (Gemini)")
        st.markdown("Haz una pregunta sobre la predicción que acabas de generar.")
        
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ej. ¿Por qué es tan alta la probabilidad?"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("La ia está analizando..."):
                    response = call_gemini_analyst(prompt, st.session_state.current_context)
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # --- CARACTERÍSTICA 3: MAPA DE HOTSPOTS ---
    st.divider()
    st.subheader("3. Mapa de Hotspots Futuros (Probabilidad >= 65%)")
    
    col_map_1, col_map_2 = st.columns(2)
    with col_map_1:
        map_fecha = st.date_input("Fecha para el Mapa:", datetime.now().date(), key="map_fecha")
    with col_map_2:
        map_hora = st.slider("Hora para el Mapa (24h):", 0, 23, 22, format="%d:00", key="map_hora")

    col_map_3, col_map_4 = st.columns(2)
    with col_map_3:
        map_alcaldia = st.selectbox(
            "Alcaldía a Predecir:",
            options=df_alcaldias['alcaldia_hecho'].tolist(),
            key="map_alcaldia"
        )
    with col_map_4:
        map_categoria = st.selectbox(
            "Categoría de Delito a Predecir:",
            options=df_categorias['categoria_delito'].tolist(),
            key="map_categoria"
        )
    
    hotspots = []
    clusters_filtrados = df_clusters[df_clusters['alcaldia_comun'].str.upper() == map_alcaldia.upper()]

    if clusters_filtrados.empty and map_alcaldia:
         st.warning(f"No se encontraron zonas de clúster pre-calculadas para {map_alcaldia}.")

    for index, cluster in clusters_filtrados.iterrows():
        try:
            input_df = preprocess_inputs_mapa(
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
        except Exception:
            pass 

    df_hotspots = pd.DataFrame(hotspots)

    # --- Renderizar el Mapa de Predicción ---
    view_state = pdk.ViewState(latitude=19.4326, longitude=-99.1332, zoom=9.5, pitch=45)
    
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

# --- 7. Lógica de Autenticación (Login) ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
def check_password():
    if st.session_state["password"] == "policia123":
        st.session_state.logged_in = True
    else:
        st.sidebar.error("Contraseña incorrecta.")

if not st.session_state.logged_in:
    st.sidebar.header("Acceso Policial")
    st.sidebar.text_input("Contraseña:", type="password", on_change=check_password, key="password")
    st.info("Esta página es solo para personal autorizado.")
else:
    show_alert_page() # Llama a la función principal que contiene todo