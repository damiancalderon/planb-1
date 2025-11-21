import streamlit as st
import pandas as pd
import numpy as np
import joblib
import database  
import requests  
import json      
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import pydeck as pdk 
import unidecode
from pathlib import Path 

# --- Configuración de página (debe estar fuera de render) ---
st.set_page_config(page_title="Alerta Policial", page_icon="🛡️", layout="wide")

# --- 1. URL del Webhook de n8n ---
N8N_WEBHOOK_URL = "https://n8n.tektititc.org/webhook/90408216-1fba-4806-b062-2ab8afb30fea"

# --- 2. Carga de Modelos y Datos ---
@st.cache_resource
def load_models_and_data():
    try:
        model = joblib.load('violence_xgb_optimizado_v3.joblib')
    except FileNotFoundError:
        st.error("Error: 'violence_xgb_optimizado_v3.joblib' no encontrado.")
        st.warning("Asegúrate de haber ejecutado la última versión de 'train_model.py'.")
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
        df_clusters = None
    
    df_alcaldias = database.get_all_alcaldias()
    df_categorias = database.get_all_crime_categories()
    
    GEOJSON_PATH = Path(__file__).parent.parent / "alcaldias.geojson"
    try:
        with open(GEOJSON_PATH, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
    except FileNotFoundError:
        st.error(f"Error: No se encontró 'alcaldias.geojson'.")
        geojson_data = None
    
    return model, kmeans, df_clusters, df_alcaldias, df_categorias, geojson_data

# --- 3. Funciones de Geocoding y Helpers ---
@st.cache_resource
def get_geocoders():
    """Crea un geolocator y un reverse_geocoder cacheados."""
    geolocator = Nominatim(user_agent="cdmx-insights-project")
    reverse_geocoder = RateLimiter(geolocator.reverse, min_delay_seconds=1.1, error_wait_seconds=10)
    return geolocator, reverse_geocoder

@st.cache_data
def get_coords_from_address(address):
    """Convierte una dirección en latitud y longitud."""
    if not address: return None
    try:
        geolocator, _ = get_geocoders()
        location = geolocator.geocode(f"{address}, Mexico City", timeout=10)
        return (location.latitude, location.longitude) if location else None
    except Exception:
        return None

@st.cache_data
def get_alcaldia_from_coords(lat, lon, all_alcaldias_list):
    """
    Convierte coordenadas (lat, lon) en un nombre de Alcaldía
    que coincida con la base de datos. (VERSIÓN CORREGIDA)
    """
    try:
        _, reverse_geocoder = get_geocoders() 
        location = reverse_geocoder((lat, lon), language='es', timeout=10)
        
        if not location:
            st.error("Error de Geopy: No se encontró ninguna ubicación para esas coordenadas.")
            return None
        
        address_parts = location.raw.get('address', {})
        
        found_name = address_parts.get('city_district', 
                                     address_parts.get('county', 
                                     address_parts.get('municipality',
                                     address_parts.get('borough'))))
        
        if not found_name:
            st.warning("Advertencia de Geopy: La API no devolvió un nombre de alcaldía reconocible.")
            print(f"DEBUG: API devolvió estas llaves: {address_parts.keys()}")
            return None

        # --- Lógica de Comparación ---
        normalized_name_from_api = unidecode.unidecode(found_name).upper()
        
        for alcaldia_db_original in all_alcaldias_list:
            normalized_db_name = unidecode.unidecode(alcaldia_db_original).upper()
            
            if normalized_name_from_api == normalized_db_name:
                return alcaldia_db_original # Devuelve el nombre OFICIAL de la BD
        
        st.warning(f"Se encontró '{found_name}' pero no coincide con ninguna alcaldía en la BD.")
        return None 
        
    except Exception as e:
        st.error(f"Error fatal en get_alcaldia_from_coords: {e}")
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

# --- 4. Funciones de Preprocessing ---
def preprocess_inputs_v3(fecha, hora, lat, lon, alcaldia, categoria, kmeans_model):
    """
    Transforma los inputs del formulario para el modelo V3 (con zona_hora).
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
    }
    input_df = pd.DataFrame(input_data)
    
    contexto = {
        "zona_cluster": int(zona_cluster), "alcaldia": alcaldia, "categoria": categoria,
        "hora": hora, "es_fin_de_semana": es_fin_de_semana, "es_quincena": es_quincena
    }
    return input_df, contexto

@st.cache_data(ttl=3600) 
def precalculate_48h_simulation_v3(_model_xgb, _model_kmeans, _df_clusters, 
                               map_fecha_sim, map_categoria_sim):
    """
    Ejecuta el modelo V3 para TODAS las 50 zonas en un rango de 48 horas.
    """
    hotspots_48h = []
    start_date = pd.to_datetime(map_fecha_sim)

    for hora_futura in range(48): # Itera de 0 a 47
        fecha_actual = start_date + timedelta(hours=hora_futura)
        hora_actual = fecha_actual.hour
        
        for index, cluster in _df_clusters.iterrows():
            try:
                fecha_dt = pd.to_datetime(fecha_actual)
                dia_de_la_semana = fecha_dt.dayofweek
                es_fin_de_semana = int(dia_de_la_semana >= 5)
                mes = fecha_dt.month
                dia_del_mes = fecha_dt.day
                es_quincena = int(dia_del_mes in [14,15,16, 28,29,30,31,1,2])
                
                zona_cluster = cluster['cluster_id']
                franja_horaria = map_to_time_slot(hora_actual)
                zona_hora = f"{zona_cluster}_{franja_horaria}" 
                mes_sin = np.sin(2 * np.pi * mes / 12)
                mes_cos = np.cos(2 * np.pi * mes / 12)
                
                input_data = {
                    'alcaldia_hecho': [cluster['alcaldia_comun']], 
                    'categoria_delito': [map_categoria_sim],
                    'dia_de_la_semana': [dia_de_la_semana], 
                    'es_fin_de_semana': [es_fin_de_semana],
                    'es_quincena': [es_quincena], 
                    'zona_hora': [zona_hora], 
                    'mes_sin': [mes_sin], 
                    'mes_cos': [mes_cos],
                }
                input_df = pd.DataFrame(input_data)
                
                probability = _model_xgb.predict_proba(input_df)
                prob_violento = probability[0][1]
                
                if prob_violento >= 0.65: 
                    hotspots_48h.append({
                        'hora_simulacion': hora_futura, 
                        'lat': cluster['latitud'],
                        'lon': cluster['longitud'],
                        'probabilidad': f"{prob_violento*100:.1f}%",
                        'calle': cluster['calle_cercana'],
                        'radius': 200 + (prob_violento * 800),
                        'color_rgb': get_color_from_probability(prob_violento)
                    })
            except Exception as e:
                print(f"Error prediciendo cluster {cluster['cluster_id']} / hora {hora_futura}: {e}")
    
    return pd.DataFrame(hotspots_48h)

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

# --- 6. Función Principal de Renderizado ---
def render():
    """
    Función principal que renderiza toda la página de Alerta Predictiva.
    """
    st.title("Sistema de Alerta Predictiva y Análisis de IA")
    
    model_xgb, model_kmeans, df_clusters, df_alcaldias, df_categorias, geojson_data = load_models_and_data()
    
    if model_xgb is None or \
       model_kmeans is None or \
       df_clusters is None or \
       df_alcaldias.empty or \
       df_categorias.empty or \
       geojson_data is None:
        
        st.error("La aplicación no se pudo cargar. Faltan componentes (modelos, CSV, GeoJSON o datos de la BD).")
        st.stop()

    if "latitud" not in st.session_state:
        st.session_state.latitud = 19.432608 # Zócalo
    if "longitud" not in st.session_state:
        st.session_state.longitud = -99.133209 # Zócalo
    if "alcaldia_seleccionada" not in st.session_state:
        st.session_state.alcaldia_seleccionada = df_alcaldias['alcaldia_hecho'].tolist()[0] 
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_context" not in st.session_state:
        st.session_state.current_context = None

    # --- CARACTERÍSTICA 1: BÚSQUEDA DE CALLES ---
    st.subheader("1. Predicción Individual")
    st.markdown("Busca una dirección o ingresa coordenadas para una predicción específica.")
    address_query = st.text_input("Buscar dirección (ej. 'Angel de la Independencia'):")
    if st.button("Buscar Dirección"):
        with st.spinner("Geocodificando y buscando alcaldía..."):
            coords = get_coords_from_address(address_query)
            if coords:
                st.session_state.latitud = coords[0]
                st.session_state.longitud = coords[1]
                st.success(f"Dirección encontrada: {coords[0]:.6f}, {coords[1]:.6f}")
                
                alcaldia_list = df_alcaldias['alcaldia_hecho'].tolist()
                found_alcaldia = get_alcaldia_from_coords(coords[0], coords[1], alcaldia_list)
                if found_alcaldia:
                    st.session_state.alcaldia_seleccionada = found_alcaldia
                else:
                    st.warning("No se pudo auto-detectar la alcaldía para esa coordenada.")
            else:
                st.error("No se pudo encontrar la dirección.")
    
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
            selected_alcaldia = st.selectbox("Alcaldía:", 
                                             options=df_alcaldias['alcaldia_hecho'].tolist(),
                                             key="alcaldia_seleccionada")
        with col6:
            selected_categoria = st.selectbox("Categoría:", options=df_categorias['categoria_delito'].tolist())
        submit_button = st.form_submit_button(label="Generar Predicción")
    
    if submit_button:
        try:
            input_df, contexto = preprocess_inputs_v3(
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

    st.divider()
    st.header("3. Simulación de Zonas de Riesgo Futuras (Próximas 48h)")
    st.markdown("Selecciona los parámetros y genera una simulación. Luego usa el slider de 'Hora' para animar la predicción.")

    col_map_1, col_map_2 = st.columns(2)
    with col_map_1:
        map_fecha_sim = st.date_input("Fecha de Inicio:", datetime.now().date(), key="sim_fecha")
    with col_map_2:
        map_categoria_sim = st.selectbox(
            "Categoría a Simular:",
            options=df_categorias['categoria_delito'].tolist(),
            key="sim_categoria"
        )
    
    if st.button("Generar Simulación de 48h"):
        with st.spinner(f"Calculando predicciones para TODA la CDMX (48 horas)..."):
            df_simulacion_completa = precalculate_48h_simulation_v3(
                model_xgb, model_kmeans, df_clusters,
                map_fecha_sim, map_categoria_sim
            )
            st.session_state.df_simulacion_completa = df_simulacion_completa
            st.session_state.simulacion_categoria = map_categoria_sim 
    
    if "df_simulacion_completa" in st.session_state:
        st.success(f"Simulación generada para '{st.session_state.simulacion_categoria}'. Mueve el slider para explorar.")
        
        hora_animada = st.slider("Hora de Simulación (0-47h):", 0, 47, 0, format="%d:00")
        
        df_hotspots_hora_actual = st.session_state.df_simulacion_completa[
            st.session_state.df_simulacion_completa['hora_simulacion'] == hora_animada
        ]

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
            data=df_hotspots_hora_actual, 
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

        if df_hotspots_hora_actual.empty:
            st.info(f"No se encontraron zonas críticas (>= 65%) para la hora {hora_animada}:00.")
        else:
            st.success(f"Mostrando {len(df_hotspots_hora_actual)} zonas críticas para la hora {hora_animada}:00.")
            with st.expander("Ver detalles de las zonas críticas para esta hora"):
                st.dataframe(df_hotspots_hora_actual[['probabilidad', 'calle', 'lat', 'lon']])
    
    elif "df_simulacion_completa" in st.session_state:
        st.info("La simulación no generó zonas críticas (>= 65%) para esta combinación de filtros.")
