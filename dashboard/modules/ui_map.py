# modules/ui_map.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import database
import json
from datetime import datetime, time
from pathlib import Path

# 🌍 Folium
import folium
from folium.plugins import (
    HeatMap,
    MarkerCluster,
    Fullscreen,
    MeasureControl,
)
from folium import GeoJson, GeoJsonTooltip
from streamlit_folium import st_folium

# Geo
import branca.colormap as cm
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

import data_processing as dp

# =========================
# RUTAS GLOBALES (ajusta si hace falta)
# =========================
BASE_PATH = Path(__file__).parent.parent
ALCALDIAS_GEOJSON_PATH = BASE_PATH / "alcaldias.geojson"


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


@st.cache_data
def find_pois(query, viewbox, limit=20):
    """Busca puntos de interés en un radio acotado usando Nominatim, cacheado para reducir latencias."""
    try:
        geolocator = dp.get_geolocator()
        pois = geolocator.geocode(
            query,
            exactly_one=False,
            limit=limit,
            viewbox=viewbox,
            bounded=True
        )
        return pois if pois else []
    except (GeocoderTimedOut, GeocoderServiceError) as exc:
        st.error(f"POI service error: {exc}")
    return []


@st.cache_data
def load_and_clean_raw_data():
    """Carga la data cruda y la limpia para habilitar filtros históricos avanzados."""
    try:
        df_raw, _, _ = dp.load_data()
        if df_raw is None:
            st.error("No se pudieron cargar los datos base para el mapa histórico.")
            return pd.DataFrame()

        df_clean = dp.clean_data(df_raw.copy())
        if 'fecha_hecho' not in df_clean.columns or 'hora_hecho_dt' not in df_clean.columns:
            st.error("Los datos cargados no contienen 'fecha_hecho' u 'hora_hecho_dt'.")
            return pd.DataFrame()

        df_clean['datetime'] = pd.to_datetime(
            df_clean['fecha_hecho'].dt.date.astype(str) + ' ' + df_clean['hora_hecho_dt'].astype(str),
            errors='coerce'
        )
        df_clean.dropna(subset=['datetime'], inplace=True)
        return df_clean

    except FileNotFoundError:
        st.error("No se encontró el archivo de datos requerido para el mapa histórico.")
        return pd.DataFrame()
    except Exception as exc:
        st.error(f"Ocurrió un error durante la carga de datos del mapa histórico: {exc}")
        return pd.DataFrame()


@st.cache_data
def load_geojson(geojson_file: Path):
    """Carga un archivo GeoJSON desde disco, devolviendo None si no existe."""
    try:
        with open(geojson_file, mode="r", encoding="utf-8") as handler:
            return json.load(handler)
    except FileNotFoundError:
        st.error(f"No se encontró el GeoJSON en {geojson_file}.")
        return None
    except Exception as exc:
        st.error(f"No se pudo cargar el GeoJSON ({geojson_file}): {exc}")
        return None


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


# --- Helpers para detectar nombres de columnas dinámicamente ---
def detect_column(df, candidates):
    """
    Devuelve el primer nombre de columna que exista en el DataFrame
    de una lista de candidatos. Si ninguno existe, devuelve None.
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_text(val: object) -> str:
    """
    Limpia valores para tooltip/popup:
    - None / NaN / 'nan' -> ''
    - Otro -> str(val).strip()
    """
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except Exception:
        pass
    s = str(val).strip()
    if s.upper() == "NAN":
        return ""
    return s


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

    # --- 3. MAPA 1: Histórico ---
    st.header("Mapa Histórico de Incidencia (Exploración Avanzada)")
    st.markdown(
        "Explora los delitos registrados en carpetas de investigación con filtros de fecha, hora, tipo de delito, "
        "búsqueda de direcciones e integración de puntos de interés."
    )

    if not df_categorias.empty:
        df_categorias = df_categorias.copy()
        df_categorias['categoria_delito'] = df_categorias['categoria_delito'].astype(str).str.upper()
        all_categories = df_categorias['categoria_delito'].tolist()
    else:
        all_categories = []

    st.session_state.setdefault('search_result', None)
    st.session_state.setdefault('last_clicked_address', None)
    st.session_state.setdefault('poi_results', {})

    df_crime = load_and_clean_raw_data()
    alcaldias_geojson = load_geojson(ALCALDIAS_GEOJSON_PATH)

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Controles del Mapa Histórico")

    with st.sidebar.expander("Búsqueda de direcciones y POI", expanded=True):
        address_query = st.text_input(
            "Buscar dirección o lugar emblemático:",
            placeholder="Ej. Palacio de Bellas Artes"
        )
        if st.button("Buscar ubicación", key="hist_search_btn", use_container_width=True):
            if address_query:
                lat, lon = dp.geocode_address(address_query)
                if lat is None or lon is None:
                    st.session_state.search_result = None
                    st.warning("No se encontró la ubicación. Intenta con otro término.")
                else:
                    class _MockLocation:
                        def __init__(self, lat_val, lon_val, addr):
                            self.latitude = lat_val
                            self.longitude = lon_val
                            self.address = addr

                    resolved_address = dp.reverse_geocode_coords(lat, lon) or address_query
                    st.session_state.search_result = _MockLocation(lat, lon, resolved_address)
            else:
                st.session_state.search_result = None

        st.markdown("---")
        poi_query = st.text_input("Buscar puntos de interés cercanos:", placeholder="Ej. Policía, Hospital")
        if st.button("Buscar POIs", key="hist_poi_btn", use_container_width=True):
            if poi_query:
                cdmx_viewbox = ((19.59, -99.36), (19.12, -98.94))
                st.session_state.poi_results[poi_query] = find_pois(poi_query, viewbox=cdmx_viewbox)
                if not st.session_state.poi_results[poi_query]:
                    st.warning(f"No se encontraron POIs para '{poi_query}'.")

        if st.session_state.poi_results and st.button("Limpiar capas de POIs", use_container_width=True):
            st.session_state.poi_results = {}
            st.rerun()

    with st.sidebar.expander("Filtrar por delito y tiempo", expanded=True):
        if not df_crime.empty:
            delitos_unicos = sorted(df_crime['delito_N'].unique())
            select_all_crimes = st.checkbox("Seleccionar todos los delitos", True)
            if select_all_crimes:
                selected_delitos = st.multiselect(
                    "Tipos de crimen:",
                    delitos_unicos,
                    default=delitos_unicos
                )
            else:
                default_subset = delitos_unicos[:3] if len(delitos_unicos) >= 3 else delitos_unicos
                selected_delitos = st.multiselect(
                    "Tipos de crimen:",
                    delitos_unicos,
                    default=default_subset
                )

            min_date = df_crime['datetime'].min().date()
            max_date = df_crime['datetime'].max().date()
            selected_date_range = st.date_input(
                "Rango de fechas:",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )

            selected_time_range = st.slider(
                "Rango horario:",
                value=(time(0, 0), time(23, 59)),
                format="HH:mm"
            )
        else:
            st.sidebar.warning("No fue posible cargar los datos históricos.")
            selected_delitos = []
            selected_date_range = []
            selected_time_range = (time(0, 0), time(23, 59))

    with st.sidebar.expander("Capas del mapa", expanded=True):
        show_alcaldias = st.toggle("Mostrar límites de alcaldías", True)
        show_heatmap = st.toggle("Mostrar heatmap delictivo", True)
        show_markers = st.toggle("Mostrar puntos individuales", False)

    if show_heatmap:
        with st.sidebar.expander("Ajustes del heatmap"):
            heatmap_radius = st.slider("Radio del heatmap", 5, 30, 15)
            heatmap_blur = st.slider("Desenfoque del heatmap", 5, 30, 10)
    else:
        heatmap_radius = heatmap_blur = None

    if not df_crime.empty and len(selected_date_range) == 2:
        start_date, end_date = pd.to_datetime(selected_date_range[0]), pd.to_datetime(selected_date_range[1])
        start_time, end_time = selected_time_range
        df_filtered = df_crime[
            (df_crime['delito_N'].isin(selected_delitos)) &
            (df_crime['datetime'].dt.date >= start_date.date()) &
            (df_crime['datetime'].dt.date <= end_date.date()) &
            (df_crime['datetime'].dt.time >= start_time) &
            (df_crime['datetime'].dt.time <= end_time)
        ].copy()
    else:
        df_filtered = pd.DataFrame()

    map_center = [CENTER_LAT, CENTER_LON]
    m_hist = folium.Map(location=map_center, zoom_start=11, tiles="CartoDB positron")

    if alcaldias_geojson and not df_filtered.empty and show_alcaldias:
        crime_counts = df_filtered['alcaldia_hecho_N'].value_counts().reset_index()
        crime_counts.columns = ['alcaldia', 'crime_count']
        max_count = crime_counts['crime_count'].max()
        colormap = cm.linear.YlOrRd_09.scale(0, max_count if max_count > 0 else 1)
        colormap.caption = 'Delitos en el rango seleccionado'
        m_hist.add_child(colormap)

        def _style_alcaldias(feature):
            aln = dp.strip_accents_upper(feature['properties']['NOMGEO'])
            if aln in crime_counts['alcaldia'].values:
                value = crime_counts.loc[
                    crime_counts['alcaldia'] == aln, 'crime_count'
                ].iloc[0]
                fill_color = colormap(value)
            else:
                fill_color = 'gray'
            return {
                'fillColor': fill_color,
                'color': 'black',
                'weight': 1.2,
                'dashArray': '5, 5',
                'fillOpacity': 0.6
            }

        folium.GeoJson(
            alcaldias_geojson,
            name='Alcaldías (choropleth)',
            style_function=_style_alcaldias,
            tooltip=folium.GeoJsonTooltip(fields=['NOMGEO'], aliases=['Alcaldía:']),
            highlight_function=lambda _: {'weight': 3, 'color': 'yellow'}
        ).add_to(m_hist)

    if not df_filtered.empty:
        if show_heatmap:
            heat_data = [[row['latitud'], row['longitud']] for _, row in df_filtered.iterrows()]
            HeatMap(
                heat_data,
                radius=heatmap_radius or 15,
                blur=heatmap_blur or 10,
                name="Heatmap delictivo"
            ).add_to(m_hist)

        if show_markers:
            marker_cluster = MarkerCluster(name="Incidentes individuales").add_to(m_hist)
            for _, row in df_filtered.head(1000).iterrows():
                popup_html = (
                    f"<b>Delito:</b> {row.get('delito_N', 'N/D')}<br>"
                    f"<b>Fecha:</b> {row['datetime'].strftime('%Y-%m-%d %H:%M')}"
                )
                folium.Marker(
                    location=[row['latitud'], row['longitud']],
                    popup=popup_html,
                    icon=folium.Icon(color="purple", icon="info-sign")
                ).add_to(marker_cluster)

    if st.session_state.search_result:
        loc = st.session_state.search_result
        folium.Marker(
            location=[loc.latitude, loc.longitude],
            popup=f"<b>Resultado de búsqueda:</b><br>{loc.address}",
            tooltip="Ubicación buscada",
            icon=folium.Icon(color='green', icon='search')
        ).add_to(m_hist)
        m_hist.location = [loc.latitude, loc.longitude]
        m_hist.zoom_start = 15

    if st.session_state.poi_results:
        for query, pois in st.session_state.poi_results.items():
            poi_fg = folium.FeatureGroup(name=f"POIs: {query.title()}", show=True)
            for poi in pois:
                folium.Marker(
                    location=[poi.latitude, poi.longitude],
                    popup=f"<b>{poi.raw.get('type', 'poi').title()}:</b><br>{poi.address}",
                    tooltip=poi.address.split(',')[0],
                    icon=folium.Icon(color='blue', icon='star')
                ).add_to(poi_fg)
            poi_fg.add_to(m_hist)

    Fullscreen(position="topleft").add_to(m_hist)
    MeasureControl(position="bottomleft", primary_length_unit="kilometers").add_to(m_hist)
    folium.LayerControl(collapsed=False).add_to(m_hist)
    m_hist.add_child(folium.LatLngPopup())

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Incidentes en selección", f"{len(df_filtered):,}")
    with col2:
        top_crime = df_filtered['delito_N'].mode()[0] if not df_filtered.empty else "N/D"
        st.metric("Delito más frecuente", top_crime)
    with col3:
        top_alcaldia = df_filtered['alcaldia_hecho_N'].mode()[0] if not df_filtered.empty else "N/D"
        st.metric("Alcaldía con más incidentes", top_alcaldia)

    # Explicit key avoids st_folium reusing the same component when multiple maps share the same layout
    map_output = st_folium(
        m_hist,
        height=600,
        use_container_width=True,
        key="hist_map_component"
    )

    if map_output and map_output.get("last_clicked"):
        clicked_lat = map_output["last_clicked"]["lat"]
        clicked_lon = map_output["last_clicked"]["lng"]
        address = dp.reverse_geocode_coords(clicked_lat, clicked_lon)
        st.session_state.last_clicked_address = address or "No se pudo obtener la dirección."

    if st.session_state.last_clicked_address:
        st.info(f"📍 Dirección del último clic: {st.session_state.last_clicked_address}")

    st.subheader("🧾 Detalle de incidentes filtrados")
    if not df_filtered.empty:
        st.caption(f"Incidentes encontrados: **{len(df_filtered):,}** (máximo 1,000 filas en la tabla)")
        expected_cols = ['datetime', 'delito_N', 'alcaldia_hecho_N', 'colonia_hecho_N']
        display_df = df_filtered.reindex(columns=expected_cols).reset_index(drop=True).head(1000)
        with st.expander("Ver registros filtrados"):
            st.dataframe(display_df, use_container_width=True)
    else:
        st.caption("No hay incidentes para mostrar con los filtros actuales.")

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
    st_folium(
        m_pred,
        height=600,
        use_container_width=True,
        key="pred_map_component"
    )
