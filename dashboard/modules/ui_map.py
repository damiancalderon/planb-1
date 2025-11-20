# modules/ui_map.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import database
import json
from datetime import datetime
from pathlib import Path

# 🌍 Folium
import folium
from folium.plugins import (
    HeatMap,
    MarkerCluster,
    HeatMapWithTime,
    Fullscreen,
    MiniMap,
    MeasureControl,
    MousePosition,
)
from folium import GeoJson, GeoJsonTooltip
from streamlit_folium import st_folium

# Geo
import geopandas as gpd
from branca.colormap import linear

# =========================
# RUTAS GLOBALES (ajusta si hace falta)
# =========================
BASE_PATH = Path(__file__).parent.parent
COLONIAS_SHP = BASE_PATH / "data" / "colonias_cdmx.shp"
METRO_CSV = BASE_PATH / "data" / "metro_stations.csv"


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

    # Definir la lista de exclusión (DELITOS DE BAJO IMPACTO)
    DEBA = [
        "DELITO DE BAJO IMPACTO",
        "LESIONES CULPOSAS",
        "DAÑO A LA PROPIEDAD",
        "ABUSO DE CONFIANZA",
        "FRAUDE",
        "ROBO DE OBJETOS",
        "ROBO A NEGOCIO SIN VIOLENCIA",
        "ROBO DE VEHÍCULO ESTACIONADO",
        "ROBO A TRANSEÚNTE SIN VIOLENCIA",
        "AMENAZAS",
        "ALLANAMIENTO",
        "HOSTIGAMIENTO",
        "ACOSO SEXUAL",
        "VIOLENCIA FAMILIAR",
        "USO DE DOCUMENTOS FALSOS",
    ]

    # --- 3. MAPA 1: Histórico ---
    st.header("Mapa Histórico de Incidencia (Filtrado)")
    st.markdown(
        "Usa los filtros de la barra lateral **exclusiva del Mapa** para explorar los datos históricos. "
        "Si no seleccionas tipos de crimen, se muestran **todos los delitos NO de Bajo Impacto**."
    )

    # Barra lateral exclusiva del mapa
    st.sidebar.markdown("---")
    st.sidebar.header("Filtros del Mapa Histórico")

    # Filtros de categorías
    if not df_categorias.empty:
        df_categorias = df_categorias.copy()
        df_categorias['categoria_delito'] = df_categorias['categoria_delito'].astype(str).str.upper()
        df_categorias_filtradas = df_categorias[~df_categorias['categoria_delito'].isin(DEBA)]

        all_categories = df_categorias_filtradas['categoria_delito'].tolist()
        default_cats = all_categories[:2] if len(all_categories) >= 2 else all_categories

        crime_type = st.sidebar.multiselect(
            "Selecciona tipo de crimen (vacío = todos, **sin Bajo Impacto**):",
            options=all_categories,
            default=default_cats,
        )
    else:
        all_categories = []
        crime_type = []

    # Filtro de hora
    hour_slider = st.sidebar.slider(
        "Selecciona hora del día:",
        min_value=0, max_value=23,
        value=(0, 23),
        format="%d:00",
        key="hist_slider"
    )

    # Filtro de clasificación
    crime_classification = st.sidebar.radio(
        "Selecciona clasificación:",
        ('Violent', 'Non-Violent', 'Ambos'),
        index=2,
        key="hist_radio"
    )

    # Toggle para mostrar puntos individuales
    show_points = st.sidebar.checkbox(
        "Mostrar puntos individuales en el mapa (tooltip con delito)",
        value=True
    )

    # --- LÓGICA DE FILTROS (llamada a la BD se mantiene igual) ---
    if all_categories:
        crime_types_query = crime_type if crime_type else all_categories

        df_mapa = database.get_filtered_map_data(
            crime_types=crime_types_query,
            hour_range=hour_slider,
            classification=crime_classification
        )
    else:
        df_mapa = pd.DataFrame({'longitud': [], 'latitud': []})

    # Normalizar nombres de columnas a minúsculas para evitar problemas de detección
    if not df_mapa.empty:
        df_mapa = df_mapa.copy()
        df_mapa.columns = [c.strip().lower() for c in df_mapa.columns]

    # Detectar nombres reales de columnas (delito, alcaldía, fecha, hora, clasificación)
    if not df_mapa.empty:
        delito_col = detect_column(df_mapa, [
            'categoria_delito', 'delito', 'tipo_delito', 'delito_comun',
            'nombre_real_de_tu_columna_de_delito'  # placeholder
        ])
        alcaldia_col = detect_column(df_mapa, [
            'alcaldia_hecho', 'alcaldia', 'municipio'
        ])
        fecha_col = detect_column(df_mapa, [
            'fecha_hecho', 'fecha', 'fecha_inicio'
        ])
        hora_col = detect_column(df_mapa, [
            'hora_hecho', 'hora', 'hora_inicio'
        ])
        clasif_col = detect_column(df_mapa, [
            'clasificacion', 'clasificacion_delito', 'tipo_violencia'
        ])
    else:
        delito_col = alcaldia_col = fecha_col = hora_col = clasif_col = None

    # FILTRO DE SEGURIDAD EN LOS DATOS CARGADOS
    if not df_mapa.empty and delito_col:
        df_mapa[delito_col] = df_mapa[delito_col].astype(str).str.upper()

        before = len(df_mapa)
        df_mapa = df_mapa[~df_mapa[delito_col].isin(DEBA)]
        after = len(df_mapa)

        if before - after > 0:
            st.sidebar.caption(
                f"⚡ Se filtraron {before - after} delitos de bajo impacto del histórico (quedan {after})."
            )

    # Recalcular detección de columnas por si hubo cambios
    if not df_mapa.empty:
        delito_col = detect_column(df_mapa, [
            'categoria_delito', 'delito', 'tipo_delito', 'delito_comun',
            'nombre_real_de_tu_columna_de_delito'
        ])
        alcaldia_col = detect_column(df_mapa, [
            'alcaldia_hecho', 'alcaldia', 'municipio'
        ])
        fecha_col = detect_column(df_mapa, [
            'fecha_hecho', 'fecha', 'fecha_inicio'
        ])
        hora_col = detect_column(df_mapa, [
            'hora_hecho', 'hora', 'hora_inicio'
        ])
        clasif_col = detect_column(df_mapa, [
            'clasificacion', 'clasificacion_delito', 'tipo_violencia'
        ])
    else:
        delito_col = alcaldia_col = fecha_col = hora_col = clasif_col = None

    # ===============================
    #  HISTÓRICO: CAPAS AVANZADAS
    # ===============================

    # GeoDataFrame de delitos
    gdf_crimes = None
    df_points = pd.DataFrame(columns=['latitud', 'longitud'])
    if not df_mapa.empty:
        if 'latitud' in df_mapa.columns and 'longitud' in df_mapa.columns:
            df_points = df_mapa.dropna(subset=['latitud', 'longitud']).copy()
            try:
                gdf_crimes = gpd.GeoDataFrame(
                    df_points,
                    geometry=gpd.points_from_xy(df_points['longitud'], df_points['latitud']),
                    crs='EPSG:4326'
                )
            except Exception:
                gdf_crimes = None

    # Colonias + choropleth de Incidents/km²
    colonias = None
    choropleth = None
    bounds = None

    if COLONIAS_SHP.exists():
        try:
            colonias_raw = gpd.read_file(COLONIAS_SHP).to_crs(4326)
            col_name_src = None
            for c in ['nomut', 'nomdt', 'nombre', 'nom_col', 'nomgeo', 'name', 'colonia']:
                if c in colonias_raw.columns:
                    col_name_src = c
                    break

            if col_name_src is None:
                colonias_raw['Colony'] = 'Colonia sin nombre'
            else:
                colonias_raw['Colony'] = (
                    colonias_raw[col_name_src]
                    .astype(str)
                    .str.strip()
                )

            colonias = colonias_raw.dissolve(by='Colony', as_index=False)
            col_m = colonias.to_crs(3857)
            colonias['Area_km²'] = (col_m.geometry.area / 1e6).values

            if gdf_crimes is not None:
                joined = gpd.sjoin(
                    gdf_crimes,
                    colonias[['Colony', 'geometry']],
                    how='inner',
                    predicate='within'
                )
                counts = joined.groupby('Colony').size().rename('Incidents').reset_index()

                choropleth = (
                    colonias[['Colony', 'geometry', 'Area_km²']]
                    .merge(counts, on='Colony', how='left')
                    .fillna({'Incidents': 0})
                )

                choropleth['Incidents_per_km²'] = (
                    choropleth['Incidents'] /
                    choropleth['Area_km²'].replace({0: np.nan})
                ).fillna(0.0)

                minx, miny, maxx, maxy = choropleth.total_bounds
                bounds = [[miny, minx], [maxy, maxx]]
        except Exception as e:
            st.warning(f"No se pudo cargar/usar el shapefile de colonias: {e}")
            colonias = None
            choropleth = None

    # Centro del mapa
    if bounds:
        center_lon = (bounds[0][1] + bounds[1][1]) / 2
        center_lat = (bounds[0][0] + bounds[1][0]) / 2
    else:
        center_lat, center_lon = CENTER_LAT, CENTER_LON

    # Crear mapa base Folium para histórico
    m_hist = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles='cartodbpositron',
        control_scale=True
    )
    if bounds:
        m_hist.fit_bounds(bounds)

    # Controles extra
    Fullscreen().add_to(m_hist)
    MiniMap(toggle_display=True).add_to(m_hist)
    MeasureControl(primary_length_unit='meters').add_to(m_hist)
    MousePosition(position='bottomright').add_to(m_hist)

    # Capa GeoJSON de alcaldías (bordes suaves)
    folium.GeoJson(
        geojson_data,
        name="Alcaldías",
        style_function=lambda feature: {
            "fillColor": "#000000",
            "color": "#FFFFFF",
            "weight": 1,
            "fillOpacity": 0.05,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["NOMGEO"],
            aliases=["Alcaldía:"],
            localize=True
        )
    ).add_to(m_hist)

    # --- Choropleth de densidad por colonia (si existe shapefile) ---
    if choropleth is not None and len(choropleth):
        vmin = float(choropleth["Incidents_per_km²"].min())
        vmax = float(choropleth["Incidents_per_km²"].max())
        if vmin == vmax:
            vmin, vmax = 0.0, max(1.0, vmax)

        cmap = linear.Purples_09.scale(vmin, vmax)
        cmap.caption = "Incidentes por km² (según filtros)"

        density_layer = folium.FeatureGroup(
            name="Densidad de delitos (Inc./km²)",
            show=True
        )

        _ch = choropleth.copy()
        _ch["val"] = _ch["Incidents_per_km²"].astype(float)

        GeoJson(
            data=_ch.to_json(),
            style_function=lambda f: {
                "fillColor": cmap(f["properties"]["val"]) if f["properties"]["val"] is not None else "#f3f0ff",
                "color": "#4b5563",
                "weight": 0.3,
                "fillOpacity": 0.85,
            },
            tooltip=GeoJsonTooltip(
                fields=["Colony", "Incidents", "Area_km²", "Incidents_per_km²"],
                aliases=["Colonia", "Incidentes", "Área (km²)", "Incidentes/km²"],
                localize=True,
                sticky=True,
            ),
            name="Densidad de delitos"
        ).add_to(density_layer)

        density_layer.add_to(m_hist)
        cmap.add_to(m_hist)

        # Capa de límites de colonias opcional
        colonies_layer = folium.FeatureGroup(
            name='Colonias (límites)',
            show=False
        )
        GeoJson(
            data=colonias[['Colony', 'geometry']].to_json(),
            style_function=lambda f: {"fillOpacity": 0.0, "color": "#7e22ce", "weight": 1.2},
            tooltip=GeoJsonTooltip(fields=['Colony'], aliases=['Colonia'])
        ).add_to(colonies_layer)
        colonies_layer.add_to(m_hist)


    # --- Heatmap estático + animado (24h si hay timestamp) ---
    if not df_points.empty:
        heat_data = df_points[['latitud', 'longitud']].values.tolist()
        if len(heat_data) > 0:
            HeatMap(
                heat_data,
                radius=12,
                blur=18,
                max_zoom=16,
                name="Heatmap (todos los incidentes filtrados)"
            ).add_to(m_hist)

        if fecha_col:
            df_time = df_points.copy()
            if hora_col:
                df_time['_ts'] = pd.to_datetime(
                    df_time[fecha_col].astype(str) + ' ' + df_time[hora_col].astype(str),
                    errors='coerce'
                )
            else:
                df_time['_ts'] = pd.to_datetime(df_time[fecha_col], errors='coerce')

            if df_time['_ts'].notna().any():
                tmax = df_time['_ts'].max()
                tmin = tmax - pd.Timedelta(hours=24)
                recent = df_time[(df_time['_ts'] >= tmin) & (df_time['_ts'] <= tmax)].copy()

                if len(recent):
                    recent['hour'] = recent['_ts'].dt.floor('H')
                    hours = sorted(recent['hour'].dropna().unique())
                    heat_seq = [
                        recent[recent['hour'] == h][['latitud', 'longitud']].values.tolist()
                        for h in hours
                    ]
                    HeatMapWithTime(
                        heat_seq,
                        index=[str(h) for h in hours],
                        auto_play=False,
                        max_opacity=0.8,
                        radius=9,
                        name="Heatmap 24h (animado)"
                    ).add_to(m_hist)

    # --- Capa de puntos individuales con info del delito (GOTAS) ---
    if not df_points.empty and show_points:
        max_points = 5000
        if len(df_points) > max_points:
            df_points = df_points.sample(max_points, random_state=42)

        marker_cluster = MarkerCluster(name="Crímenes individuales").add_to(m_hist)

        for _, row in df_points.iterrows():
            # 1️⃣ Leemos todos los campos posibles
            delito_val   = _safe_text(row.get(delito_col))   if delito_col   else ""
            alcaldia_row = _safe_text(row.get(alcaldia_col)) if alcaldia_col else ""
            hora_row     = _safe_text(row.get(hora_col))     if hora_col     else ""
            fecha_row    = _safe_text(row.get(fecha_col))    if fecha_col    else ""
            clasif_row   = _safe_text(row.get(clasif_col))   if clasif_col   else ""

            # 2️⃣ Elegimos el texto principal del delito
            if delito_val:
                delito_main = delito_val
            elif clasif_row:
                delito_main = clasif_row
            else:
                delito_main = "Incidente"

            # 3️⃣ Tooltip (hover sobre la gota)
            tooltip_parts = [delito_main]
            if alcaldia_row:
                tooltip_parts.append(alcaldia_row)
            if hora_row:
                tooltip_parts.append(hora_row)

            tooltip_text = " · ".join([p for p in tooltip_parts if p])
            if not tooltip_text:
                tooltip_text = "Incidente sin detalles de texto"

            # 4️⃣ Popup (clic sobre la gota)
            partes_popup = []
            delito_popup = delito_val or clasif_row or "Desconocido / No registrado"
            partes_popup.append(f"<b>Delito:</b> {delito_popup}")

            if alcaldia_row:
                partes_popup.append(f"<b>Alcaldía:</b> {alcaldia_row}")
            if fecha_row:
                partes_popup.append(f"<b>Fecha:</b> {fecha_row}")
            if hora_row:
                partes_popup.append(f"<b>Hora:</b> {hora_row}")
            if clasif_row:
                partes_popup.append(f"<b>Clasificación:</b> {clasif_row}")

            if not partes_popup:
                partes_popup.append("Sin información adicional disponible.")

            popup_html = "<br>".join(partes_popup)

            # 5️⃣ Marcador tipo gota (pin)
            folium.Marker(
                location=[row['latitud'], row['longitud']],
                tooltip=tooltip_text,
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color='red', icon='tint', prefix='fa')  # gota roja tipo pin
            ).add_to(marker_cluster)
    elif df_mapa.empty:
        st.info("No se encontraron incidentes para los filtros seleccionados (posiblemente solo había bajo impacto o la base de datos está vacía).")

    # LayerControl para elegir capas
    folium.LayerControl(collapsed=False).add_to(m_hist)

    # Mostrar el mapa histórico en Streamlit
    st_folium(m_hist, width="100%", height=600)

    # --- WIDGET DE DETALLE DE INCIDENTES ---
    st.subheader("🧾 Detalle de incidentes filtrados")

    if not df_mapa.empty:
        st.caption(f"Incidentes encontrados con los filtros actuales (sin bajo impacto): **{len(df_mapa):,}**")

        candidate_cols_groups = [
            ['categoria_delito', 'delito', 'tipo_delito', 'delito_comun'],
            ['alcaldia_hecho', 'alcaldia', 'municipio'],
            ['fecha_hecho', 'fecha', 'fecha_inicio'],
            ['hora_hecho', 'hora', 'hora_inicio'],
            ['clasificacion', 'clasificacion_delito', 'tipo_violencia'],
            ['latitud'],
            ['longitud'],
        ]

        cols_to_show = []
        for group in candidate_cols_groups:
            col = detect_column(df_mapa, group)
            if col and col not in cols_to_show:
                cols_to_show.append(col)

        with st.expander("Ver tabla detallada de incidentes"):
            if cols_to_show:
                max_rows = 2000
                if len(df_mapa) > max_rows:
                    st.warning(f"Se muestran solo las primeras {max_rows} filas de {len(df_mapa):,} incidentes.")
                    st.dataframe(df_mapa[cols_to_show].head(max_rows))
                else:
                    st.dataframe(df_mapa[cols_to_show])
            else:
                st.info("No se encontraron columnas estándar (fecha, hora, tipo de crimen, etc.) en el dataset.")
    else:
        st.caption("No hay incidentes para mostrar en detalle con los filtros actuales.")

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

