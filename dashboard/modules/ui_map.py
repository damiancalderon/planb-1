# modules/ui_map.py
import streamlit as st
import pandas as pd
from datetime import time
import folium
from folium.plugins import (
    HeatMap, HeatMapWithTime, MarkerCluster, Fullscreen, MeasureControl,
    MiniMap, MousePosition, LocateControl, TimestampedGeoJson
)
from streamlit_folium import st_folium
import branca.colormap as cm
import duckdb
import json

# --- Config ---
DB_FILE = "cdmx_insights.db"

# =================== Helpers ===================
def _tod_bucket(dt):
    """Clasifica hora en franja: mañana/tarde/noche."""
    h = dt.hour
    if 6 <= h < 12:
        return "Mañana (06–12)"
    elif 12 <= h < 18:
        return "Tarde (12–18)"
    else:
        return "Noche (18–06)"

def _build_heat_time_frames_by_hour(df):
    """
    Crea frames por hora (0–23) para HeatMapWithTime: lista de frames,
    cada frame es [[lat, lon, peso], ...]. Peso=1 por default.
    """
    if df.empty:
        return []
    use = df.dropna(subset=["latitud", "longitud", "datetime"]).copy()
    use["hour"] = use["datetime"].dt.hour
    frames = []
    for h in range(24):
        chunk = use[use["hour"] == h]
        frames.append(chunk[["latitud", "longitud"]].values.tolist())
    return frames

def _build_timestamped_geojson_points(df):
    """
    Construye un FeatureCollection para TimestampedGeoJson animando puntos por hora.
    Usa 'times' en formato ISO (una lista de un solo timestamp) por cada punto.
    """
    features = []
    use = df.dropna(subset=["latitud", "longitud", "datetime"]).copy()
    use["iso"] = use["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    for _, r in use.iterrows():
        feat = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(r["longitud"]), float(r["latitud"])]},
            "properties": {
                "time": r["iso"],
                "style": {"color": "#7e22ce", "fillColor": "#7e22ce"},
                "icon": "circle",
                "popup": f"<b>Delito:</b> {r.get('delito_N','N/A')}<br>"
                         f"<b>Alcaldía:</b> {r.get('alcaldia_hecho_N','N/A')}<br>"
                         f"<b>Fecha:</b> {pd.to_datetime(r['datetime']).strftime('%Y-%m-%d %H:%M')}"
            }
        }
        features.append(feat)
    return {"type": "FeatureCollection", "features": features}

# =================== Data Layer ===================
@st.cache_resource
def get_db_connection():
    """Crea y cachea la conexión DuckDB"""
    try:
        con = duckdb.connect(DB_FILE, read_only=True)
        return con
    except duckdb.Error as e:
        st.error(f"Error al conectar con la base de datos: {e}")
        return None

@st.cache_data(show_spinner=False)
def load_map_data(year_min=2016, limit=None):
    """Carga datos básicos para el mapa (lat, lon, delito, alcaldía, fecha, hora)."""
    con = get_db_connection()
    if not con:
        return pd.DataFrame()

    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
        SELECT 
            latitud,
            longitud,
            categoria_delito AS delito_N,
            alcaldia_hecho AS alcaldia_hecho_N,
            fecha_hecho,
            hora_hecho,
            anio_hecho
        FROM crimes
        WHERE anio_hecho >= {year_min}
          AND latitud IS NOT NULL AND longitud IS NOT NULL
        {limit_clause}
    """
    try:
        df = con.execute(query).fetchdf()
        # Construye datetime robusto
        dt_str = df["fecha_hecho"].astype(str).str.strip() + " " + df["hora_hecho"].astype(str).str.strip()
        df["datetime"] = pd.to_datetime(dt_str, errors="coerce", utc=False)
        return df
    except duckdb.Error as e:
        st.error(f"Error al cargar los datos del mapa: {e}")
        return pd.DataFrame()

# =================== Render ===================
def render():
    st.title("🗺️ CDMX Crime Intelligence Platform — Map")

    # --- Filtros esenciales en sidebar (mantenemos solo lo necesario) ---
    st.sidebar.header("⚙️ Filtros")
    year_min = st.sidebar.number_input("Año mínimo", value=2016, min_value=2000, max_value=2030, step=1)
    row_limit = st.sidebar.select_slider(
        "Límite de filas (para cargar más rápido)",
        options=[50_000, 100_000, 200_000, 400_000, None],
        value=100_000
    )

    df_map = load_map_data(year_min=year_min, limit=row_limit)
    if df_map.empty:
        st.warning("No hay datos para mostrar.")
        return

    delitos_unicos = sorted(df_map["delito_N"].dropna().unique())
    all_crimes = st.sidebar.checkbox("Todos los delitos", True)
    selected_delitos = st.sidebar.multiselect(
        "Delitos",
        delitos_unicos,
        default=delitos_unicos if all_crimes else delitos_unicos[:3]
    )

    # Rango de fechas disponibles
    min_dt, max_dt = df_map["datetime"].min(), df_map["datetime"].max()
    min_date, max_date = min_dt.date(), max_dt.date()
    selected_date_range = st.sidebar.date_input(
        "Rango de fechas",
        value=(min_date, max_date),
        min_value=min_date, max_value=max_date
    )

    # IMPORTANTE: quitamos el slider de hora de la sidebar.
    # El control de tiempo ahora vive dentro del mapa (HeatMapWithTime/TimestampedGeoJson).

    # --- Filtrado (delitos + fechas) ---
    start_date, end_date = pd.to_datetime(selected_date_range[0]), pd.to_datetime(selected_date_range[1])
    df_map["datetime"] = pd.to_datetime(df_map["datetime"], errors="coerce")
    mask = (df_map["datetime"].dt.date >= start_date.date()) & (df_map["datetime"].dt.date <= end_date.date())
    if selected_delitos:
        mask &= df_map["delito_N"].isin(selected_delitos)
    df_filtered = df_map[mask].copy()

    # --- Mapa base (arranca en oscuro) ---
    m = folium.Map(location=[19.4326, -99.1332], zoom_start=11, tiles=None)

    # Base layers (cambiables dentro del mapa)
    folium.TileLayer(tiles="CartoDB dark_matter", name="🌙 Noche (oscuro)", control=True).add_to(m)
    folium.TileLayer(tiles="CartoDB positron", name="☀️ Día (claro)", control=True).add_to(m)
    # Puedes añadir más si quieres:
    # folium.TileLayer(tiles="Stamen Toner", name="Toner").add_to(m)

    # --- Leyenda/colormap simple por conteo (placeholder para futuro choropleth) ---
    if not df_filtered.empty:
        counts = df_filtered["alcaldia_hecho_N"].value_counts().reset_index()
        counts.columns = ["alcaldia", "crime_count"]
        maxc = int(counts["crime_count"].max()) if not counts.empty else 1
        colormap = cm.linear.YlOrRd_09.scale(0, maxc)
        colormap.caption = "Crime Count in Selection"
        m.add_child(colormap)

    # --- Overlay: Heatmap estático (opcional) ---
    if not df_filtered.empty:
        heat_data = df_filtered[["latitud", "longitud"]].dropna().values.tolist()
        if heat_data:
            HeatMap(
                heat_data, radius=15, blur=10,
                name="Heatmap (estático)"
            ).add_to(m)

    # --- Overlay: Heatmap con slider dentro del mapa (por hora) ---
    frames = _build_heat_time_frames_by_hour(df_filtered)
    if any(len(f) for f in frames):
        HeatMapWithTime(
            data=frames,
            radius=15,
            auto_play=False,
            max_opacity=0.85,
            use_local_extrema=False,
            name="⏱️ Heatmap temporal (slider en mapa)"
        ).add_to(m)

    # --- Overlay: Puntos animados por tiempo (slider en mapa) ---
    # (Si prefieres ver puntos en vez de calor)
    anim_points_fc = _build_timestamped_geojson_points(
        df_filtered.sample(min(len(df_filtered), 4000), random_state=42)  # cap por rendimiento
    )
    if anim_points_fc["features"]:
        TimestampedGeoJson(
            data=anim_points_fc,
            transition_time=200,
            period="PT1H",           # pasos de 1 hora
            add_last_point=True,
            auto_play=False,
            loop=False,
            time_slider_drag_update=True,
            name="⏱️ Puntos temporales (slider en mapa)"
        ).add_to(m)

    # --- Cluster de marcadores (no animado) ---
    if not df_filtered.empty:
        marker_cluster = MarkerCluster(name="Incidentes (cluster)").add_to(m)
        for _, row in df_filtered.dropna(subset=["latitud", "longitud"]).head(1000).iterrows():
            popup_html = (
                f"<b>Delito:</b> {row.get('delito_N','N/A')}<br>"
                f"<b>Alcaldía:</b> {row.get('alcaldia_hecho_N','N/A')}<br>"
                f"<b>Fecha:</b> {pd.to_datetime(row['datetime']).strftime('%Y-%m-%d %H:%M')}"
            )
            folium.Marker(
                location=[row["latitud"], row["longitud"]],
                popup=popup_html,
                icon=folium.Icon(color="purple", icon="info-sign")
            ).add_to(marker_cluster)

    # --- Controles del mapa (todo in-map) ---
    MiniMap(toggle_display=True).add_to(m)
    Fullscreen(position="topleft").add_to(m)
    LocateControl(auto_start=False, position="topleft").add_to(m)
    MousePosition(position="bottomright").add_to(m)
    MeasureControl(position="bottomleft", primary_length_unit="kilometers").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.add_child(folium.LatLngPopup())

    # --- KPIs ---
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Crimes in Selection", f"{len(df_filtered):,}")
    with c2:
        top_crime = df_filtered["delito_N"].mode()[0] if not df_filtered.empty else "N/A"
        st.metric("Most Frequent Crime", top_crime)
    with c3:
        top_alc = df_filtered["alcaldia_hecho_N"].mode()[0] if not df_filtered.empty else "N/A"
        st.metric("Busiest Alcaldía", top_alc)

    # --- Render mapa ---
    st_folium(m, height=650, width="100%")

    # --- Tabla + export ---
    st.markdown("---")
    st.header("Filtered Data Explorer")
    cols_show = [c for c in ["datetime", "delito_N", "alcaldia_hecho_N", "latitud", "longitud"] if c in df_filtered.columns]
    preview = df_filtered[cols_show].reset_index(drop=True)
    st.dataframe(preview.head(1000), use_container_width=True)
    st.caption(f"Showing the first 1,000 rows of {len(df_filtered):,} total records in your selection.")
    st.download_button(
        "⬇️ Descargar CSV filtrado",
        data=preview.to_csv(index=False).encode("utf-8"),
        file_name="crimes_filtered.csv",
        mime="text/csv"
    )
