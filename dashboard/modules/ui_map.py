# modules/ui_map.py
import streamlit as st
import pandas as pd
from datetime import time
import folium
from folium.plugins import (
    HeatMap, HeatMapWithTime, MarkerCluster, Fullscreen, MeasureControl,
    MiniMap, MousePosition, LocateControl
)
from streamlit_folium import st_folium
import branca.colormap as cm
import duckdb

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

def _build_heat_time_frames(df, max_frames=24):
    """Crea frames para HeatMapWithTime agrupando por hora (0–23)."""
    if df.empty:
        return []
    df = df.dropna(subset=["latitud", "longitud", "datetime"]).copy()
    df["hour"] = df["datetime"].dt.hour
    frames = []
    for h in range(24):
        chunk = df[df["hour"] == h]
        frames.append(chunk[["latitud", "longitud"]].values.tolist())
    return frames[:max_frames]

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
    st.sidebar.header("⚙️ Filtros de Mapa")

    # --- Filtros iniciales ---
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

    # --- Filtros de delito y tiempo ---
    delitos_unicos = sorted(df_map["delito_N"].dropna().unique())
    all_crimes = st.sidebar.checkbox("Todos los delitos", True)
    selected_delitos = st.sidebar.multiselect(
        "Delitos",
        delitos_unicos,
        default=delitos_unicos if all_crimes else delitos_unicos[:3]
    )

    # Rango de fechas/horas disponibles
    min_dt, max_dt = df_map["datetime"].min(), df_map["datetime"].max()
    min_date, max_date = min_dt.date(), max_dt.date()

    selected_date_range = st.sidebar.date_input(
        "Rango de fechas",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    selected_time_range = st.sidebar.slider("Hora del día", value=(time(0, 0), time(23, 59)), format="HH:mm")

    # --- Capas / features ---
    st.sidebar.subheader("Capas")
    show_heatmap = st.sidebar.toggle("Heatmap de delitos", True)
    show_heatmap_time = st.sidebar.toggle("Heatmap temporal (por hora)", False)
    show_markers = st.sidebar.toggle("Puntos individuales (cluster)", False)
    show_dayparts = st.sidebar.toggle("Capas por franja horaria", False)
    show_buffers = st.sidebar.toggle("Buffer alrededor del centro (demo)", False)

    # Ajustes de heatmap
    if show_heatmap or show_heatmap_time:
        with st.sidebar.expander("Ajustes Heatmap"):
            heatmap_radius = st.slider("Radio", 5, 30, 15)
            heatmap_blur = st.slider("Blur", 5, 30, 10)
    else:
        heatmap_radius, heatmap_blur = 15, 10

    # Ajustes de cluster
    if show_markers:
        with st.sidebar.expander("Ajustes Marcadores"):
            max_markers = st.number_input("Máx. marcadores (cluster)", value=1000, min_value=100, step=100)

    # Ajustes de buffers
    if show_buffers:
        with st.sidebar.expander("Ajustes Buffer"):
            buffer_radius = st.slider("Radio buffer (m)", 100, 1500, 500, step=50)
            buffer_lat = st.number_input("Latitud centro", value=19.4326, step=0.0001, format="%.6f")
            buffer_lon = st.number_input("Longitud centro", value=-99.1332, step=0.0001, format="%.6f")

    # --- Filtrado ---
    start_date, end_date = pd.to_datetime(selected_date_range[0]), pd.to_datetime(selected_date_range[1])
    start_time, end_time = selected_time_range

    df_map["datetime"] = pd.to_datetime(df_map["datetime"], errors="coerce")
    mask = (
        (df_map["datetime"].dt.date >= start_date.date()) &
        (df_map["datetime"].dt.date <= end_date.date()) &
        (df_map["datetime"].dt.time >= start_time) &
        (df_map["datetime"].dt.time <= end_time)
    )
    if selected_delitos:
        mask &= df_map["delito_N"].isin(selected_delitos)

    df_filtered = df_map[mask].copy()

    # --- Mapa base ---
    m = folium.Map(location=[19.4326, -99.1332], zoom_start=11, tiles="CartoDB positron")

    # --- Choropleth placeholder (leyenda dinámica por conteo) ---
    # Nota: aquí solo escalamos el colormap para armonizar con futuras capas poligonales
    if not df_filtered.empty:
        counts = df_filtered["alcaldia_hecho_N"].value_counts().reset_index()
        counts.columns = ["alcaldia", "crime_count"]
        maxc = int(counts["crime_count"].max()) if not counts.empty else 1
        colormap = cm.linear.YlOrRd_09.scale(0, maxc)
        colormap.caption = "Crime Count in Selection"
        m.add_child(colormap)

    # --- Heatmap simple ---
    if show_heatmap and not df_filtered.empty:
        heat_data = df_filtered[["latitud", "longitud"]].dropna().values.tolist()
        if heat_data:
            HeatMap(
                heat_data, radius=heatmap_radius, blur=heatmap_blur,
                name="Heatmap (estático)"
            ).add_to(m)

    # --- Heatmap temporal por hora ---
    if show_heatmap_time and not df_filtered.empty:
        frames = _build_heat_time_frames(df_filtered)
        if any(len(f) for f in frames):
            HeatMapWithTime(
                data=frames, radius=heatmap_radius, auto_play=False,
                max_opacity=0.85, use_local_extrema=False,
                name="Heatmap temporal (hora del día)"
            ).add_to(m)

    # --- Cluster de marcadores ---
    if show_markers and not df_filtered.empty:
        marker_cluster = MarkerCluster(name="Incidentes (cluster)").add_to(m)
        geo_df = df_filtered.dropna(subset=["latitud", "longitud"]).head(int(max_markers))
        for _, row in geo_df.iterrows():
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

    # --- Capas por franja horaria (mañana/tarde/noche) ---
    if show_dayparts and not df_filtered.empty:
        tmp = df_filtered.dropna(subset=["latitud", "longitud", "datetime"]).copy()
        tmp["franja"] = tmp["datetime"].apply(_tod_bucket)

        for franja, color in [
            ("Mañana (06–12)", "green"),
            ("Tarde (12–18)", "blue"),
            ("Noche (18–06)", "red"),
        ]:
            sub = tmp[tmp["franja"] == franja]
            if not sub.empty:
                fg = folium.FeatureGroup(name=f"Franja: {franja}")
                for _, r in sub.iterrows():
                    folium.CircleMarker(
                        location=[r["latitud"], r["longitud"]],
                        radius=3,
                        color=color,
                        fill=True,
                        fill_opacity=0.5,
                        tooltip=f"{r.get('delito_N','N/A')} • {franja}"
                    ).add_to(fg)
                fg.add_to(m)

    # --- Buffer demo (círculo alrededor del centro/latlon) ---
    if show_buffers:
        folium.Circle(
            location=[buffer_lat, buffer_lon],
            radius=buffer_radius,
            color="#2563eb",
            fill=True, fill_opacity=0.08,
            tooltip=f"Buffer {buffer_radius} m"
        ).add_to(folium.FeatureGroup(name="Buffer centro").add_to(m))

    # --- Controles del mapa ---
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
    st_folium(m, height=600, width="100%")

    # --- Export & Data table ---
    st.markdown("---")
    st.header("Filtered Data Explorer")
    cols_show = [c for c in ["datetime", "delito_N", "alcaldia_hecho_N", "latitud", "longitud"] if c in df_filtered.columns]
    preview = df_filtered[cols_show].reset_index(drop=True)
    st.dataframe(preview.head(1000), use_container_width=True)
    st.caption(f"Showing the first 1,000 rows of {len(df_filtered):,} total records in your selection.")

    csv_bytes = preview.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar CSV filtrado", data=csv_bytes, file_name="crimes_filtered.csv", mime="text/csv")
