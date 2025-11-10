# modules/ui_map.py
import streamlit as st
import pandas as pd
from datetime import time
import folium
from folium.plugins import HeatMap, MarkerCluster, Fullscreen, MeasureControl
from streamlit_folium import st_folium
import branca.colormap as cm
import duckdb

# --- Config ---
DB_FILE = "cdmx_insights.db"

@st.cache_resource
def get_db_connection():
    """Crea y cachea la conexión DuckDB"""
    try:
        con = duckdb.connect(DB_FILE, read_only=True)
        return con
    except duckdb.Error as e:
        st.error(f"Error al conectar con la base de datos: {e}")
        return None

@st.cache_data
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
        df["datetime"] = pd.to_datetime(df["fecha_hecho"].astype(str) + " " + df["hora_hecho"].astype(str),
                                        errors="coerce")
        return df
    except duckdb.Error as e:
        st.error(f"Error al cargar los datos del mapa: {e}")
        return pd.DataFrame()


# --- Render del Mapa ---
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

    min_date, max_date = df_map["datetime"].min().date(), df_map["datetime"].max().date()
    selected_date_range = st.sidebar.date_input(
        "Rango de fechas",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    selected_time_range = st.sidebar.slider("Hora del día", value=(time(0, 0), time(23, 59)), format="HH:mm")

    st.sidebar.subheader("Capas")
    show_heatmap = st.sidebar.toggle("Heatmap de delitos", True)
    show_markers = st.sidebar.toggle("Puntos individuales", False)

    if show_heatmap:
        with st.sidebar.expander("Ajustes Heatmap"):
            heatmap_radius = st.slider("Radio", 5, 30, 15)
            heatmap_blur = st.slider("Blur", 5, 30, 10)
    else:
        heatmap_radius, heatmap_blur = 15, 10

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

    # --- Crear mapa base ---
    m = folium.Map(location=[19.4326, -99.1332], zoom_start=11, tiles="CartoDB positron")

    # --- Choropleth por alcaldía ---
    if not df_filtered.empty:
        counts = df_filtered["alcaldia_hecho_N"].value_counts().reset_index()
        counts.columns = ["alcaldia", "crime_count"]
        maxc = int(counts["crime_count"].max()) if not counts.empty else 1
        colormap = cm.linear.YlOrRd_09.scale(0, maxc)
        colormap.caption = "Crime Count in Selection"
        m.add_child(colormap)

    # --- Heatmap y marcadores ---
    if not df_filtered.empty:
        has_geo = df_filtered[["latitud", "longitud"]].notna().all(axis=1).any()

        if show_heatmap and has_geo:
            heat_data = df_filtered[["latitud", "longitud"]].dropna().values.tolist()
            HeatMap(heat_data, radius=heatmap_radius, blur=heatmap_blur, name="Crime Heatmap").add_to(m)

        if show_markers and has_geo:
            marker_cluster = MarkerCluster(name="Crime Incidents").add_to(m)
            for _, row in df_filtered.dropna(subset=["latitud", "longitud"]).head(1000).iterrows():
                popup_html = (
                    f"<b>Delito:</b> {row.get('delito_N','N/A')}<br>"
                    f"<b>Fecha:</b> {pd.to_datetime(row['datetime']).strftime('%Y-%m-%d %H:%M')}"
                )
                folium.Marker(
                    location=[row["latitud"], row["longitud"]],
                    popup=popup_html,
                    icon=folium.Icon(color="purple", icon="info-sign")
                ).add_to(marker_cluster)

    # --- Controles del mapa ---
    Fullscreen(position="topleft").add_to(m)
    MeasureControl(position="bottomleft", primary_length_unit="kilometers").add_to(m)
    folium.LayerControl().add_to(m)
    m.add_child(folium.LatLngPopup())

    # --- KPIs ---
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Crimes in Selection", f"{len(df_filtered):,}")
    with c2:
        top_crime = (
            df_filtered["delito_N"].mode()[0]
            if not df_filtered.empty else "N/A"
        )
        st.metric("Most Frequent Crime", top_crime)
    with c3:
        top_alc = (
            df_filtered["alcaldia_hecho_N"].mode()[0]
            if not df_filtered.empty else "N/A"
        )
        st.metric("Busiest Alcaldía", top_alc)

    # --- Render ---
    st_folium(m, height=600, width="100%")

    # --- Tabla de datos ---
    st.markdown("---")
    st.header("Filtered Data Explorer")
    cols_show = [c for c in ["datetime", "delito_N", "alcaldia_hecho_N"] if c in df_filtered.columns]
    st.dataframe(df_filtered[cols_show].reset_index(drop=True).head(1000), use_container_width=True)
    st.caption(f"Showing the first 1,000 rows of {len(df_filtered):,} total records in your selection.")
