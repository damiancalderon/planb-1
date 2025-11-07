import streamlit as st
from datetime import time
import folium
from folium.plugins import HeatMap, MarkerCluster, Fullscreen, MeasureControl
from streamlit_folium import st_folium
import branca.colormap as cm
from .data import load_map_csv, load_geojson_choice
from .helpers import _strip_accents_upper

def render():
    st.title("🗺️ CDMX Crime Intelligence Platform — Map")

    st.sidebar.header("🧭 GeoJSON")
    geo_opt = st.sidebar.radio("Capa de alcaldías:", ["alcaldias.geojson", "alcaldias2.geojson"])
    alcaldias_geojson = load_geojson_choice(geo_opt)

    df_map = load_map_csv()

    st.sidebar.header("⚙️ Filtros de Mapa")
    if df_map.empty:
        st.warning("No hay datos para el mapa.")
        return

    delitos_unicos = sorted(df_map['delito_N'].dropna().unique()) if 'delito_N' in df_map.columns else []
    all_crimes = st.sidebar.checkbox("Todos los delitos", True)
    selected_delitos = st.sidebar.multiselect(
        "Delitos", delitos_unicos, default=delitos_unicos if all_crimes else delitos_unicos[:3]
    ) if delitos_unicos else []

    if 'datetime' in df_map.columns:
        min_date, max_date = df_map['datetime'].min().date(), df_map['datetime'].max().date()
    else:
        import pandas as pd
        min_date, max_date = pd.to_datetime('2020-01-01').date(), pd.to_datetime('2020-12-31').date()

    selected_date_range = st.sidebar.date_input("Rango de fechas", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    selected_time_range = st.sidebar.slider("Hora del día", value=(time(0,0), time(23,59)), format="HH:mm")

    st.sidebar.subheader("Capas")
    show_alcaldias = st.sidebar.toggle("Límites de Alcaldías", True)
    show_heatmap  = st.sidebar.toggle("Heatmap de delitos", True)
    show_markers  = st.sidebar.toggle("Puntos individuales", False)

    if show_heatmap:
        with st.sidebar.expander("Ajustes Heatmap"):
            heatmap_radius = st.slider("Radio", 5, 30, 15)
            heatmap_blur   = st.slider("Blur", 5, 30, 10)
    else:
        heatmap_radius, heatmap_blur = 15, 10

    # Filtrado
    import pandas as pd
    if isinstance(selected_date_range, (list, tuple)) and len(selected_date_range) == 2:
        start_date, end_date = pd.to_datetime(selected_date_range[0]), pd.to_datetime(selected_date_range[1])
        start_time, end_time = selected_time_range
        mask = (
            (df_map['datetime'].dt.date >= start_date.date()) &
            (df_map['datetime'].dt.date <= end_date.date()) &
            (df_map['datetime'].dt.time >= start_time) &
            (df_map['datetime'].dt.time <= end_time)
        )
        if selected_delitos:
            mask &= df_map['delito_N'].isin(selected_delitos) if 'delito_N' in df_map.columns else False
        df_filtered = df_map[mask].copy()
    else:
        df_filtered = df_map.head(0).copy()

    # Mapa
    m = folium.Map(location=[19.4326, -99.1332], zoom_start=11, tiles="CartoDB positron")

    if alcaldias_geojson is not None and not df_filtered.empty:
        counts = df_filtered['alcaldia_hecho_N'].value_counts().reset_index()
        counts.columns = ['alcaldia', 'crime_count']
        maxc = counts['crime_count'].max()
        colormap = cm.linear.YlOrRd_09.scale(0, maxc if maxc > 0 else 1)
        colormap.caption = 'Crime Count in Selected Period'
        m.add_child(colormap)

        if show_alcaldias:
            folium.GeoJson(
                alcaldias_geojson,
                name='Alcaldías Crime Density',
                style_function=lambda feat: {
                    'fillColor': colormap(
                        counts.loc[counts['alcaldia'] == _strip_accents_upper(feat['properties'].get('NOMGEO','')), 'crime_count'].iloc[0]
                    ) if _strip_accents_upper(feat['properties'].get('NOMGEO','')) in counts['alcaldia'].values else 'gray',
                    'color': 'black', 'weight': 1.5, 'dashArray': '5, 5', 'fillOpacity': 0.6
                },
                tooltip=folium.GeoJsonTooltip(fields=['NOMGEO'], aliases=['Alcaldía:']),
                highlight_function=lambda x: {'weight': 3, 'color': 'yellow'}
            ).add_to(m)

    if not df_filtered.empty:
        if show_heatmap and df_filtered[['latitud','longitud']].notna().all(axis=1).any():
            heat_data = df_filtered[['latitud','longitud']].dropna().values.tolist()
            HeatMap(heat_data, radius=heatmap_radius, blur=heatmap_blur, name="Crime Heatmap").add_to(m)

        if show_markers and df_filtered[['latitud','longitud']].notna().all(axis=1).any():
            marker_cluster = MarkerCluster(name="Crime Incidents").add_to(m)
            for _, row in df_filtered.head(1000).iterrows():
                popup_html = f"<b>Delito:</b> {row.get('delito_N','N/A')}<br><b>Fecha:</b> {pd.to_datetime(row['datetime']).strftime('%Y-%m-%d %H:%M')}"
                folium.Marker(
                    location=[row['latitud'], row['longitud']],
                    popup=popup_html,
                    icon=folium.Icon(color="purple", icon="info-sign")
                ).add_to(marker_cluster)

    Fullscreen(position="topleft").add_to(m)
    MeasureControl(position="bottomleft", primary_length_unit="kilometers").add_to(m)
    folium.LayerControl().add_to(m)
    m.add_child(folium.LatLngPopup())

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Total Crimes in Selection", f"{len(df_filtered):,}")
    with c2:
        top_crime = df_filtered['delito_N'].mode()[0] if not df_filtered.empty and 'delito_N' in df_filtered.columns else "N/A"
        st.metric("Most Frequent Crime", top_crime)
    with c3:
        top_alc = df_filtered['alcaldia_hecho_N'].mode()[0] if not df_filtered.empty and 'alcaldia_hecho_N' in df_filtered.columns else "N/A"
        st.metric("Busiest Alcaldía", top_alc)

    st_folium(m, height=600, width='stretch')

    st.markdown("---")
    st.header("Filtered Data Explorer")
    cols_show = [c for c in ['datetime','delito_N','alcaldia_hecho_N','colonia_hecho_N'] if c in df_filtered.columns]
    st.dataframe(df_filtered[cols_show].reset_index(drop=True).head(1000), use_container_width=True)
    st.caption(f"Showing the first 1,000 rows of {len(df_filtered):,} total records in your selection.")
