import json
from datetime import time
from pathlib import Path
from typing import Optional

import branca.colormap as cm
import folium
import numpy as np
import pandas as pd
import streamlit as st
from folium.elements import MacroElement
from folium.plugins import (
    HeatMap,
    HeatMapWithTime,
    MarkerCluster,
    Fullscreen,
    MeasureControl,
    TimestampedGeoJson,
)
from folium.utilities import none_max, none_min
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from jinja2 import Template
from streamlit_folium import st_folium

import data_processing as dp

BASE_PATH = Path(__file__).resolve().parent
ALCALDIAS_GEOJSON_PATH = BASE_PATH / "alcaldias.geojson"
TIMELINE_POINT_LIMIT = 2500


_TIMELINE_FREQ_OPTIONS = {
    "Hourly": {"freq": "H", "label_fmt": "%Y-%m-%d %H:%M", "period": "PT1H"},
    "Daily": {"freq": "D", "label_fmt": "%Y-%m-%d", "period": "P1D"},
    "Weekly": {"freq": "W", "label_fmt": "Semana %W - %Y", "period": "P1W"},
    "Monthly": {"freq": "M", "label_fmt": "%b %Y", "period": "P1M"},
}


class LegendTickFormatter(MacroElement):
    """Format legend tick labels to include a 'k' suffix for thousands."""

    def __init__(self) -> None:
        super().__init__()
        self._name = "LegendTickFormatter"
        self._template = Template(
            """
            {% macro script(this, kwargs) %}
            function formatLegendTicks{{ this.get_name() }}(attempt) {
                attempt = attempt || 0;
                const legendSvg = document.getElementById('legend');
                if (!legendSvg) {
                    if (attempt > 10) {
                        return;
                    }
                    setTimeout(function () {
                        formatLegendTicks{{ this.get_name() }}(attempt + 1);
                    }, 200);
                    return;
                }
                legendSvg.querySelectorAll('.tick text').forEach(function (label) {
                    const numericValue = parseFloat(label.textContent.replace(/,/g, ''));
                    if (Number.isNaN(numericValue)) {
                        return;
                    }
                    if (Math.abs(numericValue) >= 1000) {
                        const scaled = numericValue / 1000;
                        const decimals = Number.isInteger(scaled) ? 0 : 1;
                        label.textContent = scaled.toFixed(decimals) + 'k';
                    } else if (!Number.isInteger(numericValue)) {
                        label.textContent = numericValue.toFixed(1);
                    } else {
                        label.textContent = numericValue.toString();
                    }
                });
            }
            formatLegendTicks{{ this.get_name() }}(0);
            {% endmacro %}
            """
        )


class TimelineHeatMap(HeatMapWithTime):
    """Patched HeatMapWithTime that can compute bounds on nested frame data."""

    def _get_self_bounds(self):
        bounds = [[None, None], [None, None]]
        for frame in self.data or []:
            for point in frame or []:
                try:
                    lat, lon = float(point[0]), float(point[1])
                except (TypeError, ValueError, IndexError):
                    continue
                bounds = [
                    [none_min(bounds[0][0], lat), none_min(bounds[0][1], lon)],
                    [none_max(bounds[1][0], lat), none_max(bounds[1][1], lon)],
                ]
        return bounds


def _init_session_state() -> None:
    """Ensure the session keys used by the map exist."""
    if 'search_result' not in st.session_state:
        st.session_state.search_result = None
    if 'last_clicked_address' not in st.session_state:
        st.session_state.last_clicked_address = None
    if 'poi_results' not in st.session_state:
        st.session_state.poi_results = {}


def _prepare_timelapse_payload(df: pd.DataFrame, freq_key: str, max_frames: int):
    """Build HeatMapWithTime payload and labels respecting a frame budget."""
    freq_meta = _TIMELINE_FREQ_OPTIONS[freq_key]
    freq = freq_meta["freq"]
    truncated = False

    df = df.copy()
    df['time_bin'] = df['datetime'].dt.floor(freq)
    df.dropna(subset=['time_bin'], inplace=True)
    if df.empty:
        return [], [], pd.DataFrame(), truncated

    unique_bins = sorted(df['time_bin'].unique())
    if not unique_bins:
        return [], [], pd.DataFrame(), truncated

    if len(unique_bins) > max_frames:
        truncated = True
        idx = np.linspace(0, len(unique_bins) - 1, max_frames, dtype=int)
        idx = sorted(set(idx))
        selected_bins = [unique_bins[i] for i in idx]
    else:
        selected_bins = unique_bins

    grouped = df.groupby('time_bin')
    heatmap_frames = []
    labels = []
    for bin_value in selected_bins:
        slice_df = grouped.get_group(bin_value)
        heatmap_frames.append(slice_df[['latitud', 'longitud']].values.tolist())
        labels.append(bin_value.strftime(freq_meta['label_fmt']))

    sliced_df = df[df['time_bin'].isin(selected_bins)].copy()
    return heatmap_frames, labels, sliced_df, truncated


def _build_timestamped_geojson(df: pd.DataFrame, max_points: int = TIMELINE_POINT_LIMIT):
    """Convert filtered crimes to a TimestampedGeoJson feature collection."""
    if df.empty:
        return None

    limited_df = df.sort_values('datetime').head(max_points)
    features = []
    for _, row in limited_df.iterrows():
        popup = (
            f"<b>Delito:</b> {row.get('delito_N', 'N/D')}<br>"
            f"<b>Alcaldía:</b> {row.get('alcaldia_hecho_N', 'N/D')}<br>"
            f"<b>Fecha:</b> {row['datetime'].strftime('%Y-%m-%d %H:%M')}"
        )
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row['longitud'], row['latitud']],
            },
            "properties": {
                "time": row['datetime'].isoformat(),
                "popup": popup,
                "icon": "circle",
                "iconstyle": {
                    "fillColor": "#ff5722",
                    "fillOpacity": 0.7,
                    "stroke": "true",
                    "radius": 6,
                    "color": "#ff9800",
                },
            },
        })

    if not features:
        return None

    return {"type": "FeatureCollection", "features": features}


@st.cache_data
def find_pois(query, viewbox, limit=20):
    try:
        geolocator = dp.get_geolocator()
        pois = geolocator.geocode(query, exactly_one=False, limit=limit, viewbox=viewbox, bounded=True)
        return pois if pois else []
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        st.error(f"POI service error: {e}")
    return []


@st.cache_data
def load_and_clean_raw_data():
    try:
        df_raw, _, _ = dp.load_data()
        if df_raw is None:
            st.error("Error: No se pudieron cargar los datos raw (carpetasFGJ_acumulado_2025_01.csv).")
            return pd.DataFrame()

        df_clean = dp.clean_data(df_raw.copy())
        df_clean['datetime'] = pd.to_datetime(
            df_clean['fecha_hecho'].dt.date.astype(str) + ' ' +
            df_clean['hora_hecho_dt'].astype(str),
            errors='coerce'
        )
        df_clean.dropna(subset=['datetime'], inplace=True)

        return df_clean

    except FileNotFoundError:
        st.error("Error: No se encontró el archivo de datos raw (data/carpetasFGJ_acumulado_2025_01.csv).")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Ocurrió un error durante la carga y limpieza de datos: {e}")
        return pd.DataFrame()


@st.cache_data
def load_geojson(geojson_file):
    try:
        with open(geojson_file, mode="r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Error: GeoJSON file not found at '{geojson_file}'. Please provide it.")
        return None


def render_interactive_map(embed: bool = False, df_crime: Optional[pd.DataFrame] = None):
    """Renderiza el mapa histórico con controles avanzados y filtros dinámicos."""
    _init_session_state()

    if not embed:
        st.title("🗺️ CDMX Crime Intelligence Platform")
        st.markdown(
            """
            Welcome to the interactive crime visualization tool for Mexico City.
            This platform allows for in-depth analysis through dynamic filtering,
            layered data visualization, and interactive map controls.
            Use the sidebar to customize your analysis, search for locations, and discover points of interest.
            """
        )

    df_crime = df_crime if df_crime is not None else load_and_clean_raw_data()
    alcaldias_geojson = load_geojson(ALCALDIAS_GEOJSON_PATH)

    st.sidebar.header("⚙️ Map Filters & Controls")

    with st.sidebar.expander("Nominatim Search & POI Finder", expanded=True):
        address_query = st.text_input("Search for Address or Landmark:", placeholder="e.g., Palacio de Bellas Artes")
        if st.button("Search", use_container_width=True):
            if address_query:
                search_tuple = dp.geocode_address(address_query)

                if search_tuple[0] is None:
                    st.session_state.search_result = None
                    st.warning("Location not found. Please try a different query.")
                else:
                    class MockLocation:
                        def __init__(self, lat, lon, address):
                            self.latitude = lat
                            self.longitude = lon
                            self.address = address

                    lat, lon = search_tuple
                    address = dp.reverse_geocode_coords(lat, lon)
                    st.session_state.search_result = MockLocation(lat, lon, address or address_query)
            else:
                st.session_state.search_result = None

        st.markdown("---")
        poi_query = st.text_input("Find Nearby Points of Interest:", placeholder="e.g., Police, Hospital, Museum")
        if st.button("Find POIs", use_container_width=True):
            if poi_query:
                cdmx_viewbox = ((19.59, -99.36), (19.12, -98.94))
                st.session_state.poi_results[poi_query] = find_pois(poi_query, viewbox=cdmx_viewbox)
                if not st.session_state.poi_results[poi_query]:
                    st.warning(f"No results found for '{poi_query}' in CDMX.")

        if st.session_state.poi_results:
            if st.button("Clear POI Layers", use_container_width=True):
                st.session_state.poi_results = {}
                st.rerun()

    with st.sidebar.expander("Filter by Crime & Time", expanded=True):
        if not df_crime.empty:
            delitos_unicos = sorted(df_crime['delito_N'].unique())
            select_all_crimes = st.checkbox("Select All Crime Types", True)
            if select_all_crimes:
                selected_delitos = st.multiselect('Filter by Crime Type:', delitos_unicos, default=delitos_unicos)
            else:
                selected_delitos = st.multiselect('Filter by Crime Type:', delitos_unicos, default=delitos_unicos[:3])

            min_date, max_date = df_crime['datetime'].min().date(), df_crime['datetime'].max().date()
            selected_date_range = st.date_input(
                "Filter by Date Range:", value=(min_date, max_date), min_value=min_date, max_value=max_date
            )

            selected_time_range = st.slider(
                "Filter by Hour of Day:", value=(time(0, 0), time(23, 59)), format="HH:mm"
            )
        else:
            st.sidebar.warning("Crime data could not be loaded. Filtering is disabled.")
            selected_delitos, selected_date_range, selected_time_range = [], [], []

    with st.sidebar.expander("Toggle Map Layers", expanded=True):
        show_alcaldias = st.toggle("Show Alcaldías Boundaries", True)
        show_heatmap = st.toggle("Show Crime Heatmap", True)
        show_markers = st.toggle("Show Individual Crime Points", True)

    with st.sidebar.expander("Animated Playback (beta)", expanded=False):
        enable_timelapse = st.checkbox(
            "Enable Playable Timeline",
            value=False,
            help="Construye una animación por periodos para reproducir la evolución temporal.",
        )
        if enable_timelapse:
            freq_labels = list(_TIMELINE_FREQ_OPTIONS.keys())
            timeline_freq_label = st.selectbox(
                "Bucket temporal",
                options=freq_labels,
                index=freq_labels.index("Daily") if "Daily" in freq_labels else 0,
            )
            timeline_max_frames = st.slider(
                "Número máximo de fotogramas",
                min_value=5,
                max_value=120,
                value=40,
                help="Controla cuántos pasos tendrá la animación para evitar mapas saturados.",
            )
        else:
            timeline_freq_label = None
            timeline_max_frames = 0

    if show_heatmap:
        with st.sidebar.expander("Customize Heatmap"):
            heatmap_radius = st.slider("Heatmap Radius", 5, 30, 15)
            heatmap_blur = st.slider("Heatmap Blur", 5, 30, 10)
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
        ]
    else:
        df_filtered = pd.DataFrame()

    map_center = [19.4326, -99.1332]
    m = folium.Map(location=map_center, zoom_start=11, tiles="CartoDB positron")

    if alcaldias_geojson and not df_filtered.empty:
        crime_counts = df_filtered['alcaldia_hecho_N'].value_counts().reset_index()
        crime_counts.columns = ['alcaldia', 'crime_count']
        max_count = crime_counts['crime_count'].max()
        colormap = cm.linear.YlOrRd_09.scale(0, max_count if max_count > 0 else 1)
        colormap.caption = 'Crime Count in Selected Period'
        colormap.width = 800  # make legend bar wider for readability
        colormap.length = 10000  # increase length so color gradations are clearer
        m.add_child(colormap)
        m.add_child(LegendTickFormatter())

        if show_alcaldias:
            folium.GeoJson(
                alcaldias_geojson,
                name='Alcaldías Crime Density',
                style_function=lambda feature: {
                    'fillColor': colormap(
                        crime_counts.loc[
                            crime_counts['alcaldia'] == dp.strip_accents_upper(feature['properties']['NOMGEO']),
                            'crime_count'
                        ].iloc[0]
                    ) if dp.strip_accents_upper(feature['properties']['NOMGEO']) in crime_counts['alcaldia'].values else 'gray',
                    'color': 'black',
                    'weight': 1.5,
                    'dashArray': '5, 5',
                    'fillOpacity': 0.6
                },
                tooltip=folium.GeoJsonTooltip(fields=['NOMGEO'], aliases=['Alcaldía:']),
                highlight_function=lambda x: {'weight': 3, 'color': 'yellow'}
            ).add_to(m)

    timeline_layers_added = False
    timeline_note = None

    if not df_filtered.empty:
        if show_heatmap:
            heat_data = [[row['latitud'], row['longitud']] for _, row in df_filtered.iterrows()]
            HeatMap(heat_data, radius=heatmap_radius, blur=heatmap_blur, name="Crime Heatmap").add_to(m)

        if show_markers:
            marker_cluster = MarkerCluster(name="Crime Incidents").add_to(m)
            for _, row in df_filtered.head(1000).iterrows():
                popup_html = f"<b>Delito:</b> {row['delito_N']}<br><b>Fecha:</b> {row['datetime'].strftime('%Y-%m-%d %H:%M')}"
                folium.Marker(
                    location=[row['latitud'], row['longitud']],
                    popup=popup_html,
                    icon=folium.Icon(color="purple", icon="info-sign")
                ).add_to(marker_cluster)

    if enable_timelapse and timeline_freq_label and not df_filtered.empty:
        freq_meta = _TIMELINE_FREQ_OPTIONS[timeline_freq_label]
        heatmap_frames, frame_labels, sliced_df, timeline_truncated = _prepare_timelapse_payload(
            df_filtered,
            timeline_freq_label,
            timeline_max_frames,
        )

        if heatmap_frames:
            TimelineHeatMap(
                data=heatmap_frames,
                index=frame_labels,
                auto_play=False,
                max_opacity=0.8,
                radius=heatmap_radius or 15,
                use_local_extrema=True,
                name="Heatmap Timeline",
                display_index=True,
            ).add_to(m)
            timeline_layers_added = True

        geojson_payload = _build_timestamped_geojson(sliced_df, TIMELINE_POINT_LIMIT)
        if geojson_payload:
            TimestampedGeoJson(
                data=geojson_payload,
                period=freq_meta['period'],
                duration=freq_meta['period'],
                transition_time=400,
                auto_play=False,
                loop=False,
                add_last_point=True,
            ).add_to(m)
            timeline_layers_added = True

        if not timeline_layers_added:
            timeline_note = "La animación no pudo generarse con los datos filtrados."
        elif timeline_truncated:
            timeline_note = "Timeline truncada para mantener el rendimiento."
    elif enable_timelapse and not df_filtered.empty:
        timeline_note = "Selecciona un bucket temporal para activar la animación."
    elif enable_timelapse and df_filtered.empty:
        timeline_note = "No hay datos filtrados para generar la animación."

    if st.session_state.search_result:
        loc = st.session_state.search_result
        folium.Marker(
            location=[loc.latitude, loc.longitude],
            popup=f"<b>Search Result:</b><br>{loc.address}",
            tooltip="Your Searched Location",
            icon=folium.Icon(color='green', icon='search')
        ).add_to(m)
        m.location = [loc.latitude, loc.longitude]
        m.zoom_start = 15

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
            poi_fg.add_to(m)

    if timeline_note:
        st.caption(f"ℹ️ {timeline_note}")

    Fullscreen(position="topleft").add_to(m)
    MeasureControl(position="bottomleft", primary_length_unit="kilometers").add_to(m)
    folium.LayerControl().add_to(m)
    m.add_child(folium.LatLngPopup())

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Crimes in Selection", f"{len(df_filtered):,}")
    with col2:
        top_crime = df_filtered['delito_N'].mode()[0] if not df_filtered.empty else "N/A"
        st.metric("Most Frequent Crime", top_crime)
    with col3:
        top_alcaldia = df_filtered['alcaldia_hecho_N'].mode()[0] if not df_filtered.empty else "N/A"
        st.metric("Busiest Alcaldía", top_alcaldia)

    map_output = st_folium(m, height=600, width='stretch')

    if map_output and map_output.get("last_clicked"):
        clicked_lat = map_output["last_clicked"]["lat"]
        clicked_lon = map_output["last_clicked"]["lng"]
        address = dp.reverse_geocode_coords(clicked_lat, clicked_lon)
        st.session_state.last_clicked_address = address or "Could not fetch address."

    if st.session_state.last_clicked_address:
        st.info(f"📍 Address of Last Clicked Point: {st.session_state.last_clicked_address}")

    st.markdown("---")
    st.header("Filtered Data Explorer")

    expected_cols = ['datetime', 'delito_N', 'alcaldia_hecho_N', 'colonia_hecho_N']
    if not df_filtered.empty:
        display_df = df_filtered.reindex(columns=expected_cols).reset_index(drop=True).head(1000)
    else:
        display_df = pd.DataFrame(columns=expected_cols)

    st.dataframe(display_df, use_container_width=True)
    st.caption(f"Showing the first 1,000 rows of {len(df_filtered):,} total records in your selection.")


if __name__ == "__main__":
    st.set_page_config(
        page_title="CDMX Crime Intelligence Platform",
        page_icon="🗺️",
        layout="wide",
    )
    render_interactive_map(embed=False)
