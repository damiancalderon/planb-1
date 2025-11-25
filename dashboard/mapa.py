import json
from datetime import time, timedelta
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
    MarkerCluster,
    Fullscreen,
    MeasureControl,
    TimestampedGeoJson,
)
from jinja2 import Template
from streamlit_folium import st_folium

import data_processing as dp

BASE_PATH = Path(__file__).resolve().parent
ALCALDIAS_GEOJSON_PATH = BASE_PATH / "alcaldias.geojson"
TIMELINE_POINT_LIMIT = 2500
SPANISH_MONTHS = [
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
]

HEATMAP_GRADIENT = [
    (0.0, "#0d47a1"),
    (0.3, "#1976d2"),
    (0.6, "#fdd835"),
    (0.8, "#fb8c00"),
    (1.0, "#c62828"),
]

HEATMAP_PRECISION_OPTIONS = {
    "Alta (≈10 m)": 4,
    "Media (≈100 m)": 3,
    "Baja (≈1 km)": 2,
}

INCIDENT_HEAT_COLOR = "#ff8f00"
HEATMAP_POINT_LIMIT = 40000


def format_spanish_datetime(value, include_time: bool = True) -> str:
    """Format a datetime-like value using Spanish month names."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""

    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return ""

    try:
        month_name = SPANISH_MONTHS[ts.month - 1]
    except IndexError:
        month_name = ""

    base = f"{ts.day:02d} {month_name} {ts.year}".strip()
    if include_time:
        return f"{base} {ts.strftime('%H:%M')}".strip()
    return base


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

class LayerLegend(MacroElement):
    """Simple HTML legend that documents active map layers."""

    def __init__(self, entries) -> None:
        super().__init__()
        self._name = "LayerLegend"
        self.entries = entries
        self._template = Template(
            """
            {% macro html(this, kwargs) %}
            <div id="layer-legend"
                 style="position: fixed; bottom: 35px; left: 35px; z-index: 9999;
                        background: rgba(18, 18, 18, 0.82); padding: 12px 16px;
                        color: #fff; font-size: 13px; border-radius: 10px;
                        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.3);">
                <div style="font-weight: 600; margin-bottom: 6px;">Código de colores</div>
                {% for entry in this.entries %}
                    <div style="display: flex; align-items: center; margin-bottom: 4px;">
                        <span style="display: inline-block; width: 30px; height: 14px;
                                     border-radius: {{ entry.border_radius }};
                                     background: {{ entry.background }};
                                     border: 1px solid rgba(255, 255, 255, 0.4);
                                     margin-right: 8px;"></span>
                        <span style="white-space: nowrap;">{{ entry.label }}</span>
                    </div>
                {% endfor %}
            </div>
            {% endmacro %}
            """
        )


def _init_session_state() -> None:
    """Ensure the session keys used by the map exist."""
    if 'search_result' not in st.session_state:
        st.session_state.search_result = None
    if 'last_clicked_address' not in st.session_state:
        st.session_state.last_clicked_address = None


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
            f"<b>Fecha:</b> {format_spanish_datetime(row['datetime'])}"
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
def load_and_clean_raw_data():
    try:
        df_raw, _, _ = dp.load_data()
        if df_raw is None:
            st.error("Error: No se pudieron cargar los datos limpios. "
                     "Verifica que exista 'cleaned_crime_data.csv' o el DuckDB 'cdmx_insights.db'.")
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
        st.error("Error: No se encontró ninguna fuente de datos (CSV o DuckDB).")
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


def _build_heatmap_points(df: pd.DataFrame, precision: int = 3):
    """Aggregate crimes into lat/lon bins so intensity reflects density."""
    if df.empty:
        return [], 0

    df_coords = df.dropna(subset=['latitud', 'longitud']).copy()
    if df_coords.empty:
        return [], 0

    if len(df_coords) > HEATMAP_POINT_LIMIT:
        df_coords = df_coords.sample(HEATMAP_POINT_LIMIT, random_state=42)

    df_coords['lat_bin'] = df_coords['latitud'].round(precision)
    df_coords['lon_bin'] = df_coords['longitud'].round(precision)

    grouped = (
        df_coords
        .groupby(['lat_bin', 'lon_bin'])
        .agg(
            weight=('latitud', 'size'),
            latitud=('latitud', 'mean'),
            longitud=('longitud', 'mean'),
        )
        .reset_index(drop=True)
    )
    grouped = grouped[grouped['weight'] > 0]
    if grouped.empty:
        return [], 0

    heat_data = grouped[['latitud', 'longitud', 'weight']].values.tolist()
    max_weight = grouped['weight'].max()
    return heat_data, max_weight


def _heatmap_gradient_css():
    colors = [color for _, color in HEATMAP_GRADIENT]
    return f"linear-gradient(90deg, {', '.join(colors)})"


def _build_layer_legend_entries(show_alcaldias: bool, show_heatmap: bool, show_markers: bool, show_incident_stains: bool):
    entries = []
    if show_alcaldias:
        entries.append({
            "label": "Densidad por alcaldía",
            "background": "linear-gradient(90deg, #ffffcc, #fd8d3c, #b10026)",
            "border_radius": "4px",
        })
    if show_heatmap:
        entries.append({
            "label": "Mapa de calor delictivo",
            "background": _heatmap_gradient_css(),
            "border_radius": "4px",
        })
    if show_incident_stains:
        entries.append({
            "label": "Manchas por incidente",
            "background": INCIDENT_HEAT_COLOR,
            "border_radius": "50%",
        })
    if show_markers:
        entries.append({
            "label": "Puntos individuales",
            "background": "#6a1b9a",
            "border_radius": "50%",
        })
    return entries


def _add_incident_heat_circles(map_obj, df: pd.DataFrame, limit: int, radius_meters: int) -> int:
    """Draw a soft heat stain per incident to highlight exact coordinates."""
    subset = df[['latitud', 'longitud']].dropna().head(limit)
    count = 0
    for _, row in subset.iterrows():
        folium.Circle(
            location=[row['latitud'], row['longitud']],
            radius=radius_meters,
            color=INCIDENT_HEAT_COLOR,
            weight=0,
            fill=True,
            fill_color=INCIDENT_HEAT_COLOR,
            fill_opacity=0.18,
        ).add_to(map_obj)
        count += 1
    return count


def render_interactive_map(embed: bool = False, df_crime: Optional[pd.DataFrame] = None):
    """Renderiza el mapa histórico con controles avanzados y filtros dinámicos."""
    _init_session_state()

    if not embed:
        st.title("🗺️ Plataforma de Inteligencia Delictiva CDMX")
        st.markdown(
            """
            Bienvenido a la herramienta interactiva de visualización delictiva para la Ciudad de México.
            Esta plataforma permite un análisis profundo con filtros dinámicos,
            visualizaciones por capas y controles interactivos del mapa.
            Usa la barra lateral para personalizar tu análisis y buscar ubicaciones de referencia.
            """
        )

    df_crime = df_crime if df_crime is not None else load_and_clean_raw_data()
    alcaldias_geojson = load_geojson(ALCALDIAS_GEOJSON_PATH)

    st.sidebar.header("⚙️ Controles y filtros del mapa")

    with st.sidebar.expander("Búsqueda Nominatim", expanded=True):
        address_query = st.text_input("Buscar dirección o punto de referencia:", placeholder="Ej.: Palacio de Bellas Artes")
        if st.button("Buscar", use_container_width=True):
            if address_query:
                search_tuple = dp.geocode_address(address_query)

                if search_tuple[0] is None:
                    st.session_state.search_result = None
                    st.warning("No se encontró la ubicación. Intenta con otra consulta.")
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

    with st.sidebar.expander("Filtrar por delito y tiempo", expanded=True):
        if not df_crime.empty:
            delitos_unicos = sorted(df_crime['delito_N'].unique())
            select_all_crimes = st.checkbox("Seleccionar todos los tipos de delito", True)
            if select_all_crimes:
                selected_delitos = st.multiselect('Filtrar por tipo de delito:', delitos_unicos, default=delitos_unicos)
            else:
                selected_delitos = st.multiselect('Filtrar por tipo de delito:', delitos_unicos, default=delitos_unicos[:3])

            min_date, max_date = df_crime['datetime'].min().date(), df_crime['datetime'].max().date()
            st.markdown("**Filtrar por rango de fechas:**")
            default_start = max(min_date, max_date - timedelta(days=90))
            col_start, col_end = st.columns(2)
            with col_start:
                start_date = st.date_input(
                    "Fecha inicial",
                    value=default_start,
                    min_value=min_date,
                    max_value=max_date,
                    format="DD-MM-YYYY",
                    key="date_range_start",
                )
            with col_end:
                end_date = st.date_input(
                    "Fecha final",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                    format="DD-MM-YYYY",
                    key="date_range_end",
                )
            if end_date < start_date:
                st.info("La fecha final no puede ser anterior a la inicial; se ajustó automáticamente.")
                end_date = start_date
            selected_date_range = (start_date, end_date)
            st.caption("Por defecto se muestra un rango de hasta 90 días para acelerar la carga inicial.")

            selected_time_range = st.slider(
                "Filtrar por hora del día:", value=(time(0, 0), time(23, 59)), format="HH:mm"
            )
        else:
            st.sidebar.warning("No fue posible cargar los datos delictivos. Los filtros están deshabilitados.")
            selected_delitos, selected_date_range, selected_time_range = [], [], []

    with st.sidebar.expander("Capas del mapa", expanded=True):
        show_alcaldias = st.toggle(
            "Mostrar límites de alcaldías",
            value=True,
            key="layer_alcaldias_toggle"
        )
        show_heatmap = st.toggle(
            "Mostrar mapa de calor delictivo",
            value=True,
            key="layer_heatmap_toggle"
        )
        show_incident_stains = st.toggle(
            "Resaltar incidentes con manchas individuales",
            value=False,
            key="layer_incident_stains_toggle"
        )
        show_markers = st.toggle(
            "Mostrar puntos individuales de delitos",
            value=True,
            key="layer_markers_toggle"
        )

    if show_heatmap:
        with st.sidebar.expander("Personalizar mapa de calor", expanded=True):
            heatmap_radius = st.slider("Radio del mapa de calor", 5, 35, 18)
            heatmap_blur = st.slider("Difuminado del mapa de calor", 5, 35, 12)
            precision_options = list(HEATMAP_PRECISION_OPTIONS.keys())
            precision_default = "Media (≈100 m)" if "Media (≈100 m)" in precision_options else precision_options[0]
            precision_label = st.select_slider(
                "Resolución de densidad (agrupa incidentes cercanos)",
                options=precision_options,
                value=precision_default,
                help="Mayor resolución = celdas más pequeñas (~10 m) y manchas más precisas."
            )
            heatmap_precision = HEATMAP_PRECISION_OPTIONS.get(precision_label, 3)
    else:
        heatmap_radius = heatmap_blur = None
        heatmap_precision = 3

    if show_incident_stains:
        with st.sidebar.expander("Personalizar manchas individuales", expanded=False):
            incident_stain_limit = st.slider(
                "Máximo de manchas a dibujar",
                min_value=100,
                max_value=3000,
                value=400,
                step=100,
                help="Mantener este límite ayuda a que el mapa cargue rápido en filtros muy amplios."
            )
            incident_heat_radius = st.slider(
                "Radio de cada mancha (m)",
                min_value=10,
                max_value=250,
                value=65,
                step=5
            )
    else:
        incident_stain_limit = 0
        incident_heat_radius = 0

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
        colormap.caption = 'Conteo de delitos en el periodo seleccionado'
        colormap.width = 800  # make legend bar wider for readability
        colormap.length = 10000  # increase length so color gradations are clearer
        m.add_child(colormap)
        m.add_child(LegendTickFormatter())

        if show_alcaldias:
            folium.GeoJson(
                alcaldias_geojson,
                name='Densidad delictiva por alcaldía',
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

    if not df_filtered.empty:
        if show_heatmap:
            heat_data, heat_max = _build_heatmap_points(df_filtered, precision=heatmap_precision)
            if heat_data:
                HeatMap(
                    heat_data,
                    radius=heatmap_radius,
                    blur=heatmap_blur,
                    min_opacity=0.05,
                    max_val=heat_max if heat_max > 0 else 1,
                    gradient=dict(HEATMAP_GRADIENT),
                    name="Mapa de calor delictivo",
                ).add_to(m)
            else:
                st.sidebar.info("No hay suficientes coordenadas para construir el mapa de calor con el filtro actual.")

            if show_incident_stains and incident_heat_radius > 0:
                drawn = _add_incident_heat_circles(
                    m,
                    df_filtered,
                    limit=min(incident_stain_limit, len(df_filtered)) if incident_stain_limit > 0 else len(df_filtered),
                    radius_meters=incident_heat_radius
                )
                if drawn == 0:
                    st.sidebar.info("No hay coordenadas válidas para dibujar las manchas individuales.")

        if show_markers:
            marker_cluster = MarkerCluster(name="Incidentes delictivos").add_to(m)
            for _, row in df_filtered.head(1000).iterrows():
                popup_html = (
                    f"<b>Delito:</b> {row['delito_N']}"
                    f"<br><b>Fecha:</b> {format_spanish_datetime(row['datetime'])}"
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
        ).add_to(m)
        m.location = [loc.latitude, loc.longitude]
        m.zoom_start = 15

    Fullscreen(position="topleft").add_to(m)
    MeasureControl(position="bottomleft", primary_length_unit="kilometers").add_to(m)
    folium.LayerControl().add_to(m)
    m.add_child(folium.LatLngPopup())
    legend_entries = _build_layer_legend_entries(
        show_alcaldias,
        show_heatmap,
        show_markers,
        show_incident_stains and incident_heat_radius > 0
    )
    if legend_entries:
        m.get_root().add_child(LayerLegend(legend_entries))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Delitos totales en la selección", f"{len(df_filtered):,}")
    with col2:
        top_crime = df_filtered['delito_N'].mode()[0] if not df_filtered.empty else "N/A"
        st.metric("Delito más frecuente", top_crime)
    with col3:
        top_alcaldia = df_filtered['alcaldia_hecho_N'].mode()[0] if not df_filtered.empty else "N/A"
        st.metric("Alcaldía con más casos", top_alcaldia)

    map_output = st_folium(m, height=600, width='stretch')

    if map_output and map_output.get("last_clicked"):
        clicked_lat = map_output["last_clicked"]["lat"]
        clicked_lon = map_output["last_clicked"]["lng"]
        address = dp.reverse_geocode_coords(clicked_lat, clicked_lon)
        st.session_state.last_clicked_address = address or "No se pudo obtener la dirección."

    if st.session_state.last_clicked_address:
        st.info(f"📍 Dirección del último punto seleccionado: {st.session_state.last_clicked_address}")

    st.markdown("---")
    st.header("Explorador de datos filtrados")

    expected_cols = ['datetime', 'delito_N', 'alcaldia_hecho_N', 'colonia_hecho_N']
    if not df_filtered.empty:
        display_df = df_filtered.reindex(columns=expected_cols).reset_index(drop=True).head(1000)
    else:
        display_df = pd.DataFrame(columns=expected_cols)

    if not display_df.empty and 'datetime' in display_df:
        display_df['datetime'] = display_df['datetime'].apply(format_spanish_datetime)

    st.dataframe(display_df, use_container_width=True)
    st.caption(f"Mostrando las primeras 1,000 filas de un total de {len(df_filtered):,} registros en tu selección.")


if __name__ == "__main__":
    st.set_page_config(
        page_title="Plataforma de Inteligencia Delictiva CDMX",
        page_icon="🗺️",
        layout="wide",
    )
    render_interactive_map(embed=False)
