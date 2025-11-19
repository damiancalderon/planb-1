import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster, Fullscreen, MeasureControl
from streamlit_folium import st_folium
import json
import branca.colormap as cm
from datetime import datetime, time
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import data_processing as dp 

st.set_page_config(
    page_title="CDMX Crime Intelligence Platform",
    page_icon="🗺️",
    layout="wide",
)

if 'search_result' not in st.session_state:
    st.session_state.search_result = None
if 'last_clicked_address' not in st.session_state:
    st.session_state.last_clicked_address = None
if 'poi_results' not in st.session_state:
    st.session_state.poi_results = {}

st.title("🗺️ CDMX Crime Intelligence Platform")
st.markdown("""
Welcome to the interactive crime visualization tool for Mexico City. 
This platform allows for in-depth analysis through dynamic filtering, layered data visualization, and interactive map controls.
Use the sidebar to customize your analysis, search for locations, and discover points of interest.
""")

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

ALCALDIAS_GEOJSON_PATH = '/Users/damcalde/dashboard/data/alcaldias.geojson' 

df_crime = load_and_clean_raw_data()
alcaldias_geojson = load_geojson(ALCALDIAS_GEOJSON_PATH)

st.sidebar.header("⚙️ Map Filters & Controls")

with st.sidebar.expander("Nominatim Search & POI Finder", expanded=True):
    address_query = st.text_input("Search for Address or Landmark:", placeholder="e.g., Palacio de Bellas Artes")
    if st.button("Search", width='stretch'):
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
    if st.button("Find POIs", width='stretch'):
        if poi_query:
            cdmx_viewbox = ((19.59, -99.36), (19.12, -98.94))
            st.session_state.poi_results[poi_query] = find_pois(poi_query, viewbox=cdmx_viewbox)
            if not st.session_state.poi_results[poi_query]:
                 st.warning(f"No results found for '{poi_query}' in CDMX.")
        
    if st.session_state.poi_results:
        if st.button("Clear POI Layers", width='stretch'):
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
    show_markers = st.toggle("Show Individual Crime Points", False)

if show_heatmap:
    with st.sidebar.expander("Customize Heatmap"):
        heatmap_radius = st.slider("Heatmap Radius", 5, 30, 15)
        heatmap_blur = st.slider("Heatmap Blur", 5, 30, 10)

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
    m.add_child(colormap)

    if show_alcaldias:
        folium.GeoJson(
            alcaldias_geojson,
            name='Alcaldías Crime Density',
            style_function=lambda feature: {
                'fillColor': colormap(crime_counts.loc[crime_counts['alcaldia'] == dp.strip_accents_upper(feature['properties']['NOMGEO']), 'crime_count'].iloc[0])
                if dp.strip_accents_upper(feature['properties']['NOMGEO']) in crime_counts['alcaldia'].values else 'gray',
                'color': 'black', 'weight': 1.5, 'dashArray': '5, 5', 'fillOpacity': 0.6
            },
            tooltip=folium.GeoJsonTooltip(fields=['NOMGEO'], aliases=['Alcaldía:']),
            highlight_function=lambda x: {'weight': 3, 'color': 'yellow'}
        ).add_to(m)

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

# Make final table selection robust: if df_filtered is empty or missing expected columns,
# provide an empty DataFrame with the expected columns instead of raising KeyError.
expected_cols = ['datetime', 'delito_N', 'alcaldia_hecho_N', 'colonia_hecho_N']
if not df_filtered.empty:
    # Reindex will create missing columns with NaN instead of raising
    display_df = df_filtered.reindex(columns=expected_cols).reset_index(drop=True).head(1000)
else:
    display_df = pd.DataFrame(columns=expected_cols)

st.dataframe(display_df, use_container_width=True)
st.caption(f"Showing the first 1,000 rows of {len(df_filtered):,} total records in your selection.")