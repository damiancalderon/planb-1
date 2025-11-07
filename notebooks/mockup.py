import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster, Fullscreen, MeasureControl
from streamlit_folium import st_folium
import json
import branca.colormap as cm
from datetime import datetime, time
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import sys
from pathlib import Path
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure the notebook directory is on sys.path so local helper modules can be imported
_NOTEBOOK_DIR = Path(__file__).resolve().parent
if str(_NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOK_DIR))

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

# ------------------ Ollama Chat integration ------------------
OLLAMA_CHAT_URL = "http://localhost:11434/api/generate"

@st.cache_data(show_spinner=False)
def build_corpus_vectors_from_df(_df: pd.DataFrame, _max_rows: int, _cols=None):
    # limit rows for performance
    df_sample = _df.head(_max_rows).copy()
    if _cols is None:
        # pick first 3 text/object columns
        obj_cols = [c for c, t in df_sample.dtypes.items() if t == 'object']
        _cols = obj_cols[:3] if obj_cols else list(df_sample.columns[:min(3, len(df_sample.columns))])
    text_series = df_sample[_cols].astype(str).apply(lambda r: " | ".join(r.values), axis=1)
    vec = TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(text_series.values)
    return df_sample, text_series, vec, X

def retrieve_from_vectors(query: str, vectorizer, X, k: int):
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, X).ravel()
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]

def stream_from_ollama(prompt: str, model: str, temperature: float, max_tokens: int):
    try:
        with requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            stream=True,
            timeout=0xFFFF,
        ) as r:
            r.raise_for_status()
            full = ""
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    token = data["response"]
                    full += token
                    yield token
                if data.get("done"):
                    break
    except requests.exceptions.ConnectionError:
        yield "⚠️ Cannot reach Ollama at http://localhost:11434. Is `ollama serve` running?"
    except Exception as e:
        yield f"⚠️ Error: {e}"

# Prompt template used for the local chat
SYSTEM_INSTRUCTION = (
    "You are a helpful assistant. Answer ONLY using the provided CSV CONTEXT rows. "
    "If the answer is not in the context, say you cannot find it."
)

def build_prompt(user_q: str, rows_md: str) -> str:
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"QUESTION:\n{user_q}\n\n"
        f"CONTEXT (CSV rows):\n{rows_md}\n\n"
        f"ANSWER:"
    )

if "messages_ollama" not in st.session_state:
    st.session_state.messages_ollama = [
        {"role": "assistant", "content": "Hi! Ask something about the currently selected data; I'll ground my answer on matching rows."}
    ]

with st.sidebar.expander("📚 Ollama Chat (Local)", expanded=False):
    st.header("Local CSV Chat (Ollama)")
    model = st.text_input("Ollama model", value="phi3",
                          help="Examples: llama3, llama3:8b-instruct, phi3, mistral:instruct")
    top_k = st.slider("Top-k rows as context", 1, 10, 3)
    max_rows_chat = st.number_input("Limit rows (speed)", 100, 100000, 1000, step=100)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    max_tokens = st.slider("Max new tokens", 32, 1024, 256, 32)
    if st.button("🔄 Reset chat", key="reset_ollama"):
        st.session_state.messages_ollama = [{"role": "assistant", "content": "New chat started!"}]
        st.rerun()

    for m in st.session_state.messages_ollama:
        with st.chat_message(m["role"]):
            st.markdown(m["content"]) 

    if user_q := st.chat_input("Ask a question about the filtered data…"):
        st.session_state.messages_ollama.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            with st.spinner("Searching relevant rows…"):
                source_df = df_filtered if not df_filtered.empty else df_crime
                if source_df.empty:
                    st.warning("No data available to build context. Load data first.")
                    rows_md = ""
                else:
                    df_sample, text_series, vectorizer, X = build_corpus_vectors_from_df(source_df, int(max_rows_chat))
                    idxs, scores = retrieve_from_vectors(user_q, vectorizer, X, top_k)
                    top_rows = df_sample.iloc[idxs]
                    st.caption("Top-matching rows (used as context):")
                    st.dataframe(top_rows, use_container_width=True)

                    rows_md = "\n".join(
                        f"- ROW {i}: " + " | ".join(f"{c}={str(top_rows.iloc[i][c])}" for c in top_rows.columns)
                        for i in range(len(top_rows))
                    )

            with st.spinner("Generating answer (local model)…"):
                prompt = build_prompt(user_q, rows_md)
                placeholder = st.empty()
                acc = ""
                for tok in stream_from_ollama(prompt, model, temperature, int(max_tokens)):
                    acc += tok
                    placeholder.markdown(acc)

        st.session_state.messages_ollama.append({"role": "assistant", "content": acc})


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

ALCALDIAS_GEOJSON_PATH = 'data/alcaldias.geojson' 

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
st.dataframe(
    df_filtered[['datetime', 'delito_N', 'alcaldia_hecho_N', 'colonia_hecho_N']].reset_index(drop=True).head(1000),
    use_container_width=True
)
st.caption(f"Showing the first 1,000 rows of {len(df_filtered):,} total records in your selection.")