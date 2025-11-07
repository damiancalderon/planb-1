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
import data_processing as dp  # assuming your helper file

# ---------------- CONFIG -----------------
st.set_page_config(
    page_title="CDMX Crime Intelligence Platform",
    page_icon="🗺️",
    layout="wide",
)

# Initialize session states
for key, value in {
    'search_result': None,
    'last_clicked_address': None,
    'poi_results': {},
    'messages_ollama': [{"role": "assistant", "content": "Hi! Ask something about the data."}],
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Sidebar Navigation
st.sidebar.title("📍 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "📊 Analysis", "🗺️ Map", "ℹ️ Info", "👥 Our Team"]
)

# ------------------------------------------
# LOAD DATA (shared for all tabs)
# ------------------------------------------
@st.cache_data
def load_and_clean_raw_data():
    try:
        df_raw, _, _ = dp.load_data('carpetasFGJ_acumulado_2025_01.csv')
        df_clean = dp.clean_data(df_raw.copy())
        df_clean['datetime'] = pd.to_datetime(
            df_clean['fecha_hecho'].dt.date.astype(str) + ' ' +
            df_clean['hora_hecho_dt'].astype(str),
            errors='coerce'
        )
        df_clean.dropna(subset=['datetime'], inplace=True)
        return df_clean
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_geojson(geojson_file):
    try:
        with open(geojson_file, mode="r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"GeoJSON not found at '{geojson_file}'.")
        return None

df_crime = load_and_clean_raw_data()
alcaldias_geojson = load_geojson('alcaldias.geojson')


# ==================================================
# PAGE 1: HOME
# ==================================================
if page == "🏠 Home":
    st.title("🗺️ CDMX Crime Intelligence Platform")
    st.markdown("""
    Welcome to the **Crime Intelligence Platform for Mexico City**.
    This app provides:
    - Interactive **heatmaps** of crime data  
    - Filtering by **crime type**, **time**, and **location**  
    - AI-powered **local chat (Ollama)**  
    - Points of interest & address search  

    Use the sidebar to explore each section.
    """)

# ==================================================
# PAGE 2: ANALYSIS
# ==================================================
elif page == "📊 Analysis":
    st.title("📊 Data Analysis")

    if df_crime.empty:
        st.warning("No data loaded. Please check your dataset.")
    else:
        st.write("Quick summary of the dataset:")
        st.dataframe(df_crime.head())

        # Simple statistics
        st.metric("Total Records", len(df_crime))
        st.metric("Unique Alcaldías", df_crime['alcaldia_hecho_N'].nunique())
        st.metric("Unique Crimes", df_crime['delito_N'].nunique())

        # Plot
        crimes_by_alcaldia = df_crime['alcaldia_hecho_N'].value_counts().head(10)
        st.bar_chart(crimes_by_alcaldia)

# ==================================================
# PAGE 3: MAP
# ==================================================
elif page == "🗺️ Map":
    st.title("🗺️ Interactive Crime Map")

    # Filters
    if not df_crime.empty:
        with st.sidebar.expander("Filter Options", expanded=True):
            delitos_unicos = sorted(df_crime['delito_N'].unique())
            selected_delitos = st.multiselect("Select crime types", delitos_unicos, default=delitos_unicos[:3])

            min_date, max_date = df_crime['datetime'].min().date(), df_crime['datetime'].max().date()
            selected_date_range = st.date_input("Date Range", (min_date, max_date))
            selected_time_range = st.slider("Hour Range", value=(time(0, 0), time(23, 59)))

        start_date, end_date = pd.to_datetime(selected_date_range[0]), pd.to_datetime(selected_date_range[1])
        start_time, end_time = selected_time_range
        df_filtered = df_crime[
            (df_crime['delito_N'].isin(selected_delitos)) &
            (df_crime['datetime'].dt.date >= start_date.date()) &
            (df_crime['datetime'].dt.date <= end_date.date()) &
            (df_crime['datetime'].dt.time >= start_time) &
            (df_crime['datetime'].dt.time <= end_time)
        ]

        map_center = [19.4326, -99.1332]
        m = folium.Map(location=map_center, zoom_start=11, tiles="CartoDB positron")

        if not df_filtered.empty:
            heat_data = [[row['latitud'], row['longitud']] for _, row in df_filtered.iterrows()]
            HeatMap(heat_data, radius=15, blur=10).add_to(m)

        Fullscreen(position="topleft").add_to(m)
        MeasureControl(position="bottomleft").add_to(m)
        folium.LayerControl().add_to(m)
        st_folium(m, width=1200, height=650)

    else:
        st.warning("Crime data could not be loaded.")

# ==================================================
# PAGE 4: INFO
# ==================================================
elif page == "ℹ️ Info":
    st.title("ℹ️ About this Project")
    st.markdown("""
    This platform was created to visualize and analyze crime data in **Mexico City**.  
    It integrates:
    - **Machine Learning** for text similarity search  
    - **Ollama local LLM** for context-based Q&A  
    - **Folium + Streamlit** for interactive maps  

    **Technologies used:**
    - Python (Pandas, Scikit-learn, Folium, Streamlit)  
    - Geopy, Nominatim API  
    - Ollama (local inference server)
    """)

# ==================================================
# PAGE 5: OUR TEAM
# ==================================================
elif page == "👥 Our Team":
    st.title("👥 Meet Our Team")
    st.markdown("""
    **Developers & Researchers**
    - Damián Calderón — Data & Cloud Engineering  
    - [Add more team members here]

    **Mentors / Advisors**
    - [Name], [Role or Organization]
    """)

    st.markdown("---")
    st.write("Feel free to reach out if you want to collaborate or expand the project!")

