import streamlit as st

def render():
    st.title("🏠 Home")
    st.markdown("""
    Bienvenid@ a **CDMX Crime Intelligence Platform**.  
    - **🗺️ Map:** mapa interactivo (usa `alcaldias.geojson`/`alcaldias2.geojson` y `cleaned_crime_data.csv` si existe).  
    - **ℹ️ Info:** EDA con **`carpetasFGJ_acumulado_2025_01.csv`** tal cual.  
    """)
