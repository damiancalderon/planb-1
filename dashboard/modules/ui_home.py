import streamlit as st
from pathlib import Path

def render():
    # Título grande con fuente "Impact", tamaño aumentado y color blanco
    st.markdown("""
    <h1 style='text-align: center; color: #FFFFFF; font-family: "Impact", sans-serif; font-size: 60px; text-shadow: 4px 4px 6px rgba(0, 0, 0, 0.3);'>
    CDMX: From Incidents to Insights</h1>
    """, unsafe_allow_html=True)
    
    # Descripción
    st.markdown("""
    Discover patterns, visualize trends, and explore insights about urban safety across Mexico City.
    """)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💡 Did you know?")
        st.info("Many incidents occur within short distance of **metro stations**.")

        # ===== Mexico City Card (updated image) =====
        st.markdown("### 🌆 Mexico City")

        # Cargar imagen local
        img_path = Path("/Users/damcalde/RETO/planb/dashboard/CDMX.png.avif")  # Cambia la ruta si es necesario

        st.image(img_path, caption="Skyline — Paseo de la Reforma", use_column_width=True)

    with col2:
        st.subheader("🧭 For the public")
        st.markdown("""
        Our goal is to **help you feel safer** by turning complex data into clear, actionable insights:

        - Plain-language explanations and friendly visuals.  
        - Borough- and station-level context to orient decisions.  
        - Transparent methods and sources (no black boxes).  
        - Practical tips tied to patterns in time and place.
        """)

        st.subheader("📦 What's inside")
        st.markdown("""
        - **🗺️ Map** — Explore density, clusters, and layers.  
        - **📊 Info (EDA)** — Trends by year, month, hour, and borough.  
        - **🤖 Predictive Models** — Station/time risk signals.  
        - **👥 Our Team** — Mission, people, and values.

        **Quick navigation**
        - Map uses `alcaldias.geojson` or `alcaldias2.geojson`.  
        - Info (EDA) uses `carpetasFGJ_acumulado_2025_01.csv`.
        """)

    st.divider()

    st.caption("This platform combines machine learning, geospatial analysis, and open data to support data-driven safety strategies.")
