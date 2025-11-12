import streamlit as st
from pathlib import Path
import urllib.request

def render():
    st.set_page_config(page_title="CDMX: From Incidents to Insights", layout="wide")


    # ======================
    # Estilos globales
    # ======================
    st.markdown(
        """
        <style>
        /* Ajuste del contenedor principal */
        .main > div {
            padding-top: 1.2rem;
        }
        /* Título héroe centrado */
        .hero-title {
            text-align: center;
            color: #FFFFFF;
            font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
            font-weight: 800;
            font-size: clamp(36px, 5vw, 64px);
            letter-spacing: 0.2px;
            text-shadow: 2px 3px 10px rgba(0,0,0,0.35);
            margin: 0.2rem 0 0.4rem 0;
        }
        .hero-sub {
            text-align: center;
            color: #C9D1D9;
            font-size: clamp(14px, 1.6vw, 18px);
            margin-bottom: 1.1rem;
        }
        .soft-card {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 12px;
            padding: 14px 16px;
        }
        .no-top-margin h3, .no-top-margin h2 {
            margin-top: 0.2rem;
        }
        
        /* Estilo para el objetivo */
        .objective-list {
            list-style-type: disc;
            margin-left: 20px;
            color: #C9D1D9;
            font-size: 16px;
        }

        .section-title {
            font-size: 1.5rem;
            color: #E5E7EB;
            font-weight: 600;
        }

        .section-description {
            color: #E5E7EB;
            font-size: 1rem;
            line-height: 1.6;
            margin-bottom: 1rem;
        }

        .quick-nav-list {
            list-style-type: square;
            margin-left: 20px;
            color: #E5E7EB;
            font-size: 1rem;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    # ======================
    # Hero (solo título centrado arriba)
    # ======================
    st.markdown(
        """
        <h1 class="hero-title">CDMX: From Incidents to Insights</h1>
        <div class="hero-sub">
            Discover patterns, visualize trends, and explore insights about urban safety across Mexico City.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ======================
    # Cuadrantes (2x2) debajo del título
    # ======================

    # Fila 1 (dos columnas)
    colA, colB = st.columns(2)

    # (1) Arriba-izquierda: "Did you know?"
    with colA:
        st.container()
        st.subheader("Sabías que...? 🇲🇽")
        
        # Ruta de la imagen local
        img_path_sabiasque = Path("/Users/fernandovazquezrivera/OPERACIONCONTINGENCIA/planb/dashboard/Captura de pantalla 2025-11-11 a la(s) 7.12.55 p.m..png")
        
        # Verificar si la imagen existe antes de cargarla
        if img_path_sabiasque.exists():
            st.image(img_path_sabiasque, caption="Datos interesantes sobre CDMX", use_container_width=True)
        else:
            st.warning("No se encontró la imagen local.")
        
        st.info("En la Ciudad de México se denuncian aproximadamente 26 delitos por hora, lo que equivale a alrededor de 624 delitos al día. Esta cifra incluye más de cien tipos de delitos, desde robos y fraudes hasta violencia familiar y sexual. Aunque los homicidios y robos han disminuido en los últimos años, otros delitos como violencia familiar y amenazas han aumentado, por lo que el ritmo general de denuncias se mantiene alto.**.")

    # (2) Arriba-derecha: Información de la Ciudad de México
    with colB:
        # Información sobre la CDMX (población, movilidad, etc.)
        st.markdown("### 📊 Datos sobre la Ciudad de México", help="Información relevante sobre CDMX")
        st.markdown(
            """
            La Ciudad de México, con una población de más de 9 millones de habitantes, es una de las metrópolis más grandes y densamente pobladas del mundo. 
            A diario, más de 5 millones de personas se desplazan por su sistema de transporte público, incluyendo el metro, autobuses y microbuses, 
            convirtiéndola en una de las ciudades con mayor movilidad en el planeta.
            """
        )
        
        # Ruta de la imagen local
        img_path = Path("/Users/fernandovazquezrivera/OPERACIONCONTINGENCIA/planb/dashboard/CDMX IMAGEN INICIAL.jpg")
        
        # Verificar si la imagen existe antes de cargarla
        if img_path.exists():
            st.image(img_path, caption="Palacio de Bellas Artes, tomada por: Fernanda hernandez.", use_container_width=True)
        else:
            st.warning("No se encontró la imagen local, utilizando la remota.")
            cdmx_img_url = "https://upload.wikimedia.org/wikipedia/commons/8/8b/Mexico_City_Reforma_skyline.jpg"
            st.image(cdmx_img_url, caption="Skyline — Paseo de la Reforma (CC BY-SA 4.0, Jonathan Salvador)", use_container_width=True)

    st.divider()

    # Fila 2 (dos columnas)
    colC, colD = st.columns(2)

    # (3) Abajo-izquierda: Público objetivo + Contenido
    with colC:
        st.subheader("¿Cual es el objetivo?")
        st.markdown(
            """
            El objetivo es ayudarte a sentirte más seguro transformando datos complejos en información práctica y accesible.
            
            <ul class="objective-list">
                <li>Explicaciones en lenguaje sencillo y visuales amigables.</li>
                <li>Contexto a nivel de alcaldías y estaciones para orientar las decisiones.</li>
                <li>Métodos y fuentes transparentes.</li>
                <li>Consejos prácticos vinculados a patrones en el tiempo y el lugar.</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )

    # (4) Abajo-derecha: ¿Qué podrás encontrar aquí?
    with colD:
        st.markdown("<h3 class='section-title'>¿Qué podrás encontrar aquí?</h3>", unsafe_allow_html=True)
        st.markdown(
            """
            <p class="section-description">
            - 🗺️ Mapa — Explora la densidad, los clústeres y las capas.<br>
            - 📊 Info (EDA) — Tendencias por año, mes, hora y alcaldía.<br>
            - 🤖 Modelos Predictivos — Señales de riesgo por estación/tiempo.<br>
            - 👥 Nuestro Equipo — Misión, personas y valores.
            </p>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<h4 class='section-title'>Navegación rápida</h4>", unsafe_allow_html=True)
        st.markdown(
            """
            <ul class="quick-nav-list">
                <li>El mapa utiliza `alcaldias.geojson` o `alcaldias2.geojson`.</li>
                <li>Info (EDA) utiliza `carpetasFGJ_acumulado_2025_01.csv`.</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ======================
    # Footer
    # ======================
    st.caption(
        "This platform combines machine learning, geospatial analysis, and open data to support data-driven safety strategies."
    )
    st.caption(
        "Photo: “Mexico City Reforma skyline” — Jonathan Salvador — CC BY‑SA 4.0 — via Wikimedia Commons."
    )

if __name__ == "__main__":
    render()