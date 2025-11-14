import streamlit as st
from pathlib import Path
import urllib.request
import os  # Añadido para consistencia con Path

# 🔹 NUEVOS IMPORTS PARA INSIGHTS
import plotly.express as px
import database  # Módulo de base de datos
import requests  # Para llamar a n8n
import json

# --- Webhook n8n para Gemini Insights ---
N8N_WEBHOOK_URL_INSIGHTS = "https://n8n.tektititc.org/webhook/90408216-1fba-4806-b062-2ab8afb30fea"

# --- Función para llamar a Gemini Insights ---
@st.cache_data(ttl=3600)  # Cachea los insights por 1 hora
def call_gemini_insights(contexto_datos: str):
    """
    Envía los datos del dashboard a n8n y devuelve los insights.
    """
    if not N8N_WEBHOOK_URL_INSIGHTS.startswith("https"):
        return "Error: La URL del Webhook de Insights no está configurada."

    payload = {"contexto_datos": contexto_datos}

    try:
        response = requests.post(N8N_WEBHOOK_URL_INSIGHTS, json=payload, timeout=120)
        response.raise_for_status()
        try:
            respuesta_json = response.json()
            texto_respuesta = respuesta_json["content"]["parts"][0]["text"]
            return texto_respuesta

        except (json.JSONDecodeError, TypeError, KeyError, IndexError) as e:
            # Si falla el parseo, mostramos el texto crudo para depurar
            st.error(f"Error al procesar el JSON de Gemini: {e}")
            return f"Error de parseo. Respuesta cruda: {response.text}"

    except requests.exceptions.RequestException as e:
        return f"Error de conexión con n8n: {e}"


# --- FUNCIÓN AUXILIAR PARA RUTAS RELATIVAS ---
def get_absolute_path(relative_path: str) -> Path:
    """
    Construye una ruta absoluta basándose en la ubicación del archivo app.py
    y la carpeta 'images' en la raíz del proyecto.
    """
    # 1. Obtiene la ruta al directorio actual del módulo (modules)
    MODULE_DIR = Path(__file__).parent

    # 2. Sube un nivel para llegar a la raíz del proyecto (donde está app.py)
    ROOT_DIR = MODULE_DIR.parent

    # 3. Une la ruta raíz con la ruta relativa específica (ej: images/captura-sabiasque.png)
    return ROOT_DIR / relative_path


def render():
    st.set_page_config(page_title="CDMX: From Incidents to Insights", layout="wide")

    # ======================
    # Estilos globales (Sin cambios)
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

        # --- RUTA ADAPTADA (1) ---
        # Asume que la imagen se llama 'captura-sabiasque.png' en la carpeta 'images/'
        img_path_sabiasque = get_absolute_path("images/captura-sabiasque.png")

        # Verificar si la imagen existe antes de cargarla
        if img_path_sabiasque.exists():
            st.image(
                img_path_sabiasque,
                caption="Datos interesantes sobre CDMX",
                use_container_width=True,
            )
        else:
            st.warning(
                f"❌ Imagen no encontrada. Asegúrate de que '{img_path_sabiasque.name}' esté en la carpeta 'images' en la raíz de tu proyecto."
            )

        st.info(
            "En la Ciudad de México se denuncian aproximadamente 26 delitos por hora, lo que equivale a alrededor de 624 delitos al día. "
            "Esta cifra incluye más de cien tipos de delitos, desde robos y fraudes hasta violencia familiar y sexual. Aunque los homicidios "
            "y robos han disminuido en los últimos años, otros delitos como violencia familiar y amenazas han aumentado, por lo que el ritmo "
            "general de denuncias se mantiene alto."
        )

    # (2) Arriba-derecha: Información de la Ciudad de México
    with colB:
        # Información sobre la CDMX (población, movilidad, etc.)
        st.markdown(
            "### 📊 Datos sobre la Ciudad de México",
            help="Información relevante sobre CDMX",
        )
        st.markdown(
            """
            La Ciudad de México, con una población de más de 9 millones de habitantes, es una de las metrópolis más grandes y densamente pobladas del mundo. 
            A diario, más de 5 millones de personas se desplazan por su sistema de transporte público, incluyendo el metro, autobuses y microbuses, 
            convirtiéndola en una de las ciudades con mayor movilidad en el planeta.
            """
        )

        # --- RUTA ADAPTADA (2) ---
        # Asume que la imagen se llama 'cdmx-inicial.jpg' en la carpeta 'images/'
        img_path_cdmx = get_absolute_path("images/cdmx-inicial.jpg")

        # Verificar si la imagen existe antes de cargarla
        if img_path_cdmx.exists():
            st.image(
                img_path_cdmx,
                caption="Palacio de Bellas Artes, tomada por: Fernanda hernandez.",
                use_container_width=True,
            )
        else:
            st.warning(
                f"❌ Imagen no encontrada. Asegúrate de que '{img_path_cdmx.name}' esté en la carpeta 'images' en la raíz de tu proyecto. Usando imagen remota como fallback."
            )
            cdmx_img_url = "https://upload.wikimedia.org/wikipedia/commons/8/8b/Mexico_City_Reforma_skyline.jpg"
            st.image(
                cdmx_img_url,
                caption="Skyline — Paseo de la Reforma (CC BY-SA 4.0, Jonathan Salvador)",
                use_container_width=True,
            )

    st.divider()

    # Fila 2 (dos columnas)
    colC, colD = st.columns(2)

    # (3) Abajo-izquierda: Público objetivo + Contenido (Sin cambios)
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

    # (4) Abajo-derecha: ¿Qué podrás encontrar aquí? (Sin cambios)
    with colD:
        st.markdown(
            "<h3 class='section-title'>¿Qué podrás encontrar aquí?</h3>",
            unsafe_allow_html=True,
        )
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

        st.markdown(
            "<h4 class='section-title'>Navegación rápida</h4>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <ul class="quick-nav-list">
                <li>El mapa utiliza <code>alcaldias.geojson</code> o <code>alcaldias2.geojson</code>.</li>
                <li>Info (EDA) utiliza <code>carpetasFGJ_acumulado_2025_01.csv</code>.</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ======================
    # CDMX INSIGHTS - DASHBOARD CON GEMINI
    # ======================
    st.subheader("CDMX Insights 🚨")
    st.markdown(
        "Análisis y predicción de incidencia delictiva en la Ciudad de México, "
        "combinando datos históricos con un analista IA (Gemini)."
    )

    # --- Carga de Datos ---
    try:
        stats_df = database.get_crime_stats()
    except Exception as e:
        st.error(f"Error al cargar stats_df: {e}")
        stats_df = None

    try:
        tendency_df = database.get_historical_tendency()
    except Exception as e:
        st.error(f"Error al cargar tendency_df: {e}")
        tendency_df = None

    try:
        top_alcaldias_df = database.get_top_alcaldias()
    except Exception as e:
        st.error(f"Error al cargar top_alcaldias_df: {e}")
        top_alcaldias_df = None

    # --- KPIs ---
    promedio_diario = 0
    if stats_df is not None and not stats_df.empty and "promedio_diario" in stats_df.columns:
        promedio_diario = int(stats_df["promedio_diario"].iloc[0])

    col1, col2, col3 = st.columns(3)

    # KPI 1: Crímenes promedio diarios
    with col1:
        st.metric(label="Crímenes Promedio Diarios", value=f"{promedio_diario}")

    # KPI 2: Insights Gemini
    with col2:
        st.subheader("🤖 Insights del Analista IA (Gemini)")

        if top_alcaldias_df is not None and not top_alcaldias_df.empty:
            tabla_top = top_alcaldias_df.to_string(index=False)
        else:
            tabla_top = "Sin datos disponibles de alcaldías."

        contexto_string = f"""
        - Promedio de Crímenes Diarios: {promedio_diario}
        - Zonas (Alcaldías) más afectadas (Top 5): 
        {tabla_top}
        """

        with st.spinner("Gemini está analizando los datos..."):
            insights = call_gemini_insights(contexto_string)
            st.info(insights)

    # KPI 3: Contexto Clave
    with col3:
        st.subheader("Contexto Clave")
        st.markdown("**Zonas más afectadas (Top 5):**")
        if top_alcaldias_df is not None and not top_alcaldias_df.empty:
            for _, row in top_alcaldias_df.iterrows():
                st.markdown(f"- **{row['alcaldia_hecho']}**: {row['total']} delitos")
        else:
            st.markdown("- Datos no disponibles")

    st.divider()
    st.subheader("Tendencia General de Delitos")

    if tendency_df is not None and not tendency_df.empty:
        if "fecha" in tendency_df.columns and "total_delitos" in tendency_df.columns:
            fig = px.line(
                tendency_df,
                x="fecha",
                y="total_delitos",
                title="Tendencia histórica de delitos",
            )
            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Fecha",
                yaxis_title="Total de Delitos",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(
                "El DataFrame de tendencia no contiene las columnas 'fecha' y 'total_delitos'."
            )
    else:
        st.warning("No se pudieron cargar los datos de tendencia.")

    # Ocultar "Made with Streamlit" solo si quieres
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # ======================
    # Footer (Sin cambios)
    # ======================
    st.caption(
        "This platform combines machine learning, geospatial analysis, and open data to support data-driven safety strategies."
    )
    st.caption(
        "Photo: “Mexico City Reforma skyline” — Jonathan Salvador — CC BY-SA 4.0 — via Wikimedia Commons."
    )


if __name__ == "__main__":
    render()
