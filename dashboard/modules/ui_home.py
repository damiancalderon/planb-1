import streamlit as st
from pathlib import Path

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
    Construye una ruta absoluta basándose en la ubicación del archivo actual
    y la carpeta raíz del proyecto.
    """
    MODULE_DIR = Path(__file__).parent
    ROOT_DIR = MODULE_DIR.parent
    return ROOT_DIR / relative_path


def render():
    st.set_page_config(page_title="CDMX: From Incidents to Insights", layout="wide")

    # ======================
    # Estilos globales combinados
    # ======================
    st.markdown(
        """
        <style>
        .main > div {
            padding-top: 1.2rem;
        }

        /* Título héroe centrado */
        .hero-title {
            text-align: center;
            color: #FFFFFF;
            font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
            font-weight: 900;
            font-size: clamp(38px, 5vw, 64px);
            letter-spacing: 0.3px;
            text-shadow: 2px 3px 10px rgba(0,0,0,0.35);
            margin: 0.2rem 0 0.4rem 0;
        }
        .hero-sub {
            text-align: center;
            color: #C9D1D9;
            font-size: clamp(14px, 1.6vw, 18px);
            margin-bottom: 1.4rem;
        }

        /* Tarjetas de métricas */
        .metric-card {
            background: radial-gradient(circle at top left, rgba(129,140,248,0.35), rgba(15,23,42,0.95));
            padding: 18px 22px;
            border-radius: 14px;
            border: 1px solid rgba(148,163,184,0.35);
            text-align: center;
            backdrop-filter: blur(6px);
            box-shadow: 0 14px 30px rgba(15,23,42,0.7);
            height: 165px;
        }

        .metric-title {
            font-size: 15px;
            color: #A5B4FC;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .metric-value {
            font-size: 48px;
            font-weight: 900;
            color: #FFFFFF;
            margin-top: 6px;
        }

        .metric-sub {
            font-size: 13px;
            color: #CBD5F5;
            margin-top: 6px;
        }

        /* 🔵 Etiquetas circulares zonas */
        .circle-badge {
            width: 180px;
            height: 180px;
            border-radius: 50%;
            background: radial-gradient(circle at 30% 30%, rgba(129,140,248,0.35), rgba(15,23,42,0.95));
            border: 1px solid rgba(148,163,184,0.35);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            box-shadow: 0 10px 25px rgba(0,0,0,0.45);
            padding: 10px;
            text-align: center;
            color: #FFFFFF;
        }

        .circle-title {
            font-size: 15px;
            font-weight: 600;
            color: #A5B4FC;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }

        .circle-value {
            font-size: 22px;
            font-weight: 800;
            color: #FFFFFF;
            line-height: 1.2;
        }

        /* Estilos intro segunda versión */
        .soft-card {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 12px;
            padding: 14px 16px;
        }
        .no-top-margin h3, .no-top-margin h2 {
            margin-top: 0.2rem;
        }
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
    # Hero
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
    # 🟦 Imagen THALES ARRIBA
    # ======================
    # Intentamos primero como ruta relativa, si no, usamos la absoluta que tenías
    img_path_thales_rel = get_absolute_path("images/Thales_Logo.png")
    img_path_thales_abs = Path(
        "/Users/fernandovazquezrivera/OPERACIONCONTINGENCIA/planb/dashboard/Thales_Logo.png"
    )

    if img_path_thales_rel.exists():
        st.image(img_path_thales_rel, use_container_width=True)
    elif img_path_thales_abs.exists():
        st.image(img_path_thales_abs, use_container_width=True)
    else:
        st.warning("No se encontró el logo de Thales (ni en images/Thales_Logo.png ni en la ruta absoluta).")

    st.divider()

   
    st.divider()

    # ======================
    # Carga de datos para KPIs / Insights
    # ======================
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

    # ======================
    # Métricas en tarjetas (mezclando fijo + dinámico)
    # ======================
    # valor por defecto si la DB falla
    default_crimes_per_day = 627
    default_crimes_per_hour = 26

    if stats_df is not None and not stats_df.empty and "promedio_diario" in stats_df.columns:
        crimes_per_day = int(stats_df["promedio_diario"].iloc[0])
        crimes_per_hour = round(crimes_per_day / 24)
    else:
        crimes_per_day = default_crimes_per_day
        crimes_per_hour = default_crimes_per_hour

    top_crime = "Robo a transeúnte en vía pública"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Crímenes promedio por día</div>
                <div class="metric-value">{crimes_per_day}</div>
                <div class="metric-sub">carpetas registradas en CDMX</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Crímenes promedio por hora</div>
                <div class="metric-value">{crimes_per_hour}</div>
                <div class="metric-sub">denuncias cada 60 minutos aprox.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Delito más común</div>
                <div class="metric-value" style="font-size:26px; line-height:1.25;">
                    {top_crime}
                </div>
                <div class="metric-sub">según registros acumulados</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ======================
    # 🟣 SECCIÓN — ZONAS MÁS AFECTADAS (círculos)
    # ======================
    st.markdown(
        "<h2 style='text-align:center; color:#FFFFFF; margin-top:10px;'> Zonas más afectadas</h2>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; color:#C9D1D9; margin-bottom:20px;'>Top 5 alcaldías con más delitos acumulados</p>",
        unsafe_allow_html=True
    )

    # Puedes dejar estos valores fijos o luego adaptarlos a top_alcaldias_df
    zonas = [
        ("CUAUHTÉMOC", "291K delitos"),
        ("IZTAPALAPA", "285K delitos"),
        ("GUSTAVO A. MADERO", "198K delitos"),
        ("BENITO JUÁREZ", "158k delitos"),
        ("COYOACÁN", "136k delitos"),
    ]

    colA2, colB2, colC2, colD2, colE2 = st.columns(5)

    for col, (zona, cant) in zip([colA2, colB2, colC2, colD2, colE2], zonas):
        with col:
            st.markdown(
                f"""
                <div class="circle-badge">
                    <div class="circle-title">{zona}</div>
                    <div class="circle-value">{cant}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # ======================
    # CDMX INSIGHTS - DASHBOARD CON GEMINI
    # ======================
    st.markdown(
        "Análisis y predicción de incidencia delictiva en la Ciudad de México, "
        "combinando datos históricos con un analista IA (Gemini)."
    )

    colI1, colI2, colI3 = st.columns(3)

    promedio_diario = crimes_per_day


    # KPI 2: Insights Gemini
    with colI2:
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

    # Ocultar "Made with Streamlit"
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # ======================
    # Footer Thales (texto bonito)
    # ======================
    st.markdown(
        """
        <hr style="border:none;border-top:1px solid rgba(255,255,255,0.12);margin:28px 0;">
        <div style="text-align:center;padding:16px 8px;color:#E5E7EB;">
            <h3 style="color:#FFFFFF;margin:0 0 6px 0;font-size:1.25rem;">🛰️ Acerca de Thales</h3>
            <p style="margin:0 auto 12px auto;max-width:900px;line-height:1.6;color:#C9D1D9;">
                Thales es una empresa global líder en tecnología que desarrolla soluciones en seguridad, transporte, defensa y espacio.
                En México impulsa proyectos de innovación, análisis de datos y seguridad urbana para construir ciudades más inteligentes y seguras.
            </p>
            <h3 style="color:#FFFFFF;margin:8px 0 6px 0;font-size:1.15rem;">🤝 En colaboración con Thales México</h3>
            <p style="margin:0 auto 8px auto;max-width:900px;line-height:1.6;color:#C9D1D9;">
                Este dashboard fue desarrollado por el <b>equipo de estudiantes del Tec de Monterrey Campus Ciudad de México</b> en conjunto con <b>Thales México</b>,
                integrando IA y analítica de datos para apoyar la toma de decisiones en seguridad ciudadana.
                <i> Powered by Python, Streamlit & DuckDB.</i>
            </p>
            <p style="margin-top:6px;">
                🌐 <a href="https://www.thalesgroup.com" target="_blank" style="color:#93C5FD;text-decoration:none;">www.thalesgroup.com</a>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Créditos finales
    st.caption("This platform combines machine learning, geospatial analysis, and open data to support data-driven safety strategies.")


if __name__ == "__main__":
    render()
