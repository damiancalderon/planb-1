# app.py
import streamlit as st
<<<<<<< HEAD
from modules import ui_home, ui_analysis, ui_map, ui_info
=======
from modules import ui_home, ui_analysis, ui_map, ui_info, ui_ourteam
>>>>>>> 57695c1260b2c477ebf3d336435ea8dea6a39431

# --- Configuración de página (debe ir antes de cualquier render) ---
st.set_page_config(
    page_title="CDMX Crime Intelligence Platform",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar: navegación ---
st.sidebar.title("🔎 Navegación")
page = st.sidebar.radio(
    "Ir a:",
    ["🏠 Home", "📊 Analysis", "🗺️ Map", "ℹ️ Info", "👥 Our Team"],
    index=2  # arranca en Map por defecto, cámbialo si prefieres
)

<<<<<<< HEAD
# Router
if page == "🏠 Home":
    ui_home.render()
elif page == "📊 Analysis":
    ui_analysis.render()
elif page == "🗺️ Map":
    ui_map.render()
elif page == "ℹ️ Info":
    ui_info.render()
else:
<<<<<<< HEAD
    st.title("👥 Our Team")
    st.markdown("**Project Leads:** Tú y tu bandita 🔥  \n**Contacto:** agrega tus correos y roles aquí, ca.")
=======
    ui_ourteam.render()
    
>>>>>>> 57695c1260b2c477ebf3d336435ea8dea6a39431
=======
# --- Router de páginas ---
try:
    if page == "🏠 Home":
        ui_home.render()
    elif page == "📊 Analysis":
        ui_analysis.render()
    elif page == "🗺️ Map":
        ui_map.render()
    elif page == "ℹ️ Info":
        ui_info.render()
    else:
        ui_ourteam.render()
except Exception as e:
    st.error(f"⚠️ Ocurrió un error al renderizar la página: {e}")
    st.info("Revisa que la base de datos 'cdmx_insights.db' exista y que los módulos estén actualizados.")
>>>>>>> 4b7f15c1b2340f25c6f0587369366ac1f02a9cc3
