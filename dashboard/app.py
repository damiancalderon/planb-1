import streamlit as st
# Importa tus módulos de renderizado, se asume que existen:
from modules import ui_home, ui_analysis, ui_map, ui_info, ui_ourteam, ui_alerts
# Si estos módulos no existen, el código fallará al intentar llamarlos.

# --- Configuración de página (debe ir antes de cualquier render) ---
st.set_page_config(
    page_title="CDMX Crime Intelligence Platform",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed" # Mantenemos colapsado
)

# Constantes de configuración
PASSWORD_POLICIA = "policia"
# Opciones de navegación para cada usuario
PAGES_POLICIA = ["🏠 Home", "📊 Analysis", "🗺️ Map", "ℹ️ Info", "👥 Our Team", "🚨 Alertas"]
PAGES_THALES = ["🏠 Home", "📊 Analysis", "🗺️ Map", "ℹ️ Info", "👥 Our Team"] # Sin "🚨 Alertas"

# Inicializar o asegurar el estado de la sesión
if 'user' not in st.session_state:
    st.session_state.user = None

# --- Función para manejar la acción de Acceso ---
def handle_login(user_type):
    """Establece el usuario en la sesión y fuerza una nueva ejecución."""
    st.session_state.user = user_type
    st.rerun()

# --- Función para renderizar la página de selección de usuario ---
def render_selection_page():
    # Usamos un contenedor principal para centrar el contenido y evitar otros elementos
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(
            """
            <style>
            .stSelectbox label {
                font-size: 1.25rem;
                font-weight: 600;
            }
            .stButton>button {
                width: 100%;
                font-size: 1.1rem;
                padding: 0.5rem;
                margin-top: 1rem;
            }
            .title-selection {
                font-size: 2rem;
                font-weight: 700;
                text-align: center;
                margin-bottom: 2rem;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div class='title-selection'>Selecciona tu usuario para continuar</div>", unsafe_allow_html=True)

        user = st.selectbox(
            "Elige un rol:",
            ["Thales", "Policía"],
            index=0,
            key="user_role_select"
        )

        # Si el usuario es Policía, pide la contraseña
        if user == "Policía":
            password = st.text_input("Introduce la contraseña:", type="password", key="password_input")
            
            # Se usa el argumento 'on_click' del botón para manejar la lógica
            if st.button("Acceder como Policía"):
                if password == PASSWORD_POLICIA:
                    # Llama a la función de manejo de login si la contraseña es correcta
                    handle_login("Policía")
                else:
                    st.error("Contraseña incorrecta. Inténtalo de nuevo.")
        
        # Si el usuario es Thales, permite el acceso directo
        elif user == "Thales":
            # Se usa el argumento 'on_click' del botón para manejar la lógica
            if st.button("Acceder como Thales"):
                # Llama a la función de manejo de login directamente
                handle_login("Thales")

# --- Función principal de la aplicación ---
def render_main_dashboard():
    # 1. Configuración del Sidebar (Navegación y Logout)
    st.sidebar.title("🔎 Navegación")
    
    # Seleccionar las páginas disponibles según el usuario
    if st.session_state.user == "Policía":
        available_pages = PAGES_POLICIA
        st.sidebar.markdown(f"**Usuario:** `Policía`")
    else:
        available_pages = PAGES_THALES
        st.sidebar.markdown(f"**Usuario:** `Thales`")

    # Muestra las opciones de navegación
    # CAMBIO: Usamos index=0 para que por defecto sea "🏠 Home"
    page = st.sidebar.radio(
        "Ir a:",
        available_pages,
        index=0 # Ahora siempre arranca en el primer elemento, que es "🏠 Home"
    )

    # Botón de "Cerrar sesión"
    if st.sidebar.button("Cerrar sesión"):
        st.session_state.user = None  # Reiniciar la selección de usuario
        st.rerun()      # Volver a cargar la página para mostrar la selección

    # 2. Router de páginas (Llamada a los módulos de renderizado)
    # Muestra el título del dashboard solo cuando el usuario está logueado
    st.title(f"{page.split(' ')[-1]} - {st.session_state.user} View")

    try:
        # Nota: La comparación de cadenas ahora incluye el emoji para ser más robusto
        if page == "Home":
            ui_home.render() 
            st.info("Renderizando la página Home...") # Placeholder
        elif page == "Análisis":
            ui_analysis.render()
            st.info("Renderizando la página de Análisis...") # Placeholder
        elif page == "Map":
            ui_map.render()
            st.info("Renderizando la página de Mapas...") # Placeholder
        elif page == "Info":
            ui_info.render()
            st.info("Renderizando la página de Información...") # Placeholder
        elif page == "Our Team":
            ui_ourteam.render()
            st.info("Renderizando la página Nuestro Equipo...") # Placeholder
        elif page == "Alertas" and st.session_state.user == "Policía":
            ui_alerts.render()
            st.info("Renderizando la página de Alertas (Sólo visible para Policía)...") # Placeholder
        else:
            st.warning("Selecciona una opción en el menú lateral.")

    except NameError as e:
        st.error(f"⚠️ Error de módulo: {e}. Asegúrate de que todos los módulos (`ui_home`, etc.) estén disponibles.")
    except Exception as e:
        st.error(f"⚠️ Ocurrió un error al renderizar la página: {e}")
        st.info("Revisa que la base de datos 'cdmx_insights.db' exista y que los módulos estén actualizados.")

# --- Lógica de arranque (Control de flujo) ---
if st.session_state.user is None:
    # Si no hay usuario, sólo se renderiza la página de selección (sin st.sidebar)
    render_selection_page()
else:
    # Si hay usuario, se renderiza el dashboard principal (con st.sidebar)
    render_main_dashboard()