import streamlit as st
from bootstrap_assets import ensure_assets
# Importa tus módulos de renderizado, se asume que existen:
# En app.py
from modules import ui_home, ui_analysis, ui_info, ui_ourteam, ui_alerts, ui_map  # Importar desde el paquete modules

# Si estos módulos no existen, el código fallará al intentar llamarlos.

# Garantiza que los artefactos críticos existan incluso en despliegues limpios.
ensure_assets()

# --- Configuración de página (debe ir antes de cualquier render) ---
st.set_page_config(
    page_title="Plataforma de Inteligencia Delictiva CDMX",
    layout="wide",
    initial_sidebar_state="collapsed" # Mantenemos colapsado
)

# Constantes de configuración
PASSWORD_POLICIA = "policia"
# Opciones de navegación para cada usuario
PAGES_POLICIA = ["Inicio", "Análisis", "Mapa", "Información", "¿Quiénes Somos?", "Alertas"]
PAGES_THALES = ["Inicio", "Análisis", "Mapa", "Información", "¿Quiénes Somos?"] # Sin "🚨 Alertas"

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
    st.markdown(
        """
        <style>
        body {
            background-color: #E0F2F7;
        }
        .top-left-logo {
            position: absolute;
            top: 10px;
            left: 10px;
            width: 400px; /* Adjust size as needed */
            z-index: 1000;
        }
        </style>
        <img src="https://upload.wikimedia.org/wikipedia/commons/2/21/Thales_Logo.svg" class="top-left-logo" alt="Thales Logo">
        """,
        unsafe_allow_html=True
    )
    # Usamos un contenedor principal para centrar el contenido y evitar otros elementos
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(
            """
            <style>
            .stSelectbox label {
                font-size: 1.25rem;
                font-weight: 600;
                display: block; /* To apply text-align */
            }
            .stButton>button {
                width: 100%;
                font-size: 1.1rem;
                padding: 0.5rem;
                margin-top: 1rem;
                text-align: center;
            }
            .title-selection {
                font-size: 2rem;
                font-weight: 700;
                text-align: center;
                margin-bottom: 2rem;
            }
            .centered-content-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                width: 100%;
                margin-top: 15vh; /* Push content further down */
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div class='centered-content-container'>", unsafe_allow_html=True)
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
        st.markdown("</div>", unsafe_allow_html=True)

# --- Función principal de la aplicación ---
def render_main_dashboard():
    # 1. Configuración del Sidebar (Navegación y Logout)
    st.sidebar.title("Navegación")
    
    # Seleccionar las páginas disponibles según el usuario
    if st.session_state.user == "Policía":
        available_pages = PAGES_POLICIA
        st.sidebar.markdown(f"Usuario: Policía")
    else:
        available_pages = PAGES_THALES
        st.sidebar.markdown(f"Usuario: Thales")

    # Muestra las opciones de navegación
    page = st.sidebar.radio(
        "Ir a:",
        available_pages,
        index=0 # Arranca en el primer elemento, que es "🏠 Home"
    )

    # Botón de "Cerrar sesión"
    if st.sidebar.button("Cerrar sesión"):
        st.session_state.user = None  # Reiniciar la selección de usuario
        st.rerun()      # Volver a cargar la página para mostrar la selección

    # 2. Router de páginas (Llamada a los módulos de renderizado)
    # Muestra el título del dashboard solo cuando el usuario está logueado
    st.title(f"{page} - Vista de {st.session_state.user}")

    try:
        # CORRECCIÓN: Los nombres de las páginas deben coincidir exactamente con los elementos de la lista PAGES_...
        # Si tienes tus módulos instalados, DESCOMENTA las líneas de llamada (e.g., ui_home.render())

        if page == "Inicio":
            ui_home.render()
        elif page == "Análisis":
            ui_analysis.render()
        elif page == "Mapa":
            ui_map.render()
        elif page == "Información":
            ui_info.render()
        elif page == "¿Quiénes Somos?":
            ui_ourteam.render()
        elif page == "Alertas" and st.session_state.user == "Policía":
            ui_alerts.render()
        else:
            # Esta condición solo debería alcanzarse si hay un error o un estado inesperado
            st.warning(f"Error de navegación: No se encontró la página '{page}'.")

    except NameError as e:
        st.error(f"⚠ Error de módulo: {e}. Asegúrate de que todos los módulos (ui_home, ui_analysis, etc.) estén disponibles en tu entorno.")
    except Exception as e:
        st.error(f"⚠ Ocurrió un error al renderizar la página: {e}")
        st.info("Verifica que las funciones de renderizado dentro de tus módulos no contengan errores.")

# --- Lógica de arranque (Control de flujo) ---
if st.session_state.user is None:
    # Si no hay usuario, sólo se renderiza la página de selección (sin st.sidebar)
    render_selection_page()
else:
    # Si hay usuario, se renderiza el dashboard principal (con st.sidebar)
    render_main_dashboard()
