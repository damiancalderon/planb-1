import streamlit as st
import os
from pathlib import Path


def render():
    st.markdown("""
        <style>
        .team-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr); /* 3 columnas en la parte superior */
            gap: 18px;
            margin-top: 1rem;
        }
        .team-grid-bottom {
            display: grid;
            grid-template-columns: repeat(2, 1fr); /* 2 columnas en la parte inferior */
            gap: 18px;
        }
        .card {
            background: var(--background-color, #ffffff);
            border: 1px solid rgba(0,0,0,.08);
            border-radius: 16px;
            padding: 14px;
            box-shadow: 0 2px 8px rgba(0,0,0,.04);
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .quote {
            font-size: .95rem;
            color: #374151;
            line-height: 1.35;
            margin-top: 8px;
        }
        .section {
            background: rgba(0, 0, 0, .02);
            border: 1px solid rgba(0, 0, 0, .06);
            border-radius: 16px;
            padding: 16px;
        }
        .kicker {
            letter-spacing: .08em;
            text-transform: uppercase;
            font-weight: 600;
            font-size: .8rem;
            color: #6b7280;
            margin-bottom: 6px;
        }
        .card img {
            width: 80%; /* Hacemos la imagen más pequeña */
            height: auto; /* Ajustamos la altura automáticamente */
            object-fit: cover;
            border-radius: 999px;
            margin-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("Nuestro Equipo")
    st.write("Conoce a las personas detrás de nuestro equipo y descubre nuestra misión y los objetivos que guían todo lo que hacemos")

    # Team data
    team = [
        {
            "name": "Damian Calderon Capallera",
            "img": Path("/Users/dmart/OneDrive/Imágenes/Documentos/reto/planb/dashboard/damian.png"),
            "quote": "Forecasting crime risk helps allocate resources before spikes happen."
        },
        {
            "name": "Daniela Martínez Xolalpa",
            "img": Path("/Users/dmart/OneDrive/Imágenes/Documentos/reto/planb/dashboard/dani.png"),
            "quote": "Space-time patterns reveal where prevention can be most effective."
        },
        {
            "name": "José de Jesús Rodríguez Rocha",
            "img": Path("/Users/dmart/OneDrive/Imágenes/Documentos/reto/planb/dashboard/chuy.png"),
            "quote": "From noise to narrative: modeling explains the ‘why’, not just the ‘what’."
        },
        {
            "name": "Fernando Vázquez Rivera",
            "img": Path("/Users/dmart/OneDrive/Imágenes/Documentos/reto/planb/dashboard/fercho.png"),
            "quote": "Responsible prediction means insight that informs action—never bias."
        },
        {
            "name": "Luis Gallegos Pérez",
            "img": Path("/Users/damcalde/RETO/planb/dashboard/luis.png"),  # Ruta local con Path
            "quote": "Innovative solutions come from the ability to see patterns that others miss."
        },
        {
            "name": "Alejandro Cortés",
            "img": Path("/Users/damcalde/RETO/planb/dashboard/alejandro.png"),  # Ruta local con Path
            "quote": "Transforming data into meaningful insights is key to strategic decision-making."
        },
    ]

    # Mostrar información del equipo (Parte superior: 3 columnas)
    st.markdown('<div class="kicker">Team</div>', unsafe_allow_html=True)
    cols = st.columns(len(team))
    for idx, m in enumerate(team):
        with cols[idx]:
            if os.path.exists(m["img"]):
                st.image(m["img"], width=120)
            else:
                st.warning(f"Image not found: {m['img']}")
            st.markdown(f"**{m['name']}**", unsafe_allow_html=True)
            st.markdown(f'<div class="quote">“{m["quote"]}”</div>', unsafe_allow_html=True)

    st.write("")  # spacing

    # Mission
    st.markdown('<div class="kicker">Nuestra misión</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section">
        <p><strong>Nuestra misión es transformar los datos en conocimiento que impulse un cambio significativo.</strong>
        En una ciudad donde se reportan cientos de delitos cada día, comprender el cuándo, dónde y por qué de cada incidente es esencial. Creemos que los datos, cuando se analizan con propósito y precisión, pueden iluminar los patrones que configuran la seguridad urbana y ayudar a guiar decisiones más inteligentes.</p>

        <p>A través del uso de análisis de datos, visualización e información social, nuestro objetivo es descubrir las historias ocultas dentro de los números, revelando cómo las rutinas diarias, el comportamiento social y la estructura de la ciudad influyen en la dinámica del crimen. Al hacerlo, buscamos apoyar estrategias de prevención basadas en evidencia que hagan nuestras comunidades más seguras y resilientes.</p>

        <p>Sostenemos la idea de que la información por sí sola no es suficiente: debe conducir a la comprensión, y la comprensión debe conducir a la acción. Ese es el camino que seguimos: pasar de los incidentes a las ideas.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")  # spacing

    # Goals
    st.markdown('<div class="kicker">Nuestros objetivos</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section">
        <p><strong>Nuestro objetivo es transformar los datos complejos sobre delitos en información clara y accionable que ayude a comprender y prevenir la inseguridad urbana.</strong>
         Aspiramos a combinar tecnología, pensamiento analítico y conciencia social para identificar tendencias significativas y apoyar a los responsables de la toma de decisiones, investigadores y comunidades en el desarrollo de estrategias efectivas para la seguridad y la prevención.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

