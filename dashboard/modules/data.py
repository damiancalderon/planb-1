# modules/data.py
import streamlit as st
import duckdb
import pandas as pd

DB_FILE = "cdmx_insights.db"

@st.cache_resource
def get_db_connection():
    try:
        con = duckdb.connect(DB_FILE, read_only=True)
        return con
    except duckdb.Error as e:
        st.error(f"Error al conectar con la base de datos: {e}")
        return None

@st.cache_data
def run_query(query, params=None):
    con = get_db_connection()
    if con:
        try:
            if params:
                return con.execute(query, params).fetchdf()
            return con.execute(query).fetchdf()
        except duckdb.Error as e:
            st.error(f"Error al ejecutar la consulta: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

# --- Dashboard ---
def get_all_alcaldias():
    query = """
    SELECT DISTINCT alcaldia_hecho
    FROM crimes
    WHERE alcaldia_hecho IS NOT NULL
    ORDER BY alcaldia_hecho
    """
    return run_query(query)

def get_crime_stats():
    query = """
    SELECT
        COUNT(*) AS total_delitos,
        COUNT(*) * 1.0 / COUNT(DISTINCT CAST(fecha_hecho AS DATE)) AS promedio_diario
    FROM crimes
    """
    return run_query(query)

def get_top_alcaldias():
    query = """
    SELECT
        alcaldia_hecho,
        COUNT(*) AS total
    FROM crimes
    WHERE alcaldia_hecho IS NOT NULL
    GROUP BY alcaldia_hecho
    ORDER BY total DESC
    LIMIT 5
    """
    return run_query(query)

# --- Analysis.py ---
def get_historical_tendency():
    query = """
    SELECT 
        CAST(fecha_hecho AS DATE) AS fecha,
        COUNT(*) AS total_delitos
    FROM crimes
    GROUP BY fecha
    ORDER BY fecha ASC
    """
    return run_query(query)

def get_distribution_by_category():
    query = """
    SELECT
        categoria_delito,
        COUNT(*) AS total
    FROM crimes
    GROUP BY categoria_delito
    ORDER BY total DESC
    """
    return run_query(query)

def get_distribution_by_hour():
    query = """
    SELECT
        hora_hecho,
        COUNT(*) AS total
    FROM crimes
    GROUP BY hora_hecho
    ORDER BY hora_hecho ASC
    """
    return run_query(query)

def get_violence_heatmap_data():
    query = """
    SELECT
        hora_hecho,
        violence_type,
        COUNT(*) AS total
    FROM crimes
    WHERE violence_type IS NOT NULL
    GROUP BY hora_hecho, violence_type
    ORDER BY hora_hecho
    """
    return run_query(query)

def get_violence_time_metrics():
    query = """
    WITH time_classified AS (
        SELECT
            violence_type,
            CASE 
                WHEN HOUR(hora_hecho) >= 19 OR HOUR(hora_hecho) <= 7 THEN 'Noche (19-07)'
                ELSE 'Día (07-19)'
            END AS franja_horaria
        FROM crimes
        WHERE violence_type = 'Violent'
    )
    SELECT
        franja_horaria,
        COUNT(*) AS total,
        (COUNT(*) * 100.0 / SUM(COUNT(*)) OVER ()) AS porcentaje
    FROM time_classified
    GROUP BY franja_horaria
    """
    return run_query(query)

# --- Mapa.py ---
def get_all_crime_categories():
    query = """
    SELECT DISTINCT categoria_delito
    FROM crimes
    WHERE categoria_delito IS NOT NULL
    ORDER BY categoria_delito
    """
    return run_query(query)

def get_filtered_map_data(crime_types, hour_range, classification):
    query = """
    SELECT latitud, longitud
    FROM crimes
    WHERE latitud IS NOT NULL AND longitud IS NOT NULL
    """
    params = []

    if crime_types:
        placeholders = ", ".join(["?"] * len(crime_types))
        query += f" AND categoria_delito IN ({placeholders})"
        params.extend(crime_types)

    query += " AND HOUR(hora_hecho) BETWEEN ? AND ?"
    params.append(hour_range[0])
    params.append(hour_range[1])

    if classification != 'Ambos':
        query += " AND violence_type = ?"
        params.append(classification)

    query += " LIMIT 50000"
    return run_query(query, params)
