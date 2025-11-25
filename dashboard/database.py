import duckdb
import pandas as pd
import streamlit as st

from paths import DUCKDB_FILE

# Nombre del archivo de la base de datos
DB_FILE = DUCKDB_FILE

@st.cache_resource
def get_db_connection():
    """
    Crea y cachea una conexión a la base de datos DuckDB.
    """
    try:
        con = duckdb.connect(str(DB_FILE), read_only=True)
        return con
    except duckdb.Error as e:
        st.error(f"Error al conectar con la base de datos: {e}")
        return None

@st.cache_data
def run_query(query, params=None):
    """
    Ejecuta una consulta y cachea el resultado.
    """
    con = get_db_connection()
    if con:
        try:
            if params:
                df = con.execute(query, params).fetchdf()
            else:
                df = con.execute(query).fetchdf()
            return df
        except duckdb.Error as e:
            st.error(f"Error al ejecutar la consulta: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

# --- Funciones para Dashboard (app.py) ---

def get_all_alcaldias():
    """
    Obtiene una lista única de todas las 'alcaldia_hecho' para el filtro.
    """
    query = """
    SELECT DISTINCT 
        alcaldia_hecho 
    FROM 
        crimes 
    WHERE 
        alcaldia_hecho IS NOT NULL
    ORDER BY 
        alcaldia_hecho
    """
    return run_query(query)

def get_crime_stats():
    """
    Calcula KPIs: Total de delitos y promedio diario.
    """
    query = """
    SELECT
        COUNT(*) AS total_delitos,
        COUNT(*) / COUNT(DISTINCT CAST(fecha_hecho AS DATE)) AS promedio_diario
    FROM
        crimes
    """
    return run_query(query)

def get_top_alcaldias():
    """
    Obtiene las 5 alcaldías con más delitos.
    """
    query = """
    SELECT
        alcaldia_hecho,
        COUNT(*) AS total
    FROM
        crimes
    WHERE
        alcaldia_hecho IS NOT NULL
    GROUP BY
        alcaldia_hecho
    ORDER BY
        total DESC
    LIMIT 5
    """
    return run_query(query)

# --- Funciones para Analysis.py ---

def get_historical_tendency():
    """
    Consulta la tendencia histórica de delitos por fecha.
    """
    query = """
    SELECT 
        CAST(fecha_hecho AS DATE) AS fecha,
        COUNT(*) AS total_delitos
    FROM 
        crimes
    GROUP BY 
        fecha
    ORDER BY 
        fecha ASC
    """
    return run_query(query)

def get_distribution_by_category():
    """
    Obtiene la cuenta de delitos por 'categoria_delito'.
    """
    query = """
    SELECT
        categoria_delito,
        COUNT(*) AS total
    FROM
        crimes
    GROUP BY
        categoria_delito
    ORDER BY
        total DESC
    """
    return run_query(query)

def get_distribution_by_hour():
    """
    Obtiene la cuenta de delitos por 'hora_hecho'.
    """
    query = """
    SELECT
        hora_hecho,
        COUNT(*) AS total
    FROM
        crimes
    GROUP BY
        hora_hecho
    ORDER BY
        hora_hecho ASC
    """
    return run_query(query)

def get_violence_heatmap_data():
    """
    Obtiene la cuenta de delitos por hora y tipo de violencia.
    """
    query = """
    SELECT
        hora_hecho,
        violence_type,
        COUNT(*) AS total
    FROM
        crimes
    WHERE
        violence_type IS NOT NULL
    GROUP BY
        hora_hecho, violence_type
    ORDER BY
        hora_hecho
    """
    return run_query(query)

def get_violence_time_metrics():
    """
    Calcula los porcentajes de crímenes violentos de noche vs día.
    (Mockup: 40% vs 60%)
    """
    query = """
    WITH time_classified AS (
        SELECT
            violence_type,
            CASE 
                WHEN HOUR(hora_hecho) >= 19 OR HOUR(hora_hecho) <= 7 THEN 'Noche (19-07)'
                ELSE 'Día (07-19)'
            END AS franja_horaria
        FROM
            crimes
        WHERE
            violence_type = 'Violent'
    )
    SELECT
        franja_horaria,
        COUNT(*) AS total,
        (COUNT(*) * 100.0 / SUM(COUNT(*)) OVER ()) AS porcentaje
    FROM
        time_classified
    GROUP BY
        franja_horaria
    """
    return run_query(query)


# --- Funciones para Mapa.py ---

def get_all_crime_categories():
    """
    Obtiene una lista única de todas las 'categoria_delito' para el filtro.
    """
    query = """
    SELECT DISTINCT 
        categoria_delito 
    FROM 
        crimes 
    WHERE 
        categoria_delito IS NOT NULL
    ORDER BY 
        categoria_delito
    """
    return run_query(query)

def get_filtered_map_data(crime_types, hour_range, classification):
    """
    Obtiene datos filtrados de latitud y longitud para el mapa.
    """
    query = """
    SELECT 
        latitud, 
        longitud 
    FROM 
        crimes 
    WHERE 
        latitud IS NOT NULL AND longitud IS NOT NULL
    """
    
    params = []

    # 1. Filtro por tipo de crimen (multiselect)
    if crime_types:
        placeholders = ", ".join(["?"] * len(crime_types))
        query += f" AND categoria_delito IN ({placeholders})"
        params.extend(crime_types)
        
    # 2. Filtro por hora (slider)
    query += " AND HOUR(hora_hecho) BETWEEN ? AND ?"
    params.append(hour_range[0])
    params.append(hour_range[1])

    # 3. Filtro por clasificación (radio)
    if classification != 'Ambos':
        query += " AND violence_type = ?"  # Asumiendo 'violence_type' para 'Violent'/'Non-Violent'
        params.append(classification)
    
    # Para optimizar, limitamos a 50,000 puntos en el mapa
    query += " LIMIT 50000"

    return run_query(query, params)

# --- Funciones para EDA.py ---

def get_category_violence_heatmap():
    """
    Obtiene la cuenta de delitos por categoría y tipo de violencia.
    (Para el heatmap en EDA.py)
    """
    query = """
    SELECT
        crime_classification,
        violence_type,
        COUNT(*) AS total
    FROM
        crimes
    WHERE
        violence_type IS NOT NULL AND crime_classification IS NOT NULL
    GROUP BY
        crime_classification, violence_type
    """
    return run_query(query)

def get_yearly_violence_trend():
    """
    Obtiene la tendencia anual de crímenes violentos vs no violentos.
    (Para el gráfico de línea en EDA.py)
    """
    query = """
    SELECT
        anio_hecho,
        violence_type,
        COUNT(*) AS total
    FROM
        crimes
    WHERE
        violence_type IS NOT NULL
    GROUP BY
        anio_hecho, violence_type
    ORDER BY
        anio_hecho ASC
    """
    return run_query(query)

def get_alcaldia_distribution(limit=10):
    """
    Obtiene el top 10 de alcaldías por número de crímenes.
    (Para el gráfico de barras en EDA.py)
    """
    query = f"""
    SELECT
        alcaldia_hecho,
        COUNT(*) AS total
    FROM
        crimes
    WHERE
        alcaldia_hecho IS NOT NULL
    GROUP BY
        alcaldia_hecho
    ORDER BY
        total DESC
    LIMIT {limit}
    """
    return run_query(query)

@st.cache_data
def get_map_data_by_date(anio, mes):
    """
    Obtiene datos de lat/lon para el mapa, filtrados por año y mes.
    """
    query = """
    SELECT 
        latitud, 
        longitud 
    FROM 
        crimes 
    WHERE 
        anio_hecho = ? AND mes_hecho = ? 
        AND latitud IS NOT NULL AND longitud IS NOT NULL
    LIMIT 50000 -- Limita a 50k puntos para que Pydeck sea rápido
    """
    params = [anio, mes]
    return run_query(query, params)
