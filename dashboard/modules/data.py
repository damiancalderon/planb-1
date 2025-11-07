import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
from .helpers import _strip_accents_upper, classify_crime, classify_violence

@st.cache_data
def load_info_csv():
    path = Path('carpetasFGJ_acumulado_2025_01.csv')
    if not path.exists():
        st.error("No encontré 'carpetasFGJ_acumulado_2025_01.csv' en el folder actual.")
        return pd.DataFrame()
    df = pd.read_csv(path)

    if 'fecha_hecho' in df.columns:
        df['fecha_hecho'] = pd.to_datetime(df['fecha_hecho'], errors='coerce')
    if 'hora_hecho' in df.columns:
        hh = pd.to_datetime(df['hora_hecho'], errors='coerce', format='%H:%M:%S')
        df['hour_hecho'] = hh.dt.hour
    if 'hour_hecho' not in df.columns or df['hour_hecho'].isna().all():
        df['hour_hecho'] = 12

    if 'fecha_hecho' in df.columns:
        if 'hora_hecho' in df.columns:
            hh_str = df['hora_hecho'].astype(str).fillna('12:00:00')
            df['datetime'] = pd.to_datetime(df['fecha_hecho'].dt.date.astype(str)+' '+hh_str, errors='coerce')
        else:
            df['datetime'] = df['fecha_hecho']
    else:
        df['datetime'] = pd.to_datetime('2020-01-01')

    df['anio_inicio'] = df['datetime'].dt.year.astype('Int64')
    df['mes_inicio']  = df['datetime'].dt.month.astype('Int64')
    df['dia_semana']  = df['datetime'].dt.day_name()

    if 'delito' in df.columns: df['delito_std'] = df['delito']
    elif 'delito_N' in df.columns: df['delito_std'] = df['delito_N']
    else: df['delito_std'] = ''

    if 'alcaldia_hecho_N' in df.columns: df['alcaldia_std'] = df['alcaldia_hecho_N']
    elif 'alcaldia_hecho' in df.columns: df['alcaldia_std'] = df['alcaldia_hecho'].apply(_strip_accents_upper)
    else: df['alcaldia_std'] = ''

    if 'colonia_hecho_N' in df.columns: df['colonia_std'] = df['colonia_hecho_N']
    elif 'colonia_hecho' in df.columns: df['colonia_std'] = df['colonia_hecho'].apply(_strip_accents_upper)
    else: df['colonia_std'] = ''

    df['crime_classification'] = df['delito_std'].apply(classify_crime)
    df['violence_type'] = df.apply(lambda r: classify_violence(r['crime_classification'], r['delito_std']), axis=1)

    if 'latitud' not in df.columns:  df['latitud'] = np.nan
    if 'longitud' not in df.columns: df['longitud'] = np.nan
    return df

@st.cache_data
def load_map_csv():
    cleaned = Path('cleaned_crime_data.csv')
    raw = Path('carpetasFGJ_acumulado_2025_01.csv')
    if cleaned.exists(): df = pd.read_csv(cleaned)
    elif raw.exists():   df = pd.read_csv(raw)
    else:
        st.error("No encontré 'cleaned_crime_data.csv' ni 'carpetasFGJ_acumulado_2025_01.csv'.")
        return pd.DataFrame()

    if 'datetime' not in df.columns:
        if 'fecha_hecho' in df.columns:
            df['fecha_hecho'] = pd.to_datetime(df['fecha_hecho'], errors='coerce')
            if 'hora_hecho' in df.columns:
                hh = pd.to_datetime(df['hora_hecho'], errors='coerce', format='%H:%M:%S')
                df['datetime'] = pd.to_datetime(df['fecha_hecho'].dt.date.astype(str)+' '+df['hora_hecho'].astype(str), errors='coerce')
                df['hour_hecho'] = hh.dt.hour
            else:
                df['datetime'] = df['fecha_hecho']
        else:
            df['datetime'] = pd.to_datetime('2020-01-01')

    if 'delito_N' not in df.columns and 'delito' in df.columns:
        df['delito_N'] = df['delito'].apply(_strip_accents_upper)
    if 'alcaldia_hecho_N' not in df.columns:
        if 'alcaldia_hecho' in df.columns: df['alcaldia_hecho_N'] = df['alcaldia_hecho'].apply(_strip_accents_upper)
        else: df['alcaldia_hecho_N'] = ''

    if 'latitud' not in df.columns or 'longitud' not in df.columns:
        st.warning("El CSV para mapa no trae latitud/longitud; el heatmap/markers no podrán mostrarse.")
        df['latitud'] = np.nan; df['longitud'] = np.nan
    return df

@st.cache_data
def load_geojson_choice(choice: str):
    path = Path(choice)
    if not path.exists():
        st.error(f"No encontré '{choice}' en el folder actual.")
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
