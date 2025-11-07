import pandas as pd
from unidecode import unidecode
import geopandas as gpd
from shapely.geometry import Point
import folium
from folium.plugins import HeatMap
import os
from folium import plugins
import streamlit as st
from streamlit.components.v1 import html
import time
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import ssl, certifi

@st.cache_data
def load_data():
  try:
    df = pd.read_csv('data/carpetasFGJ_acumulado_2025_01.csv')
    boroughs_coordinates = 'data/alcaldias_cdmx.geojson'
    metro_coordinates = 'data/estaciones_metro.geojson'
    print("Data loaded successfully.")
  except FileNotFoundError:
    print(f"Error: One or more files were not found.")
    df = None
    boroughs_coordinates = None
    metro_coordinates = None
  except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    df = None
    boroughs_coordinates = None
    metro_coordinates = None
  return df, boroughs_coordinates, metro_coordinates

def strip_accents_upper(text):
  return unidecode(str(text)).upper().strip() if pd.notna(text) else text

@st.cache_data
def clean_data(df):  
  if df is not None:
    if 'fecha_hecho' in df.columns:
      df['fecha_hecho'] = pd.to_datetime(df['fecha_hecho'], errors='coerce')
      print("fecha_hecho format standardized.")
    else:
      print("Column 'fecha_hecho' not found in the dataframe.")
    if 'hora_hecho' in df.columns:
      try:
        df['hora_hecho_dt'] = pd.to_datetime(df['hora_hecho'], format='%H:%M:%S', errors='coerce').dt.time
      except Exception as e:
        df['hora_hecho_dt'] = pd.to_datetime(df['hora_hecho'], errors='coerce').dt.time
        print(f"Could not parse time with format 'HH:MM:SS'. Error: {e}")
        print("hora_hecho format standardized.")
    else:
      print("Column 'hora_hecho' not found in the dataframe.")

  categorical_cols = ['delito', 'alcaldia_hecho', 'colonia_hecho']
  for col in categorical_cols:
    if col in df.columns:
      df[col + '_N'] = df[col].apply(strip_accents_upper)
      df[col + '_N'] = df[col + '_N'].fillna('UNKNOWN')
      print(f"Categorical text in '{col}' normalized.")
    else:
      print(f"Column '{col}' not found in the dataframe.")

  initial_rows = len(df)
  df.drop_duplicates(inplace=True)
  rows_after_duplicates = len(df)
  print(f"Removed {initial_rows - rows_after_duplicates} duplicate rows.")

  critical_columns = ['fecha_hecho', 'hora_hecho', 'delito', 'alcaldia_hecho']
  df.dropna(subset=critical_columns, inplace=True)
  rows_after_missing = len(df)
  print(f"Removed {rows_after_duplicates - rows_after_missing} \
 rows with missing critical data.")
  
  df = df[
    (df['fecha_hecho'].dt.year >= 2016) &
    (df['fecha_hecho'].dt.year <= 2024)
  ]
  
  rows_after_year_filter = len(df)
  print(f"Filtered crimes to include only records from 2016 to 2024. Remaining rows: {rows_after_year_filter}")

  rows_after_future_dates = len(df)
  print(f"Removed {rows_after_missing - rows_after_future_dates}\
 rows with future dates.")
  
  if {'latitud', 'longitud'}.issubset(df.columns):
    df['latitud'] = pd.to_numeric(df['latitud'], errors='coerce')
    df['longitud'] = pd.to_numeric(df['longitud'], errors='coerce')
    initial_rows_geo = len(df)
    df = df[
        df['latitud'].between(19.0, 19.6) &
        df['longitud'].between(-99.4, -98.9)
    ]
    print(f"Removed {initial_rows_geo - len(df)} rows with invalid coordinates.")

    cols_to_drop = [col for col in df.columns if df[col].isna().all()]
    if cols_to_drop:
      df.drop(columns=cols_to_drop, inplace=True)
      print(f"Dropped entirely empty columns: {cols_to_drop}")

  if 'fecha_hecho' in df.columns:
    df['Weekday'] = df['fecha_hecho'].dt.day_name()
    df['Month'] = df['fecha_hecho'].dt.month_name()
    df['Year'] = df['fecha_hecho'].dt.year
  if 'hora_hecho_dt' in df.columns:
    df['Hour'] = pd.to_datetime(df['hora_hecho_dt'],
                                     format='%H:%M:%S').dt.hour
  print("Derived columns (Weekday, Month, Year, Hour) created.")

  print("\nData cleaning and initial normalization steps complete.")
  print("\nDisplaying info of the cleaned data before full normalization:")
  print(df.info())
  
  return df

def get_year_range(df):
    return int(df['Year'].min()), int(df['Year'].max())

@st.cache_resource
def get_geolocator():
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    return Nominatim(
        user_agent="cdmx-crime-map/1.0",
        ssl_context=ssl_context
    )

@st.cache_data
def geocode_address(address):
  try:
    geolocator = get_geolocator()
    time.sleep(1) 
    location = geolocator.geocode(address, country_codes="MX")
    if location:
      return (location.latitude, location.longitude)
    else:
      return (None, None)
  except (GeocoderTimedOut, GeocoderUnavailable) as e:
    print(f"Error geocoding '{address}': {e}")
    return (None, None)
  except Exception as e:
    print(f"An unexpected error occurred during geocoding: {e}")
    return (None, None)

@st.cache_data
def reverse_geocode_coords(lat, lon):
  try:
    geolocator = get_geolocator()
    time.sleep(1)
    location = geolocator.reverse((lat, lon), language="es")
    if location:
      return location.address
    else:
      return None
  except (GeocoderTimedOut, GeocoderUnavailable) as e:
    print(f"Error reverse geocoding '({lat}, {lon})': {e}")
    return None
  except Exception as e:
    print(f"An unexpected error occurred during reverse geocoding: {e}")
    return None
  
@st.cache_data
def load_all_years():
  df_raw, _, _ = load_data()
  
  if df_raw is None:
    print("Failed to load raw data. Aborting.")
    return {}
    
  cleaned_df = clean_data(df_raw)
    
  min_year, max_year = get_year_range(cleaned_df)
    
  yearly_dfs = {
    year: cleaned_df[cleaned_df['Year'] == year].reset_index(drop=True)
    for year in range(min_year, max_year + 1)
  }
    
  print(f"Loaded and split data for years {min_year} to {max_year}.")
  return yearly_dfs
  
def create_map(year):
  yearly_data = load_all_years()

  if year not in yearly_data:
    print(f"Year {year} not found in data.")
    return

  cleaned_crimes_df = yearly_data[year]

  _, boroughs_coordinates, metro_coordinates = load_data()

  if not boroughs_coordinates or not metro_coordinates:
      print("Missing geojson coordinate files. Map cannot be fully rendered.")
      
  if 'delito' not in cleaned_crimes_df.columns:
    print("Error: 'delito' column missing from cleaned data.")
    return

  robberies_df = cleaned_crimes_df[
    cleaned_crimes_df['delito'].str.contains('ROBO', case=False, na=False)
  ]
  
  crimes_gdf = gpd.GeoDataFrame(
    robberies_df,
    geometry=gpd.points_from_xy(robberies_df.longitud, robberies_df.latitud),
    crs="EPSG:4326"
  )
  
  try:
    metro_stations_gdf = gpd.read_file(metro_coordinates)
    print(f"Loaded {len(metro_stations_gdf)} metro stations.")
  except Exception as e:
    print(f"Could not load metro stations file: {e}")
    metro_stations_gdf = gpd.GeoDataFrame(columns=['nombre', 'geometry'], crs="EPSG:4326")

  crimes_proj = crimes_gdf.to_crs(epsg=3857)
  metro_proj = metro_stations_gdf.to_crs(epsg=3857)

  if not metro_proj.empty:
    metro_proj['buffer'] = metro_proj.buffer(50)

    metro_buffer_gdf = gpd.GeoDataFrame(
        metro_proj[['nombre']],
        geometry=metro_proj['buffer'],
        crs=metro_proj.crs
    )

    joined = gpd.sjoin(crimes_proj, metro_buffer_gdf, predicate='within')

    station_crime_counts = joined.groupby('nombre').size().reset_index(name='crime_count')

    metro_with_crimes = metro_stations_gdf.merge(station_crime_counts, on='nombre', how='left')
    metro_with_crimes['crime_count'] = metro_with_crimes['crime_count'].fillna(0)
  else:
    station_df = pd.DataFrame(columns=["nombre", "lat", "lon", "robos"])

  os.makedirs("output", exist_ok=True)

  m = folium.Map(location=[19.4326, -99.1332], zoom_start=11, tiles="cartodbpositron")

  if not metro_stations_gdf.empty:
    station_counts = []
    for idx, station in metro_stations_gdf.iterrows():
      nearby = crimes_gdf[
        crimes_gdf.geometry.within(station.geometry.buffer(0.0005)) 
      ]  
      station_counts.append(
        {
          "nombre": station["nombre"],
          "lat": station.geometry.y,
          "lon": station.geometry.x,
          "robos": len(nearby),
        }
      )
    station_df = pd.DataFrame(station_counts)
  
    max_robos = station_df["robos"].max() if station_df["robos"].max() > 0 else 1
    station_df["radius"] = (station_df["robos"] / max_robos) * 40

    robbery_layer = folium.FeatureGroup(name="Robberies near or in stations.", show=True)
    for _, row in station_df.iterrows():
      if row["robos"] == 0:
        color = "lightblue"
        radius = 3
      else:
        color = (
          "red"
          if row["robos"] > max_robos * 0.6
          else "orange"
          if row["robos"] > max_robos * 0.3
          else "green"
        )
        radius = row["radius"]

      folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=radius,
        color=color,
        fill=True,
        fill_opacity=0.6,
        popup=f"<b>{row['nombre']}</b><br>Robberies: {row['robos']}",
      ).add_to(robbery_layer)
    robbery_layer.add_to(m)

    metro_layer = folium.FeatureGroup(name="Metro stations", show=False)
    for _, row in metro_stations_gdf.iterrows():
      popup_text = f"<b>{row['nombre']}</b>"
      if 'ubicacion' in row:
          popup_text += f"<br>{row['ubicacion']}"
          
      folium.Marker(
        location=[row.geometry.y, row.geometry.x],
        icon=folium.Icon(color="blue", icon="train", prefix="fa"),
        popup=popup_text,
      ).add_to(metro_layer)
    metro_layer.add_to(m)
  
  if boroughs_coordinates:
    try:
      folium.GeoJson(
        boroughs_coordinates,
        name='Boroughs',
        style_function=lambda x: {'fillOpacity': 0, 'color': 'black', 'weight': 0.5}
      ).add_to(m)
    except Exception as e:
      print(f"Could not load boroughs geojson: {e}")

  folium.LayerControl(collapsed=False).add_to(m)

  map_html = m._repr_html_()
  html(map_html, height=600, scrolling=True)

def year_slider(df, label="Slide to select a year"):
  min_year = int(df['Year'].min())
  max_year = int(df['Year'].max())
  selected_year = st.slider(label, min_year, max_year, min_year)
  return selected_year