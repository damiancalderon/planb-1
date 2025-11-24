# Plataforma de Inteligencia Delictiva CDMX – Overview

## Purpose

The project delivers an interactive intelligence dashboard for crime incidents in Mexico City. Analysts and command centers can explore descriptive metrics, spatial patterns, predictive alerts, and AI-generated summaries tailored either for Thales stakeholders or police operators. The stack combines a local DuckDB warehouse, several ML models (classification, clustering, time-series), and a Streamlit UI that orchestrates dedicated pages for Inicio, Análisis, Mapa, Información, ¿Quiénes Somos? y Alertas.

## High-level architecture

```
cleaned_crime_data.csv ─┐
                        │   load_data.py      ┌─────────────────┐
                        └─▶ DuckDB (crimes) ─▶│ modules/data.py │
                                                │   run_query     │
                                                └────────┬────────┘
                                                         │
                                                         ▼
                                                  Streamlit app.py
                                                (modules/ui_*.py)
```

Supporting services add predictive layers:

1. **KMeans + metadata**
   - `train_model.py` computes spatial clusters (`kmeans_zonas.joblib`).
   - `crear_cluster_info.py` enriches each centroid with nearby addresses and crime context (`cluster_info.csv`).
2. **Violence classification**
   - `train_model.py` trains RandomForest and XGBoost models that output `Violent` vs `Non-Violent` probabilities (`violence_*_v3.joblib`).
3. **Crime forecasting**
   - A SARIMA model (`crime_forecaster.pkl`) predicts daily totals for long-term planning.
4. **Generative insights**
   - `modules/ui_home.py` and `modules/ui_alerts.py` push context to an n8n webhook which forwards it to Gemini for narrative summaries.

## Data flow

1. **Raw ingestion** – `load_data.py` creates/updates `cdmx_insights.db` by reading `cleaned_crime_data.csv`, filtering `anio_hecho >= 2016`, and storing the result in a DuckDB table called `crimes`.
2. **Operational queries** – `database.py` and `modules/data.py` expose cached helpers (`run_query`, `get_crime_stats`, `get_all_alcaldias`, etc.) so UI components read consistent snapshots.
3. **Pre-processing for maps** – `data_processing.py` cleans the CSV (dates, coordinates, categoricals) and prepares yearly slices. `mapa.py` and `modules/ui_map.py` transform those dataframes into Folium layers, heatmaps, and timeline animations.
4. **Model training** – `train_model.py` pulls fresh samples from DuckDB, engineers temporal/spatial features, performs hyperparameter search, and persists the fitted estimators in the repo root. The models are loaded lazily from inside Streamlit via `joblib.load`.
5. **Cluster annotation** – Once KMeans is refreshed, `crear_cluster_info.py` uses Nominatim reverse geocoding to label each zone with the closest address plus its dominant alcaldía and crime category. `modules/location_utils.py` provides caching and join helpers.

## Streamlit structure (`app.py`)

`app.py` handles authentication-lite switching between Thales and Policía personas:

1. **Role selection** – unauthenticated users choose the persona; Policía requires the password defined in `PASSWORD_POLICIA`.
2. **Navigation router** – for the logged-in user, the sidebar radio toggles between the allowed modules. Each option calls the matching renderer inside `modules/ui_<page>.py`.
3. **Session state** – `st.session_state.user` decides which navigation set (`PAGES_POLICIA` vs `PAGES_THALES`) is available and whether the Alertas page is exposed.

### Key UI modules

| Module | Responsibilities |
| --- | --- |
| `modules/ui_home.py` | Hero metrics, KPI cards, auto-generated “Gemini insights.” |
| `modules/ui_analysis.py` | Time-series charts (Altair/Plotly), SARIMAX forecasts, category distributions. |
| `modules/ui_map.py` | Interactive Folium map with filters by delito, horario, violencia; integrates `mapa.py` and DuckDB queries. |
| `modules/ui_info.py` | Contextual stats, storyboards, descriptive sections for stakeholders. |
| `modules/ui_alerts.py` | Predictive alert lab: geocodes addresses, classifies risk with XGBoost, shows PyDeck maps, and uses n8n for narrative output. |
| `modules/ui_ourteam.py` | Static branding and team member information. |

## How a typical request is served

1. User selects a persona, then chooses a page from the sidebar.
2. The page renderer grabs cached data through `modules/data.py` or `database.py`.
3. If the view requires predictive context (Alertas), it loads `violence_xgb_optimizado_v3.joblib`, `kmeans_zonas.joblib`, and `cluster_info.csv`, then preprocesses the user input via helper functions before scoring.
4. Visual components (Altair charts, Folium maps, PyDeck layers) render inside Streamlit. Heavy operations (DuckDB queries, Map generation, geocoding) are wrapped in `@st.cache_*` to avoid recomputation.
5. Optional: the module packages summarized context into JSON and posts it to the configured n8n webhook, which returns a Gemini-generated narrative shown in the UI.

## External integrations

- **DuckDB** – embedded analytics database. All dashboards read from `cdmx_insights.db`.
- **n8n webhook** – triggers Gemini or other automation for natural-language summaries.
- **Nominatim** – Geocoding and reverse geocoding for cluster labeling and user-entered addresses.
- **Folium / PyDeck** – Map visualization layers; these depend on GeoJSON files (`alcaldias.geojson`, `coloniascdmx.geojson`) bundled in the repo.

## Extending the system

- **New data sources** – append columns to `cleaned_crime_data.csv`, rerun `load_data.py`, then expose the new fields via `run_query` helpers to keep the UI loosely coupled.
- **Additional models** – follow the structure in `train_model.py`: encapsulate feature engineering, persist trained artifacts to disk, and load them with `st.cache_resource` inside the relevant module.
- **More user roles** – add role strings to `PAGES_*` in `app.py` and create corresponding UI modules. Remember to align navigation labels with the conditionals in `render_main_dashboard`.

This document should give new contributors enough context to understand why the platform exists and how its pieces communicate from ingestion to visualization.
