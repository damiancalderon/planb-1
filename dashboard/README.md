# Plataforma de Inteligencia Delictiva CDMX

Interactive Streamlit dashboard, DuckDB data mart, and ML models that surface real‑time insights, spatial analysis, and predictive alerts for incidents in Mexico City. The repository already bundles cleaned datasets, geojson layers, pretrained models, and helper scripts so you can rebuild the full stack end‑to‑end.

## Repository map

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit entry point that routes between Inicio, Análisis, Mapa, Información, ¿Quiénes Somos? y Alertas. |
| `modules/` | UI fragments plus shared helpers (`data.py`, `database.py`, `helpers.py`, `location_utils.py`, etc.). |
| `cleaned_crime_data.csv` | Canonical incident dataset used to seed the DuckDB database. |
| `load_data.py` | CLI helper that ingests the CSV into `cdmx_insights.db (DuckDB)` with the expected schema. |
| `train_model.py` | Re‑trains classification (Violent vs Non‑Violent), clustering, and SARIMA models. |
| `crear_cluster_info.py` | Adds geocoded labels for every spatial cluster created by `train_model.py`. |
| `data_processing.py`, `mapa.py` | Higher‑level helpers for folium/pydeck visualizations. |

## Requirements

- Python 3.10+ (PyEnv or the bundled `venv/` works; avoid the system Python).
- pip >= 23.
- DuckDB ≥ 0.9 (pulled via pip).
- Native libs required by GeoPandas/Shapely (on macOS: `brew install gdal`).  
- Internet access for Streamlit, OpenStreetMap/Nominatim geocoding, and the `n8n` webhooks used by `ui_home` y `ui_alerts`.

Install Python dependencies inside a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install streamlit pandas numpy duckdb geopandas shapely folium streamlit-folium statsmodels scikit-learn xgboost joblib altair plotly pydeck geopy requests unidecode branca pyproj certifi
```

> Tip: if you plan to distribute this project internally, capture the exact versions with `pip freeze > requirements.txt` once the install succeeds.

## Data & artifact preparation

1. **Place the raw files**  
   Ensure these assets stay in the repository root (already included in version control unless you removed them):  
   `cleaned_crime_data.csv`, `alcaldias.geojson`, `coloniascdmx.geojson`, and any new CSV feeds you plan to append.

2. **Create/refresh DuckDB**  
   ```bash
   python load_data.py
   ```  
   - Builds `cdmx_insights.db` with a `crimes` table filtered to `anio_hecho >= 2016`.  
   - The script asks before overwriting an existing DB; answer `s` (sí) to rebuild.

3. **(Optional) Retrain models**  
   Only needed when the dataset changes. Training downloads no external assets but will take several minutes.
   ```bash
   python train_model.py
   ```  
   Outputs: `kmeans_zonas.joblib`, `violence_rf_optimizado_v3.joblib`, `violence_xgb_optimizado_v3.joblib`, and `crime_forecaster.pkl`.

4. **Regenerate cluster metadata**  
   Required after retraining KMeans or when you delete `cluster_info.csv`. Uses Nominatim reverse geocoding, so keep the rate limit (~1 request/sec).  
   ```bash
   python crear_cluster_info.py
   ```

The rest of the modules (e.g., `data_processing.py`, `mapa.py`) work directly with the DuckDB database and the cached CSV/model artifacts produced above.

## Running the Streamlit dashboard

1. Activate your virtual environment (`source .venv/bin/activate`).
2. Execute Streamlit from the project root:
   ```bash
   streamlit run app.py
   ```
3. Visit `http://localhost:8501`.  
   - Select either **Thales** or **Policía** on the landing screen.  
   - The Policía view requires the password defined in `app.py` (`PASSWORD_POLICIA = "policia"` by default).  
   - If you change page names in `PAGES_*`, they must still match the respective `modules/ui_*.py`.

### External services

- **n8n + Gemini insights**: `modules/ui_home.py` and `modules/ui_alerts.py` call the webhook stored in `N8N_WEBHOOK_URL*_`. Replace the placeholders with your own automation endpoint and ensure outbound HTTPS is allowed.
- **Geocoding**: `modules/ui_alerts.py`, `data_processing.py`, and `crear_cluster_info.py` rely on OpenStreetMap’s Nominatim API. Heavy use should point to a self-hosted instance or add caching.

## Development workflow

- **Hot reload**: Streamlit reruns automatically on file changes; use `st.cache_*` decorators already in the code to keep queries fast.  
- **Database queries**: `modules/data.py` exposes cached helpers (`run_query`, `get_all_alcaldias`, etc.). Prefer these over ad-hoc DuckDB calls in new modules.  
- **Map assets**: add new GeoJSON layers beside `alcaldias.geojson` and load them with `Path(__file__).resolve().parent`.  
- **Credentials**: never hardcode new secrets. Convert constants like `PASSWORD_POLICIA` or webhook URLs into environment variables before production.

## Troubleshooting

- Missing shared libraries (GEOS/PROJ) while installing GeoPandas → install GDAL/GEOS via your package manager before running `pip install`.  
- `ModuleNotFoundError` for `xgboost` or `streamlit_folium` → re-run the pip install command above.  
- `violence_xgb_optimizado_v3.joblib not found` inside the Alertas view → retrain via `python train_model.py`.  
- Slow geocoding during `crear_cluster_info.py` → limit clusters, reuse cached CSV, or run against a personal Nominatim endpoint.  
- `duckdb.Error: Catalog Error: Table with name crimes already exists` → delete `cdmx_insights.db` or allow the script to overwrite it.

You now have everything needed to rebuild the database, refresh the ML artifacts, and serve the complete dashboard locally or on your preferred Streamlit host.
