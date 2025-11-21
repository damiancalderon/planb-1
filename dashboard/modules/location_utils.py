from functools import lru_cache
from typing import Optional, Tuple

import pandas as pd

from .data import run_query

_PLACEHOLDER_TOKENS = (
    "sin geocoding",
    "error de geocoding",
    "direccion no encontrada",
    "dirección no encontrada",
)


def _clean_text(value: Optional[str]) -> str:
    """Return a trimmed string representation that is safe to inspect."""
    if isinstance(value, str):
        return value.strip()
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _needs_resolution(raw_value: Optional[str]) -> bool:
    """Detect whether the stored address is a placeholder."""
    cleaned = _clean_text(raw_value)
    if not cleaned:
        return True
    lowered = cleaned.lower()
    return any(token in lowered for token in _PLACEHOLDER_TOKENS)


@lru_cache(maxsize=512)
def _lookup_colonia(lat_key: float, lon_key: float) -> Tuple[Optional[str], Optional[str]]:
    """
    Find the closest colonia name to the provided coordinates.
    Uses progressively wider bounding boxes to keep the scan inexpensive.
    """
    lat = float(lat_key)
    lon = float(lon_key)
    search_radii = (0.01, 0.02, 0.05)

    for radius in search_radii:
        params = [
            lat - radius,
            lat + radius,
            lon - radius,
            lon + radius,
            lat,
            lat,
            lon,
            lon,
        ]
        df = run_query(
            """
            SELECT colonia_hecho, alcaldia_hecho
            FROM crimes
            WHERE latitud BETWEEN ? AND ?
              AND longitud BETWEEN ? AND ?
              AND colonia_hecho IS NOT NULL
            ORDER BY ((latitud - ?) * (latitud - ?) + (longitud - ?) * (longitud - ?)) ASC
            LIMIT 1
            """,
            params,
        )
        if df.empty:
            continue

        record = df.iloc[0]
        colonia = _clean_text(record.get("colonia_hecho"))
        alcaldia = _clean_text(record.get("alcaldia_hecho"))
        if colonia:
            return colonia, alcaldia or None

    return None, None


def build_cluster_label(cluster_row: pd.Series) -> str:
    """
    Produce a friendly label for a cluster row, resolving placeholder text when needed.
    """
    if cluster_row is None:
        return "Ubicación sin referencia"

    stored_label = _clean_text(cluster_row.get("calle_cercana"))
    if stored_label and not _needs_resolution(stored_label):
        return stored_label

    lat_raw = cluster_row.get("latitud")
    lon_raw = cluster_row.get("longitud")
    alcaldia = _clean_text(cluster_row.get("alcaldia_comun"))

    try:
        lat = float(lat_raw)
        lon = float(lon_raw)
    except (TypeError, ValueError):
        lat = lon = None

    if lat is not None and lon is not None:
        colonia, colonia_alcaldia = _lookup_colonia(round(lat, 5), round(lon, 5))
        if colonia:
            parent = alcaldia or colonia_alcaldia
            colonia_label = colonia.title()
            return f"Col. {colonia_label}" + (f" ({parent})" if parent else "")

    cluster_id = cluster_row.get("cluster_id")
    if alcaldia:
        if cluster_id is not None and not pd.isna(cluster_id):
            try:
                cluster_num = int(cluster_id) + 1
            except (TypeError, ValueError):
                cluster_num = cluster_id
            return f"Cluster {cluster_num} - {alcaldia}"
        return alcaldia

    if lat is not None and lon is not None:
        return f"Lat {lat:.4f}, Lon {lon:.4f}"

    return "Ubicación sin referencia"


def enrich_cluster_locations(df_clusters: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Returns a copy of df_clusters with an extra `cluster_label` column that contains
    human friendly location text for each centroid.
    """
    if df_clusters is None or df_clusters.empty:
        return df_clusters

    df = df_clusters.copy()
    df["cluster_label"] = df.apply(build_cluster_label, axis=1)
    return df
