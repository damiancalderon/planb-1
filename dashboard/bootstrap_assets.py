from __future__ import annotations

import gzip
import shutil
from pathlib import Path
from typing import List


BASE_PATH = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_PATH / "artifacts"

REQUIRED_ASSETS = [
    {
        "target": BASE_PATH / "cdmx_insights.db",
        "archive": ARTIFACTS_DIR / "cdmx_insights.db.gz",
        "description": "DuckDB con la tabla crimes",
    },
    {
        "target": BASE_PATH / "crime_forecaster.pkl",
        "archive": ARTIFACTS_DIR / "crime_forecaster.pkl.gz",
        "description": "Modelo SARIMA para el pronóstico diario",
    },
]


def _decompress_gzip(archive_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(archive_path, "rb") as src, open(target_path, "wb") as dst:
        shutil.copyfileobj(src, dst)


def ensure_assets() -> List[Path]:
    """
    Verifica que los artefactos críticos existan. Si faltan, se extraen
    de sus contrapartes comprimidas en dashboard/artifacts/.
    """
    materialized: List[Path] = []
    for asset in REQUIRED_ASSETS:
        target = asset["target"]
        archive = asset["archive"]
        if target.exists():
            continue
        if not archive.exists():
            raise FileNotFoundError(
                f"No se encuentra {archive.name}. "
                "Ejecuta load_data.py y train_model.py para regenerarlo."
            )
        _decompress_gzip(archive, target)
        materialized.append(target)
    return materialized


if __name__ == "__main__":
    created = ensure_assets()
    if created:
        print("Artefactos restaurados:")
        for path in created:
            print(f" - {path}")
    else:
        print("Todos los artefactos requeridos ya estaban disponibles.")
