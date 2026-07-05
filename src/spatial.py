"""
Spatial feature engineering.

Turns a raw (lat, lon) into the location signal that actually drives Madrid
rents: how close is the flat to a metro entrance, to green space, to a
university, to the city centre, and how much nightlife noise surrounds it.

All distances are great-circle (haversine) metres against the real POI
coordinates in data/madrid_poi.json (243 metro stations from Wikidata, plus
curated parks / universities / nightlife hotspots).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

POI_PATH = Path(__file__).resolve().parent.parent / "data" / "madrid_poi.json"
EARTH_R = 6_371_000.0  # metres


def _load_poi() -> dict:
    return json.loads(POI_PATH.read_text(encoding="utf-8"))


def haversine_matrix(lat1, lon1, lat2, lon2) -> np.ndarray:
    """
    Pairwise great-circle distance (metres) between two sets of points.

    lat1/lon1: (N,) listing coords. lat2/lon2: (M,) POI coords.
    Returns an (N, M) matrix. Broadcasting keeps it fully vectorised.
    """
    lat1 = np.radians(np.asarray(lat1, float))[:, None]
    lon1 = np.radians(np.asarray(lon1, float))[:, None]
    lat2 = np.radians(np.asarray(lat2, float))[None, :]
    lon2 = np.radians(np.asarray(lon2, float))[None, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def _poi_coords(items: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    return (np.array([p["lat"] for p in items]),
            np.array([p["lon"] for p in items]))


def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach location features to a DataFrame that has `lat` and `lon`."""
    poi = _load_poi()
    df = df.copy()
    lat, lon = df["lat"].to_numpy(), df["lon"].to_numpy()

    # --- Metro: nearest distance + accessibility density ---
    m_lat, m_lon = _poi_coords(poi["metro_stations"])
    m_names = np.array([s["name"] for s in poi["metro_stations"]])
    d_metro = haversine_matrix(lat, lon, m_lat, m_lon)          # (N, 243)
    df["dist_metro_m"] = d_metro.min(axis=1)
    df["nearest_metro"] = m_names[d_metro.argmin(axis=1)]
    df["metro_within_500m"] = (d_metro < 500).sum(axis=1)
    df["metro_within_1km"] = (d_metro < 1000).sum(axis=1)

    # --- Parks (centroid distance is an approximation of "walk to greenery") ---
    p_lat, p_lon = _poi_coords(poi["parks"])
    d_park = haversine_matrix(lat, lon, p_lat, p_lon)
    df["dist_park_m"] = d_park.min(axis=1)

    # --- Universities ---
    u_lat, u_lon = _poi_coords(poi["universities"])
    d_uni = haversine_matrix(lat, lon, u_lat, u_lon)
    df["dist_university_m"] = d_uni.min(axis=1)

    # --- Centrality: straight-line km to Puerta del Sol ---
    c = poi["center"]
    df["dist_center_km"] = (
        haversine_matrix(lat, lon, [c["lat"]], [c["lon"]]).ravel() / 1000.0
    )

    # --- Nightlife-noise proxy: sum of Gaussian kernels over hotspots.
    #     Scale = 600 m; being near several nightlife zones stacks up. Higher
    #     value == noisier / livelier surroundings. ---
    n_lat, n_lon = _poi_coords(poi["nightlife"])
    d_night = haversine_matrix(lat, lon, n_lat, n_lon)
    df["noise_index"] = np.exp(-(d_night / 600.0) ** 2).sum(axis=1)

    return df


if __name__ == "__main__":
    # Sanity check on a few known Madrid points.
    probes = pd.DataFrame([
        {"name": "Puerta del Sol (dead centre)", "lat": 40.4169, "lon": -3.7033},
        {"name": "Near Retiro park",             "lat": 40.4150, "lon": -3.6830},
        {"name": "Cantoblanco (far north/UAM)",  "lat": 40.5450, "lon": -3.6900},
    ])
    out = add_spatial_features(probes)
    cols = ["name", "dist_metro_m", "metro_within_500m", "dist_park_m",
            "dist_university_m", "dist_center_km", "noise_index"]
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(out[cols].round(1).to_string(index=False))
