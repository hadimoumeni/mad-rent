"""
Feature engineering: raw scraped listings -> clean modelling matrices.

Produces two related but distinct views of every listing:

  * CHARACTERISTIC space (price-independent): size, layout, condition,
    amenities and the spatial signal from spatial.py. Used to find genuine
    "comparables" (a KNN neighbourhood) and to measure structural rarity —
    without letting price leak in.

  * FULL space (characteristics + price-per-m²): fed to the isolation forest so
    it can isolate listings whose *price* does not fit their characteristics.

Both are z-scored. The characteristic space is additionally weighted so that
size and location dominate what "comparable" means.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from spatial import add_spatial_features

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Amenity booleans present on every record (scraper guarantees the columns).
AMENITY_COLS = [
    "has_elevator", "has_air_conditioning", "is_furnished", "has_terrace",
    "has_garden", "has_pool", "has_garage", "has_storage_room", "has_ensuite",
    "has_equipped_kitchen", "has_balcony", "has_wardrobes",
]
FLAG_COLS = ["is_exterior", "is_modern"]

# Every boolean-ish column, coerced robustly regardless of how the CSV parsed it.
BOOL_COLS = AMENITY_COLS + FLAG_COLS + [
    "has_video", "has_floorplan", "is_opportunity", "is_new_construction",
]

# Numeric characteristics (no price). Weights tune what "comparable" means:
# comparability is driven mostly by size and where the flat is.
CHAR_NUM_WEIGHTS = {
    "log_surface": 2.2,
    "rooms": 1.6,
    "bathrooms": 1.0,
    "floor_code": 0.7,
    "conservation_code": 0.7,
    "antiquity_code": 0.5,
    "dist_metro_m": 1.0,
    "metro_within_500m": 0.6,
    "dist_park_m": 0.6,
    "dist_university_m": 0.5,
    "dist_center_km": 1.6,
    "noise_index": 0.8,
    "lat": 2.2,
    "lon": 2.2,
}
CHAR_BIN_WEIGHT = 0.5   # each amenity/flag contributes half a z-unit of spread


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "listings.csv")

    # Fill structural fields.
    df["rooms"] = pd.to_numeric(df["rooms"], errors="coerce").fillna(0).clip(0, 8)
    df["bathrooms"] = pd.to_numeric(df["bathrooms"], errors="coerce").fillna(1).clip(1, 6)
    for c in ["floor_code", "conservation_code", "antiquity_code"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())

    df["price_per_m2"] = df["price"] / df["surface"]
    # m² per room: flags coliving / whole-building listings mislabelled as one
    # flat (e.g. "8 rooms, 6 baths, 15 m²"), which are a different product.
    df["m2_per_room"] = df["surface"] / df["rooms"].clip(lower=1)

    # Drop implausible records (data-entry errors, non-standard products) but
    # keep a wide band so genuine bargains/rip-offs survive.
    before = len(df)
    df = df[
        df["price"].between(300, 20000)
        & df["surface"].between(15, 500)
        & df["price_per_m2"].between(4, 80)
        & df["rooms"].le(8)
        & ~((df["rooms"] >= 3) & (df["m2_per_room"] < 8))  # impossible density
    ].copy()
    dropped = before - len(df)
    if dropped:
        print(f"[features] dropped {dropped} implausible rows")

    df = add_spatial_features(df)

    df["log_price"] = np.log(df["price"])
    df["log_surface"] = np.log(df["surface"])
    df["log_price_per_m2"] = np.log(df["price_per_m2"])

    # Robust bool coercion: handles native bool, 0/1, or "True"/"False" strings.
    for c in BOOL_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower().isin(
                {"true", "1", "1.0"}).astype(float)

    df = df.reset_index(drop=True)
    return df


def _zscore(a: np.ndarray) -> np.ndarray:
    mu = a.mean(axis=0)
    sd = a.std(axis=0)
    sd[sd == 0] = 1.0
    return (a - mu) / sd


def build_matrices(df: pd.DataFrame):
    """
    Returns
    -------
    char_matrix : (N, D1) weighted, z-scored characteristic space (no price)
    full_matrix : (N, D2) z-scored characteristics + price-per-m² (for the iForest)
    info : dict of the column names used in each space
    """
    num_cols = list(CHAR_NUM_WEIGHTS.keys())
    bin_cols = AMENITY_COLS + FLAG_COLS

    num_z = _zscore(df[num_cols].to_numpy(float))
    # apply comparability weights
    weights = np.array([CHAR_NUM_WEIGHTS[c] for c in num_cols])
    num_zw = num_z * weights

    bin_z = _zscore(df[bin_cols].to_numpy(float)) * CHAR_BIN_WEIGHT

    char_matrix = np.hstack([num_zw, bin_z])

    # Full space for the isolation forest: unweighted z-scores of the same
    # characteristics PLUS price-per-m². (Unweighted because the iForest picks
    # split features at random and should treat them evenhandedly.)
    price_z = _zscore(df[["log_price_per_m2"]].to_numpy(float))
    full_matrix = np.hstack([
        _zscore(df[num_cols].to_numpy(float)),
        _zscore(df[bin_cols].to_numpy(float)),
        price_z,
    ])

    info = {
        "num_cols": num_cols,
        "bin_cols": bin_cols,
        "char_cols": num_cols + bin_cols,
        "full_cols": num_cols + bin_cols + ["log_price_per_m2"],
    }
    return char_matrix, full_matrix, info


if __name__ == "__main__":
    df = load_and_clean()
    char, full, info = build_matrices(df)
    print(f"clean listings: {len(df)}")
    print(f"characteristic matrix: {char.shape}  |  full matrix: {full.shape}")
    print(f"districts: {df['district'].nunique()}  "
          f"({', '.join(df['district'].value_counts().head(6).index)}...)")
    print("price/m² €:", df["price_per_m2"].describe()[["min", "50%", "max"]].round(1).to_dict())
    print("median dist to metro (m):", round(df["dist_metro_m"].median(), 0))
