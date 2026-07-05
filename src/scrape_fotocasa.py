"""
Fotocasa Madrid rental scraper.

Fotocasa (unlike Idealista, which sits behind DataDome and returns 403 to bots)
serves its search results as a server-rendered JSON blob embedded in the page:

    <script id="__initial_props__" type="application/json"> {...} </script>

The listings live at initialSearch.result.realEstates. Each carries coordinates,
district / neighborhood, surface, rooms, bathrooms, floor, conservation status,
antiquity, orientation and a rich `features` array of amenities. We paginate the
capital-wide rental search, normalise each listing to a flat record, deduplicate
by id, and write JSONL + CSV.

Politeness: one request every ~2.5 s (with jitter), a real browser UA, capped
page count, and graceful back-off on non-200 responses. This is a read-only
scrape of publicly listed data for a personal analytics project.
"""

from __future__ import annotations

import csv
import json
import random
import re
import sys
import time
from pathlib import Path

import requests

BASE = "https://www.fotocasa.es/es/alquiler/viviendas/madrid-capital"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# One search slug per official Madrid district (Salamanca's district slug 404s,
# so we use its main barrio). District-level pages carry the whole district's
# inventory, so 21 slugs cover the entire city in far fewer requests than
# sweeping ~150 barrios — which matters because Fotocasa soft-blocks on request
# bursts. Ordered rich-core -> periphery for price-spectrum coverage.
ZONES = [
    "centro", "barrio-de-salamanca", "chamberi", "retiro", "arganzuela",
    "chamartin", "tetuan", "moncloa-aravaca", "fuencarral-el-pardo",
    "ciudad-lineal", "hortaleza", "latina", "carabanchel", "usera",
    "puente-de-vallecas", "moratalaz", "villaverde", "villa-de-vallecas",
    "vicalvaro", "san-blas", "barajas",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "Cache-Control": "no-cache",
    "Referer": "https://www.fotocasa.es/es/alquiler/viviendas/madrid-capital/todas-las-zonas/l",
}

PROPS_RE = re.compile(r'id="__initial_props__"[^>]*>(.*?)</script>', re.S)

# Feature keys whose numeric value is a real measurement.
NUMERIC_FEATURES = {"surface", "rooms", "bathrooms", "floor",
                    "conservationStatus", "antiquity", "orientation", "heating"}

# Amenities we treat as booleans (present in the features array == has it).
AMENITY_KEYS = {
    "elevator": "has_elevator",
    "air_conditioner": "has_air_conditioning",
    "furnished": "is_furnished",
    "terrace": "has_terrace",
    "garden": "has_garden",
    "pool": "has_pool",
    "swimming_pool": "has_pool",
    "garage": "has_garage",
    "parking": "has_garage",
    "storage": "has_storage_room",
    "storage_room": "has_storage_room",
    "ensuite_bathroom": "has_ensuite",
    "armored_door": "has_armored_door",
    "equiped_kitchen": "has_equipped_kitchen",
    "balcony": "has_balcony",
    "wardrobe": "has_wardrobes",
    "cabinets": "has_wardrobes",
}


def fetch_page(session: requests.Session, zone: str, page: int) -> tuple[str | None, int]:
    url = f"{BASE}/{zone}/l" if page == 1 else f"{BASE}/{zone}/l/{page}"
    try:
        r = session.get(url, headers=HEADERS, timeout=25)
    except requests.RequestException as exc:
        print(f"    {zone} p{page}: request error {exc}")
        return None, 0
    if r.status_code != 200:
        return None, r.status_code
    return r.text, 200


def parse_listings(html: str) -> list[dict]:
    m = PROPS_RE.search(html)
    if not m:
        return []
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return []
    return data.get("initialSearch", {}).get("result", {}).get("realEstates", []) or []


def normalise(raw: dict) -> dict | None:
    """Flatten one Fotocasa listing into a modelling record, or None if unusable."""
    # Keep only monthly residential rentals.
    if raw.get("transactionTypeId") != 3 or raw.get("periodicityId") != 3:
        return None
    if raw.get("isTemporaryRental"):
        return None

    price = raw.get("rawPrice")
    if not price or price <= 0:
        return None

    feats = {f["key"]: f["value"] for f in raw.get("features", []) if "key" in f}
    surface = feats.get("surface")
    if not surface or surface <= 0:
        return None

    addr = raw.get("address", {}) or {}
    coord = raw.get("coordinates", {}) or {}
    lat, lon = coord.get("latitude"), coord.get("longitude")
    if lat is None or lon is None:
        return None

    dyn = set(raw.get("dynamicFeatures") or [])
    detail = (raw.get("detail") or {}).get("es-ES", "")

    rec = {
        "id": raw.get("id"),
        "url": f"https://www.fotocasa.es{detail}" if detail else None,
        "district": addr.get("district"),
        "neighborhood": addr.get("neighborhood"),
        "zip_code": addr.get("zipCode"),
        "lat": lat,
        "lon": lon,
        "price": float(price),                       # € / month
        "building_subtype": raw.get("buildingSubtype"),
        "surface": float(surface),                   # m²
        "rooms": feats.get("rooms"),
        "bathrooms": feats.get("bathrooms"),
        "floor_code": feats.get("floor"),
        "conservation_code": feats.get("conservationStatus"),
        "antiquity_code": feats.get("antiquity"),
        "orientation_code": feats.get("orientation"),
        "heating_code": feats.get("heating"),
        "is_exterior": "IS_EXTERIOR" in dyn,
        "is_modern": "IS_MODERN" in dyn,
        "n_photos": len(raw.get("multimedia") or []),
        "has_video": bool(raw.get("hasVideo")),
        "has_floorplan": bool(raw.get("hasFloorPlans")),
        "is_new_construction": bool(raw.get("isNewConstruction")),
        "is_opportunity": bool(raw.get("isOpportunity")),
        "desc_len": len(raw.get("description") or ""),
        "date_diff_days": (raw.get("date") or {}).get("diff"),
    }
    for key, col in AMENITY_KEYS.items():
        if key in feats:
            rec[col] = True
    for col in set(AMENITY_KEYS.values()):
        rec.setdefault(col, False)
    return rec


def _ingest(listings, seen, records, zone) -> int:
    added = 0
    for raw in listings:
        rid = raw.get("id")
        if rid in seen:
            continue
        rec = normalise(raw)
        if rec is None:
            continue
        rec["zone_scraped"] = zone
        seen.add(rid)
        records.append(rec)
        added += 1
    return added


def scrape(zones: list[str] = ZONES, max_pages: int = 3,
           delay: float = 7.0, target: int = 1600,
           burst: int = 5, silence: float = 100.0) -> list[dict]:
    """
    Burst-and-silence sweep, breadth-first (page 1 of every district, then page
    2, …). Fotocasa soft-blocks on request *bursts* and resets after a minute or
    two of quiet — and crucially, retrying *during* a block just refreshes the
    limit window and keeps it alive. So the rule is: make a short burst of
    requests, then go genuinely silent; and if a response comes back empty
    (blocked), stop hitting it, wait out a full silence window, and move on. A
    single shared session is kept — fresh sessions get blocked immediately.
    """
    session = requests.Session()
    seen: set = set()
    records: list[dict] = []
    dead: set = set()                      # 404 or genuinely exhausted
    since_break = 0

    def take_break(secs):
        nonlocal since_break
        print(f"    …quiet for {secs:.0f}s (let the rate limit reset)")
        time.sleep(secs)
        since_break = 0

    for page in range(1, max_pages + 1):
        live_this_page = 0
        for zone in zones:
            if zone in dead:
                continue
            if since_break >= burst:       # burst done -> go silent
                take_break(silence + random.uniform(0, 10))

            html, code = fetch_page(session, zone, page)
            since_break += 1

            if html is None and code == 404:
                dead.add(zone)
                print(f"  p{page} {zone:<22} 404 (no such district slug)")
                continue

            listings = parse_listings(html) if html else []
            if not listings:
                # Blocked (or genuinely empty). Do NOT hammer — a long silence
                # is the only thing that clears a block. Retry the zone once,
                # after the silence, on this same pass.
                print(f"  p{page} {zone:<22} empty/blocked")
                take_break(silence + random.uniform(0, 10))
                html, _ = fetch_page(session, zone, page)
                since_break += 1
                listings = parse_listings(html) if html else []
                if not listings:
                    if page > 1:
                        dead.add(zone)
                    time.sleep(delay)
                    continue

            added = _ingest(listings, seen, records, zone)
            live_this_page += 1
            if added == 0 and page > 1:
                dead.add(zone)
            print(f"  p{page} {zone:<22} +{added:>3}  (total {len(records)})")

            if len(records) >= target:
                print(f"  reached target {target} — stopping.")
                return records
            time.sleep(delay + random.uniform(0, 1.5))
        print(f"  --- page {page}: {live_this_page} districts returned data "
              f"({len(records)} listings) ---")

    return records


def save(records: list[dict]) -> None:
    """Merge with any previously scraped listings (dedup by id) so partial runs
    accumulate coverage instead of clobbering each other."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    jsonl = DATA_DIR / "listings_raw.jsonl"

    merged: dict = {}
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                merged[r["id"]] = r
    new = 0
    for r in records:
        if r["id"] not in merged:
            new += 1
        merged[r["id"]] = r
    allrecs = list(merged.values())

    with jsonl.open("w", encoding="utf-8") as f:
        for r in allrecs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if allrecs:
        cols = list({k for r in allrecs for k in r})
        # stable-ish column order: first record's keys, then any extras
        ordered = list(allrecs[0].keys()) + [c for c in cols if c not in allrecs[0]]
        csv_path = DATA_DIR / "listings.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ordered)
            w.writeheader()
            for r in allrecs:
                w.writerow(r)
    print(f"\nThis run added {new} new listings; dataset now {len(allrecs)} "
          f"total -> {jsonl.name} + listings.csv")


if __name__ == "__main__":
    mp = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    pause = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
    print(f"Scraping Fotocasa Madrid rentals (burst-and-silence) across "
          f"{len(ZONES)} districts: up to {mp} pages, {pause}s delay\n")
    recs = scrape(zones=ZONES, max_pages=mp, delay=pause)
    save(recs)
    if recs:
        import statistics
        prices = [r["price"] for r in recs]
        print(f"Price €/mo: min {min(prices):.0f}  median "
              f"{statistics.median(prices):.0f}  max {max(prices):.0f}")
        print(f"Districts: {len({r['district'] for r in recs})} distinct")
