# Madrid Rental Price Anomaly Detector

Find rentals in Madrid that are **genuinely mispriced** — not merely cheap or
expensive, but anomalous given their specific combination of location, size,
condition and features. The pipeline scrapes live listings, engineers spatial
and structural features, runs an **isolation forest implemented from scratch**,
and produces a ranked list of under- and over-priced flats with explanations,
plus the features the market systematically misprices.

![pipeline](https://img.shields.io/badge/isolation_forest-from_scratch-2a78d6)
![data](https://img.shields.io/badge/data-live_Fotocasa-1baf7a)

---

## What it does

1. **Scrapes** live Madrid rental listings from Fotocasa (Idealista is behind
   DataDome and returns HTTP 403 to non-browser clients; Fotocasa serves its
   results as an embedded JSON blob and is scrapeable).
2. **Engineers location** properly — haversine proximity to 243 real metro
   stations (from Wikidata), 15 parks, 10 universities, a nightlife-noise
   kernel, and centrality — not just raw latitude/longitude.
3. **Scores anomalies** with an isolation forest built from first principles
   (random split trees, path-length averaging, `c(n)` normalisation).
4. **Prices fairly** by finding each flat's nearest comparables in a
   *price-independent* feature space and measuring the signed residual.
5. **Explains mispricing** at two levels: individual listings (why *this* flat
   is a bargain / rip-off) and the market (which *features* it systematically
   under- or over-values), via a from-scratch hedonic regression.

The headline output is `output/report.html` — a self-contained visual report —
plus ranked CSVs.

---

## Results from the latest run

`587` clean listings across **all 21 Madrid districts** (scraped from Fotocasa),
hedonic model R² = **0.81**.

- **Bargains cluster in the periphery.** The strongest underpriced flags are
  large flats in Fuencarral-El Pardo, Usera and Villaverde renting 27–39% below
  comparable properties (e.g. a 134 m² 3-bed in Peñagrande at €1,800/mo vs a
  ~€2,960 comparable, −2.3σ).
- **Overpricing concentrates in the tourist core.** The strongest overpriced
  flags are small central flats in Sol / Lavapiés asking +100–120% over
  comparables (e.g. a 60 m² 1-bed in Sol at €3,295/mo vs ~€1,500, +7.2σ).
- **Systematic feature mispricing.** The market **overpays** for cosmetic
  signals — "modern" (+€186/mo implicit), A/C (+€123), furnished (+€108) — and
  **underpays** for durable value: a garage appears 61% more often among
  under-priced flats than in the market, a terrace 46% more. In other words,
  Fotocasa's ranking rewards how a flat is *presented* more than what it *has*.

(Figures move as inventory changes; re-run to refresh.)

---

## Quick start

```bash
pip install numpy pandas requests            # the only dependencies

# 1) scrape fresh data (~15 min, polite 4.5s/request), then analyse + report:
python run_pipeline.py --scrape

# or, if data/listings.csv already exists, just analyse + build the report:
python run_pipeline.py
```

Outputs land in `output/`:

| file | contents |
|------|----------|
| `report.html` | self-contained visual report (open in any browser) |
| `underpriced.csv` | ranked bargains with explanations |
| `overpriced.csv` | ranked overpriced listings |
| `all_scored.csv` | every listing with anomaly score, residual, verdict |
| `hedonic_coefficients.csv` | implicit €/month the market pays per feature |
| `feature_mispricing.csv` | features the market systematically misprices |

---

## How the isolation forest works (from scratch)

`src/isolation_forest.py` — no scikit-learn.

Anomalies are *few and different*, so they are easy to isolate. Build a random
binary tree by repeatedly picking a **random feature** and a **random split
value** between that feature's min and max in the node. Anomalous points get
cut off from the pack after only a few splits (a **short path** from the root);
normal points sit in dense regions and need many splits. Average the path
length over many random trees:

- Path length correction at leaves that still hold >1 point:
  `c(n) = 2·H(n−1) − 2(n−1)/n`
- Anomaly score: `s(x) = 2^( −E[h(x)] / c(ψ) )`, where `ψ` is the sub-sample size.
  `s → 1` means strongly anomalous; `s ≈ 0.5` means indistinguishable from normal.

Each tree is grown on a sub-sample of `ψ = 256` points (the paper's key insight:
small sub-samples reduce swamping/masking) with height limit `⌈log₂ ψ⌉`.

Run its self-test:

```bash
python src/isolation_forest.py     # plants outliers in a Gaussian blob, recovers them
```

---

## Turning "mispriced" into something defensible

An isolation forest alone tells you *what is unusual*, not *whether it is
under- or over-priced*, and it will happily flag a genuinely rare luxury flat.
So three signals are combined:

| signal | question it answers | how |
|--------|--------------------|-----|
| **Isolation forest** | is this listing unusual *overall*? | trees over characteristics **+** price/m² |
| **Comparables residual** | is the price off vs *similar* flats, and which way? | median price/m² of the 40 nearest neighbours in a **price-free** feature space; robust (MAD) z-score |
| **Hedonic OLS** | what does the market pay per feature, and which does it misprice? | log-price regression from scratch, with t-stats |

A listing is flagged **underpriced / overpriced** only when it is *both*
statistically anomalous *and* clearly off its comparables *and* its comparables
are trustworthy (it isn't sitting alone in feature space). Anomalous flats with
unreliable comparables are labelled **structural anomalies** instead — unusual,
but not a mispricing.

**Systematic feature mispricing.** Among flagged listings, features that cluster
with *underpriced* flats are ones the market underpays for (hidden value);
features that cluster with *overpriced* flats are ones it overpays for (hype).
The hedonic table shows the implicit €/month the market attaches to each feature.

---

## Project layout

```
madrid-rental-anomaly/
├── run_pipeline.py            # orchestrator
├── src/
│   ├── scrape_fotocasa.py     # breadth-first zone sweep, JSON-blob parser
│   ├── spatial.py             # haversine POI proximity features
│   ├── features.py            # cleaning + characteristic / full matrices
│   ├── isolation_forest.py    # the algorithm, from scratch
│   ├── analyze.py             # iForest + comparables + hedonic + verdicts
│   └── report.py              # inline-SVG HTML report (no JS deps)
├── data/
│   ├── madrid_poi.json        # 243 metro stations, parks, universities, nightlife
│   ├── listings.csv           # scraped listings
│   └── listings_raw.jsonl
└── output/                    # scores, rankings, report.html
```

Each module runs standalone (`python src/<module>.py`) for a quick self-test.

---

## Honesty notes & caveats

- **Asking prices, not transacted rents.** We observe what landlords ask, which
  already embeds their own (sometimes wrong) pricing — that is exactly what we
  probe, but it is not ground truth.
- **Fotocasa, not Idealista.** Idealista blocks automated access (DataDome / 403).
  The method is identical for any source with size, location and features.
- **Comparable sets are thin in low-inventory districts**, so residuals there are
  noisier — hence the comp-reliability gate.
- **Condition / antiquity are Fotocasa coded ordinals**, treated as ordered
  numbers; park distance uses centroids.
- Flags are **leads to inspect**, not verdicts. A 30%-below-comps flat may be a
  ground-floor interior unit the features don't capture — or a genuine bargain.

Built as a one-day project: scraper + from-scratch isolation forest + spatial
feature engineering + ranked, explained output.
