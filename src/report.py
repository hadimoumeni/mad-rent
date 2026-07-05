"""
Static HTML report builder for the Madrid rental anomaly detector.

Dependency-free: every chart is inline SVG generated here (no JS libraries, no
matplotlib), so output/report.html is fully self-contained and opens anywhere.
Colours follow a validated, colourblind-safe palette; the diverging blue↔red
axis encodes under- vs over-pricing, which is the report's whole story.
"""

from __future__ import annotations

import html
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"

# --- validated palette (see dataviz skill) ---
C_UNDER = "#2a78d6"   # blue  — underpriced (good deal)
C_OVER = "#e34948"    # red   — overpriced
C_FAIR = "#b8b6ae"    # gray  — fairly priced
C_STRUCT = "#eda100"  # yellow — structural anomaly (unusual, not mispriced)


def _esc(x) -> str:
    return html.escape(str(x))


# --------------------------------------------------------------------------- #
#  SVG primitives
# --------------------------------------------------------------------------- #
def scatter_expected_vs_actual(df: pd.DataFrame) -> str:
    """Log-log scatter: expected (comparable) price vs actual asking price."""
    W, H = 720, 460
    pad_l, pad_b, pad_t, pad_r = 64, 52, 24, 20
    x = df["expected_price"].clip(lower=1)
    y = df["price"].clip(lower=1)
    lo = math.log10(min(x.min(), y.min()) * 0.9)
    hi = math.log10(max(x.max(), y.max()) * 1.1)

    def sx(v):
        return pad_l + (math.log10(v) - lo) / (hi - lo) * (W - pad_l - pad_r)

    def sy(v):
        return H - pad_b - (math.log10(v) - lo) / (hi - lo) * (H - pad_b - pad_t)

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" '
             f'aria-label="Expected versus actual monthly rent" '
             f'style="width:100%;height:auto">']

    # gridlines + ticks at 500,1000,2000,5000,10000
    for tick in [500, 1000, 2000, 5000, 10000]:
        if lo <= math.log10(tick) <= hi:
            gx, gy = sx(tick), sy(tick)
            parts.append(f'<line x1="{gx:.1f}" y1="{pad_t}" x2="{gx:.1f}" '
                         f'y2="{H-pad_b}" class="grid"/>')
            parts.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{W-pad_r}" '
                         f'y2="{gy:.1f}" class="grid"/>')
            parts.append(f'<text x="{gx:.1f}" y="{H-pad_b+16}" class="tick" '
                         f'text-anchor="middle">€{tick:,}</text>')
            parts.append(f'<text x="{pad_l-8}" y="{gy+4:.1f}" class="tick" '
                         f'text-anchor="end">€{tick:,}</text>')

    # y = x reference line (fair price)
    p1x, p1y = sx(10 ** lo), sy(10 ** lo)
    p2x, p2y = sx(10 ** hi), sy(10 ** hi)
    parts.append(f'<line x1="{p1x:.1f}" y1="{p1y:.1f}" x2="{p2x:.1f}" '
                 f'y2="{p2y:.1f}" class="refline"/>')
    parts.append(f'<text x="{sx(10**hi)-6:.1f}" y="{sy(10**hi)-6:.1f}" '
                 f'class="tick" text-anchor="end">fair (actual = comparable)</text>')

    color = {"underpriced": C_UNDER, "overpriced": C_OVER,
             "structural_anomaly": C_STRUCT, "fair": C_FAIR}
    # plot fair first (background), flagged on top
    order = df.sort_values("verdict", key=lambda s: s.map(
        {"fair": 0, "structural_anomaly": 1, "overpriced": 2, "underpriced": 3}))
    for _, r in order.iterrows():
        c = color.get(r["verdict"], C_FAIR)
        rad = 3.2 if r["verdict"] in ("fair", "structural_anomaly") else 4.2
        op = 0.45 if r["verdict"] == "fair" else 0.9
        parts.append(f'<circle cx="{sx(r["expected_price"]):.1f}" '
                     f'cy="{sy(r["price"]):.1f}" r="{rad}" fill="{c}" '
                     f'fill-opacity="{op}" stroke="var(--surface)" '
                     f'stroke-width="0.8"><title>{_esc(r["neighborhood"])}: '
                     f'€{r["price"]:.0f} vs €{r["expected_price"]:.0f} '
                     f'expected ({r["residual_pct"]*100:+.0f}%)</title></circle>')

    parts.append(f'<text x="{(pad_l+W-pad_r)/2:.0f}" y="{H-8}" class="axis-title" '
                 f'text-anchor="middle">Comparable (expected) rent  →</text>')
    parts.append(f'<text transform="translate(16,{(pad_t+H-pad_b)/2:.0f}) '
                 f'rotate(-90)" class="axis-title" text-anchor="middle">'
                 f'Actual asking rent  →</text>')
    parts.append("</svg>")
    return "".join(parts)


def diverging_bars(rows, value_key, label_fn, sub_fn, color) -> str:
    """Horizontal bars for a ranked list (residual %)."""
    n = len(rows)
    if n == 0:
        return "<p class='muted'>None found.</p>"
    rowh, W = 34, 720
    left = 250
    H = n * rowh + 10
    vmax = max(abs(r[value_key]) for r in rows) or 1
    parts = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto" role="img">']
    for i, r in enumerate(rows):
        y = i * rowh + 6
        v = r[value_key]
        w = abs(v) / vmax * (W - left - 90)
        parts.append(f'<text x="0" y="{y+14}" class="barlab">{_esc(label_fn(r))}</text>')
        parts.append(f'<text x="0" y="{y+27}" class="barsub">{_esc(sub_fn(r))}</text>')
        parts.append(f'<rect x="{left}" y="{y+4}" width="{w:.1f}" height="16" '
                     f'rx="4" fill="{color}" fill-opacity="0.85"/>')
        parts.append(f'<text x="{left+w+6:.1f}" y="{y+16}" class="barval">'
                     f'{v*100:+.0f}%</text>')
    parts.append("</svg>")
    return "".join(parts)


def feature_mispricing_chart(fm: pd.DataFrame) -> str:
    """Diverging bars: features the market underpays (blue) vs overpays (red) for."""
    fm = fm.copy()
    fm = fm[fm["prevalence_all"] >= 0.03]
    fm = fm.reindex(fm["net_skew"].abs().sort_values(ascending=False).index).head(12)
    if fm.empty:
        return "<p class='muted'>Not enough flagged listings to profile features.</p>"
    rowh, W, mid = 30, 720, 380
    H = len(fm) * rowh + 24
    vmax = fm["net_skew"].abs().max() or 1
    span = W - mid - 120
    parts = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto" role="img">']
    parts.append(f'<line x1="{mid}" y1="18" x2="{mid}" y2="{H-6}" class="baseline"/>')
    parts.append(f'<text x="{mid-4}" y="14" class="tick" text-anchor="end">'
                 f'underpays for →</text>')
    parts.append(f'<text x="{mid+4}" y="14" class="tick">← overpays for</text>')
    for i, (_, r) in enumerate(fm.iterrows()):
        y = i * rowh + 24
        v = r["net_skew"]
        w = abs(v) / vmax * span
        parts.append(f'<text x="0" y="{y+13}" class="barlab">'
                     f'{_esc(_pretty(r["feature"]))}</text>')
        if v >= 0:   # over-represented among underpriced -> market underpays
            parts.append(f'<rect x="{mid-w:.1f}" y="{y+2}" width="{w:.1f}" '
                         f'height="15" rx="4" fill="{C_UNDER}" fill-opacity="0.85"/>')
        else:
            parts.append(f'<rect x="{mid}" y="{y+2}" width="{w:.1f}" '
                         f'height="15" rx="4" fill="{C_OVER}" fill-opacity="0.85"/>')
    parts.append("</svg>")
    return "".join(parts)


def anomaly_histogram(df: pd.DataFrame) -> str:
    W, H = 720, 200
    pad_l, pad_b, pad_t, pad_r = 44, 34, 16, 12
    scores = df["anomaly_score"]
    bins = 26
    lo, hi = scores.min(), scores.max()
    counts = [0] * bins
    for s in scores:
        b = min(int((s - lo) / (hi - lo + 1e-9) * bins), bins - 1)
        counts[b] += 1
    cmax = max(counts) or 1
    bw = (W - pad_l - pad_r) / bins
    parts = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto" role="img">']
    for i, c in enumerate(counts):
        bh = c / cmax * (H - pad_b - pad_t)
        x = pad_l + i * bw
        centre = lo + (i + 0.5) / bins * (hi - lo)
        col = C_OVER if centre >= df["anomaly_score"].quantile(0.9) else "#7aa9e0"
        parts.append(f'<rect x="{x+1:.1f}" y="{H-pad_b-bh:.1f}" '
                     f'width="{bw-2:.1f}" height="{bh:.1f}" rx="2" fill="{col}"/>')
    parts.append(f'<line x1="{pad_l}" y1="{H-pad_b}" x2="{W-pad_r}" '
                 f'y2="{H-pad_b}" class="baseline"/>')
    for t in [lo, (lo+hi)/2, hi]:
        tx = pad_l + (t - lo) / (hi - lo + 1e-9) * (W - pad_l - pad_r)
        parts.append(f'<text x="{tx:.1f}" y="{H-pad_b+16}" class="tick" '
                     f'text-anchor="middle">{t:.2f}</text>')
    parts.append(f'<text x="{pad_l}" y="12" class="tick">← more normal      '
                 f'more anomalous →</text>')
    parts.append("</svg>")
    return "".join(parts)


def _pretty(f: str) -> str:
    return (f.replace("has_", "").replace("is_", "").replace("_", " ")
            .replace("air conditioning", "A/C").strip().capitalize())


# --------------------------------------------------------------------------- #
#  Page assembly
# --------------------------------------------------------------------------- #
def _stat(label, value, sub=""):
    return (f'<div class="stat"><div class="stat-v">{value}</div>'
            f'<div class="stat-l">{label}</div>'
            f'<div class="stat-s">{sub}</div></div>')


def _listing_card(r, klass):
    url = r.get("url")
    title = (f'{r["neighborhood"]} · {r["district"]}')
    link = (f'<a href="{_esc(url)}" target="_blank" rel="noopener">view ↗</a>'
            if isinstance(url, str) and url.startswith("http") else "")
    return (
        f'<div class="card {klass}">'
        f'<div class="card-h"><span class="card-t">{_esc(title)}</span>'
        f'<span class="card-p">€{r["price"]:.0f}/mo · {r["surface"]:.0f} m²</span></div>'
        f'<div class="card-b">{_esc(r["explanation"])}</div>'
        f'<div class="card-f"><span class="pill {klass}">'
        f'{r["residual_pct"]*100:+.0f}% vs comparables</span> '
        f'<span class="pill-2">anomaly {r["anomaly_score"]:.2f}</span> {link}</div>'
        f'</div>')


def render_body(data: dict) -> tuple[str, str]:
    df = data["all"]
    under = data["under"]
    over = data["over"]
    fm = data["feature_mispricing"]
    hed = data["hedonic"]
    meta = data["meta"]

    n_under = (df["verdict"] == "underpriced").sum()
    n_over = (df["verdict"] == "overpriced").sum()
    n_struct = (df["verdict"] == "structural_anomaly").sum()

    stats = "".join([
        _stat("listings analysed", f'{len(df):,}', f'{df["district"].nunique()} districts'),
        _stat("median rent", f'€{df["price"].median():.0f}',
              f'€{df["price_per_m2"].median():.1f}/m²'),
        _stat("underpriced", f'{n_under}', "genuine bargains"),
        _stat("overpriced", f'{n_over}', "asking too much"),
        _stat("structural", f'{n_struct}', "rare, price fair"),
    ])

    under_rows = under.to_dict("records")
    over_rows = over.to_dict("records")

    # hedonic table (top |effect|, binary features shown as €/mo)
    hed_bin = hed[hed["feature"].str.startswith(("has_", "is_"))].copy()
    hed_bin = hed_bin.reindex(hed_bin["eur_effect_at_median"].abs()
                              .sort_values(ascending=False).index).head(10)
    hed_rows = "".join(
        f'<tr><td>{_esc(_pretty(r["feature"]))}</td>'
        f'<td class="num">{r["eur_effect_at_median"]:+.0f}</td>'
        f'<td class="num">{r["pct_effect"]*100:+.0f}%</td>'
        f'<td class="num muted">{r["t_stat"]:+.1f}</td></tr>'
        for _, r in hed_bin.iterrows())

    body = f"""
<div class="viz-root">
<header>
  <div class="eyebrow">Rental price anomaly detection · Madrid</div>
  <h1>Which Madrid rentals are genuinely mispriced?</h1>
  <p class="lede">An isolation forest (built from scratch) over {len(df):,} live
  Fotocasa listings, combined with a comparable-based fair-price model and a
  hedonic regression. We flag listings whose price does not fit their size,
  location, condition and features — not merely the cheapest or dearest.</p>
</header>

<section class="stats">{stats}</section>

<section>
  <h2>The map of mispricing</h2>
  <p>Each dot is a flat: its comparable (fair) rent on the horizontal axis, its
  actual asking rent on the vertical. On the diagonal, price matches comparables.
  <b style="color:{C_UNDER}">Below</b> the line = underpriced;
  <b style="color:{C_OVER}">above</b> = overpriced. Only dots the isolation
  forest also rates anomalous are highlighted.</p>
  <div class="chart">{data["scatter"]}</div>
  <div class="legend">
    <span><i style="background:{C_UNDER}"></i>underpriced</span>
    <span><i style="background:{C_OVER}"></i>overpriced</span>
    <span><i style="background:{C_STRUCT}"></i>structural anomaly</span>
    <span><i style="background:{C_FAIR}"></i>fairly priced</span>
  </div>
</section>

<section>
  <h2>Top underpriced — the bargains</h2>
  <p>Ranked by combined mispricing score (distance from comparables × isolation-forest anomaly).</p>
  <div class="cards">{"".join(_listing_card(r, "under") for r in under_rows[:12])}</div>
</section>

<section>
  <h2>Top overpriced — asking too much</h2>
  <div class="cards">{"".join(_listing_card(r, "over") for r in over_rows[:12])}</div>
</section>

<section>
  <h2>Features the market systematically misprices</h2>
  <p>Among flagged listings, which features cluster with bargains vs rip-offs. A
  <b style="color:{C_UNDER}">blue</b> bar means the feature is over-represented
  among <i>underpriced</i> flats — the market underpays for it (hidden value). A
  <b style="color:{C_OVER}">red</b> bar means it clusters with <i>overpriced</i>
  flats — the market overpays for it (hype).</p>
  <div class="chart">{data["feature_chart"]}</div>
</section>

<section class="two-col">
  <div>
    <h2>What the market pays per feature</h2>
    <p>Hedonic regression: implicit €/month a feature adds to rent, holding size,
    location and condition constant (R² = {hed.attrs.get("r2", float("nan")):.2f}).</p>
    <table>
      <thead><tr><th>feature</th><th class="num">€/mo</th>
      <th class="num">effect</th><th class="num">t</th></tr></thead>
      <tbody>{hed_rows}</tbody>
    </table>
  </div>
  <div>
    <h2>Anomaly score distribution</h2>
    <p>Isolation-forest score for every listing. The right tail
    (<b style="color:{C_OVER}">red</b>, top decile) is where genuine anomalies
    live; most listings cluster as "normal".</p>
    <div class="chart">{data["hist"]}</div>
  </div>
</section>

<section class="method">
  <h2>Method &amp; honesty note</h2>
  <ul>
    <li><b>Data:</b> {len(df):,} live rental listings scraped from Fotocasa
      public search ({meta["scraped"]}), across {df["district"].nunique()} Madrid
      districts. Idealista was intended too but returns HTTP&nbsp;403 to
      non-browser clients (DataDome), so Fotocasa — which serves listings as an
      embedded JSON blob — was used.</li>
    <li><b>Location encoding:</b> haversine proximity to 243 real metro stations
      (Wikidata), 15 parks, 10 universities, plus a nightlife-noise kernel and
      centrality — not raw lat/lon alone.</li>
    <li><b>Isolation forest:</b> implemented from scratch (random split trees,
      path-length + c(n) normalisation, s = 2^(−E[h]/c(ψ))); no scikit-learn.</li>
    <li><b>Fair price:</b> median price/m² of each listing's 40 nearest
      comparables in a price-independent characteristic space; residual z-scored
      robustly (MAD).</li>
    <li><b>Caveats:</b> asking prices, not transacted rents; comparable sets are
      thin in low-inventory districts; Fotocasa's condition/antiquity fields are
      coded ordinals. Treat flags as leads to inspect, not verdicts.</li>
  </ul>
</section>
<footer>Generated by the Madrid rental anomaly pipeline · {meta["scraped"]}</footer>
</div>
"""
    return _STYLE, body


_STYLE = """
:root{
  --surface:#fcfcfb; --plane:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --baseline:#c3c2b7; --ring:rgba(11,11,11,.10);
  --under:#2a78d6; --over:#e34948; --struct:#eda100;
}
@media (prefers-color-scheme:dark){:root{
  --surface:#1a1a19; --plane:#0d0d0d; --ink:#fff; --ink2:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --baseline:#383835; --ring:rgba(255,255,255,.10);
  --under:#3987e5; --over:#e66767; --struct:#c98500;
}}
:root[data-theme=dark]{
  --surface:#1a1a19; --plane:#0d0d0d; --ink:#fff; --ink2:#c3c2b7;
  --grid:#2c2c2a; --baseline:#383835; --ring:rgba(255,255,255,.10);
  --under:#3987e5; --over:#e66767;}
:root[data-theme=light]{
  --surface:#fcfcfb; --plane:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e;
  --grid:#e1e0d9; --baseline:#c3c2b7; --under:#2a78d6; --over:#e34948;}
*{box-sizing:border-box}
body,.viz-root{background:var(--plane);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.5;margin:0}
.viz-root{max-width:880px;margin:0 auto;padding:40px 22px 80px}
.eyebrow{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);font-weight:600}
h1{font-size:31px;line-height:1.15;margin:.3em 0 .35em;letter-spacing:-.02em}
h2{font-size:20px;margin:0 0 .3em;letter-spacing:-.01em}
.lede{font-size:16px;color:var(--ink2);max-width:66ch}
p{color:var(--ink2);max-width:70ch}
section{margin-top:44px}
.stats{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-top:28px}
.stat{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:14px 14px 12px}
.stat-v{font-size:26px;font-weight:650;letter-spacing:-.02em}
.stat-l{font-size:12.5px;font-weight:600;margin-top:2px}
.stat-s{font-size:11.5px;color:var(--muted)}
.chart{background:var(--surface);border:1px solid var(--ring);border-radius:14px;
  padding:16px;margin-top:14px;overflow-x:auto}
.grid{stroke:var(--grid);stroke-width:1}
.baseline{stroke:var(--baseline);stroke-width:1.4}
.refline{stroke:var(--baseline);stroke-width:1.4;stroke-dasharray:5 4}
.tick{fill:var(--muted);font-size:11px;font-family:inherit}
.axis-title{fill:var(--ink2);font-size:12px;font-weight:600}
.barlab{fill:var(--ink);font-size:13px;font-weight:600}
.barsub{fill:var(--muted);font-size:11px}
.barval{fill:var(--ink2);font-size:12px;font-weight:600;font-variant-numeric:tabular-nums}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12.5px;color:var(--ink2);margin-top:10px}
.legend i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:14px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:12px;
  padding:13px 14px;border-left:3px solid var(--baseline)}
.card.under{border-left-color:var(--under)}
.card.over{border-left-color:var(--over)}
.card-h{display:flex;justify-content:space-between;gap:8px;align-items:baseline}
.card-t{font-weight:650;font-size:14px}
.card-p{font-size:12.5px;color:var(--muted);white-space:nowrap;font-variant-numeric:tabular-nums}
.card-b{font-size:12.8px;color:var(--ink2);margin:7px 0 9px}
.card-f{display:flex;gap:8px;align-items:center;font-size:12px;flex-wrap:wrap}
.pill{padding:2px 8px;border-radius:20px;font-weight:600;font-size:11.5px;color:#fff}
.pill.under{background:var(--under)} .pill.over{background:var(--over)}
.pill-2{padding:2px 8px;border-radius:20px;background:var(--grid);color:var(--ink2);font-size:11.5px}
.card-f a{color:var(--under);text-decoration:none;font-weight:600}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}
th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--grid)}
th{color:var(--muted);font-weight:600;font-size:11.5px;text-transform:uppercase;letter-spacing:.03em}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.muted{color:var(--muted)}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:30px}
.method ul{color:var(--ink2);font-size:13.5px;padding-left:18px}
.method li{margin:6px 0}
footer{margin-top:50px;color:var(--muted);font-size:12px;border-top:1px solid var(--grid);padding-top:16px}
@media(max-width:680px){.stats{grid-template-columns:repeat(2,1fr)}
  .cards,.two-col{grid-template-columns:1fr}h1{font-size:25px}}
"""


def load_data() -> dict:
    df = pd.read_csv(OUT / "all_scored.csv")
    under = pd.read_csv(OUT / "underpriced.csv")
    over = pd.read_csv(OUT / "overpriced.csv")
    fm = pd.read_csv(OUT / "feature_mispricing.csv")
    hed = pd.read_csv(OUT / "hedonic_coefficients.csv")
    # recover R² if present as attr is lost on CSV round-trip: recompute proxy
    hed.attrs["r2"] = _read_r2()
    scraped = _scrape_date()
    return {"all": df, "under": under, "over": over,
            "feature_mispricing": fm, "hedonic": hed,
            "meta": {"scraped": scraped}}


def _read_r2() -> float:
    p = OUT / "hedonic_r2.txt"
    try:
        return float(p.read_text().strip())
    except Exception:
        return float("nan")


def _scrape_date() -> str:
    log = ROOT / "data" / "scrape_log.txt"
    try:
        import os, datetime
        ts = os.path.getmtime(log)
        return datetime.datetime.fromtimestamp(ts).strftime("%d %b %Y")
    except Exception:
        return "2026"


def build() -> Path:
    data = load_data()
    data["scatter"] = scatter_expected_vs_actual(data["all"])
    data["feature_chart"] = feature_mispricing_chart(data["feature_mispricing"])
    data["hist"] = anomaly_histogram(data["all"])
    style, body = render_body(data)
    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>Madrid rental anomalies</title><style>{style}</style></head>'
           f'<body>{body}</body></html>')
    out = OUT / "report.html"
    out.write_text(doc, encoding="utf-8")

    # Body-only variant for publishing as a claude.ai Artifact (the host wraps
    # it in its own <head>/<body> skeleton — no doctype/html/head/body here).
    (OUT / "artifact_body.html").write_text(
        f"<style>{style}</style>{body}", encoding="utf-8")

    print(f"[report] wrote {out} and artifact_body.html")
    return out


if __name__ == "__main__":
    build()
