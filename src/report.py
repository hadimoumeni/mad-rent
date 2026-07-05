"""
Static HTML report builder for the Madrid rental anomaly detector.

Dependency-free: every chart is inline SVG generated here (no JS libraries, no
matplotlib), so output/report.html is fully self-contained and opens anywhere.

Frontend follows the hadimoumeni.com design system: dark (#0a0a0a), JetBrains
Mono (+ Sora for the wordmark), an opacity-only text hierarchy, a "/" slash
motif, and a restrained two-accent scheme — teal for under-priced (a good deal),
red for over-priced — which is the report's whole story.
"""

from __future__ import annotations

import html
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"

# Accents reference CSS custom properties defined once in :root, so the palette
# lives in a single place and inline SVG inherits it.
C_UNDER = "var(--under)"    # teal  — underpriced (good deal)
C_OVER = "var(--over)"      # red   — overpriced
C_FAIR = "var(--fair)"      # faint — fairly priced
C_STRUCT = "var(--struct)"  # amber — structural anomaly (unusual, not mispriced)


def _esc(x) -> str:
    return html.escape(str(x))


# --------------------------------------------------------------------------- #
#  SVG primitives
# --------------------------------------------------------------------------- #
def scatter_expected_vs_actual(df: pd.DataFrame) -> str:
    """Log-log scatter: expected (comparable) price vs actual asking price."""
    W, H = 760, 470
    pad_l, pad_b, pad_t, pad_r = 62, 52, 22, 18
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

    for tick in [500, 1000, 2000, 5000, 10000]:
        if lo <= math.log10(tick) <= hi:
            gx, gy = sx(tick), sy(tick)
            parts.append(f'<line x1="{gx:.1f}" y1="{pad_t}" x2="{gx:.1f}" '
                         f'y2="{H-pad_b}" class="grid"/>')
            parts.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{W-pad_r}" '
                         f'y2="{gy:.1f}" class="grid"/>')
            parts.append(f'<text x="{gx:.1f}" y="{H-pad_b+15}" class="tick" '
                         f'text-anchor="middle">{tick//1000}k</text>'
                         if tick >= 1000 else
                         f'<text x="{gx:.1f}" y="{H-pad_b+15}" class="tick" '
                         f'text-anchor="middle">{tick}</text>')
            parts.append(f'<text x="{pad_l-8}" y="{gy+3.5:.1f}" class="tick" '
                         f'text-anchor="end">€{tick//1000 if tick>=1000 else tick}'
                         f'{"k" if tick>=1000 else ""}</text>')

    p1x, p1y = sx(10 ** lo), sy(10 ** lo)
    p2x, p2y = sx(10 ** hi), sy(10 ** hi)
    parts.append(f'<line x1="{p1x:.1f}" y1="{p1y:.1f}" x2="{p2x:.1f}" '
                 f'y2="{p2y:.1f}" class="refline"/>')
    parts.append(f'<text x="{sx(10**hi)-6:.1f}" y="{sy(10**hi)-6:.1f}" '
                 f'class="tick" text-anchor="end">fair · actual = comparable</text>')

    color = {"underpriced": C_UNDER, "overpriced": C_OVER,
             "structural_anomaly": C_STRUCT, "fair": C_FAIR}
    order = df.sort_values("verdict", key=lambda s: s.map(
        {"fair": 0, "structural_anomaly": 1, "overpriced": 2, "underpriced": 3}))
    for _, r in order.iterrows():
        c = color.get(r["verdict"], C_FAIR)
        rad = 2.8 if r["verdict"] in ("fair", "structural_anomaly") else 3.8
        op = 0.5 if r["verdict"] == "fair" else 0.92
        parts.append(f'<circle cx="{sx(r["expected_price"]):.1f}" '
                     f'cy="{sy(r["price"]):.1f}" r="{rad}" fill="{c}" '
                     f'fill-opacity="{op}" stroke="var(--bg)" '
                     f'stroke-width="0.7"><title>{_esc(r["neighborhood"])}: '
                     f'€{r["price"]:.0f} vs €{r["expected_price"]:.0f} '
                     f'expected ({r["residual_pct"]*100:+.0f}%)</title></circle>')

    parts.append(f'<text x="{(pad_l+W-pad_r)/2:.0f}" y="{H-6}" class="axis-title" '
                 f'text-anchor="middle">comparable (expected) rent →</text>')
    parts.append(f'<text transform="translate(15,{(pad_t+H-pad_b)/2:.0f}) '
                 f'rotate(-90)" class="axis-title" text-anchor="middle">'
                 f'actual asking rent →</text>')
    parts.append("</svg>")
    return "".join(parts)


def feature_mispricing_chart(fm: pd.DataFrame) -> str:
    """Diverging bars: features the market underpays (teal) vs overpays (red) for."""
    fm = fm.copy()
    fm = fm[fm["prevalence_all"] >= 0.03]
    fm = fm.reindex(fm["net_skew"].abs().sort_values(ascending=False).index).head(11)
    if fm.empty:
        return "<p class='muted'>Not enough flagged listings to profile features.</p>"
    rowh, W, mid = 30, 760, 380
    H = len(fm) * rowh + 26
    vmax = fm["net_skew"].abs().max() or 1
    span = W - mid - 120
    parts = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto" role="img">']
    parts.append(f'<line x1="{mid}" y1="18" x2="{mid}" y2="{H-6}" class="baseline"/>')
    parts.append(f'<text x="{mid-6}" y="13" class="tick" text-anchor="end">'
                 f'← underpays for</text>')
    parts.append(f'<text x="{mid+6}" y="13" class="tick">overpays for →</text>')
    for i, (_, r) in enumerate(fm.iterrows()):
        y = i * rowh + 26
        v = r["net_skew"]
        w = abs(v) / vmax * span
        parts.append(f'<text x="0" y="{y+13}" class="barlab">'
                     f'{_esc(_pretty(r["feature"]))}</text>')
        if v >= 0:   # over-represented among underpriced -> market underpays
            parts.append(f'<rect x="{mid-w:.1f}" y="{y+2}" width="{w:.1f}" '
                         f'height="14" rx="3" fill="{C_UNDER}" fill-opacity="0.9"/>')
        else:
            parts.append(f'<rect x="{mid}" y="{y+2}" width="{w:.1f}" '
                         f'height="14" rx="3" fill="{C_OVER}" fill-opacity="0.9"/>')
    parts.append("</svg>")
    return "".join(parts)


def anomaly_histogram(df: pd.DataFrame) -> str:
    W, H = 760, 200
    pad_l, pad_b, pad_t, pad_r = 30, 34, 16, 10
    scores = df["anomaly_score"]
    bins = 28
    lo, hi = scores.min(), scores.max()
    counts = [0] * bins
    for s in scores:
        b = min(int((s - lo) / (hi - lo + 1e-9) * bins), bins - 1)
        counts[b] += 1
    cmax = max(counts) or 1
    bw = (W - pad_l - pad_r) / bins
    q90 = df["anomaly_score"].quantile(0.9)
    parts = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto" role="img">']
    for i, c in enumerate(counts):
        bh = c / cmax * (H - pad_b - pad_t)
        x = pad_l + i * bw
        centre = lo + (i + 0.5) / bins * (hi - lo)
        col = C_OVER if centre >= q90 else "rgba(255,255,255,0.30)"
        parts.append(f'<rect x="{x+1:.1f}" y="{H-pad_b-bh:.1f}" '
                     f'width="{bw-2:.1f}" height="{bh:.1f}" rx="1.5" fill="{col}"/>')
    parts.append(f'<line x1="{pad_l}" y1="{H-pad_b}" x2="{W-pad_r}" '
                 f'y2="{H-pad_b}" class="baseline"/>')
    for t in [lo, (lo + hi) / 2, hi]:
        tx = pad_l + (t - lo) / (hi - lo + 1e-9) * (W - pad_l - pad_r)
        parts.append(f'<text x="{tx:.1f}" y="{H-pad_b+15}" class="tick" '
                     f'text-anchor="middle">{t:.2f}</text>')
    parts.append(f'<text x="{pad_l}" y="12" class="tick">← more normal        '
                 f'more anomalous →</text>')
    parts.append("</svg>")
    return "".join(parts)


def _pretty(f: str) -> str:
    return (f.replace("has_", "").replace("is_", "").replace("_", " ")
            .replace("air conditioning", "a/c").strip())


# --------------------------------------------------------------------------- #
#  Page assembly
# --------------------------------------------------------------------------- #
def _stat(label, value, sub=""):
    return (f'<div class="stat"><div class="stat-v">{value}</div>'
            f'<div class="stat-l">{label}</div>'
            f'<div class="stat-s">{sub}</div></div>')


def _listing_card(r, klass):
    url = r.get("url")
    title = f'{r["neighborhood"]} · {r["district"]}'
    link = (f'<a href="{_esc(url)}" target="_blank" rel="noopener">view ↗</a>'
            if isinstance(url, str) and url.startswith("http") else "")
    return (
        f'<div class="card {klass}">'
        f'<div class="card-h"><span class="card-t">{_esc(title)}</span>'
        f'<span class="card-p">€{r["price"]:.0f}/mo · {r["surface"]:.0f} m²</span></div>'
        f'<div class="card-b">{_esc(r["explanation"])}</div>'
        f'<div class="card-f"><span class="pill {klass}">'
        f'{r["residual_pct"]*100:+.0f}% vs comps</span> '
        f'<span class="pill-2">anomaly {r["anomaly_score"]:.2f}</span> {link}</div>'
        f'</div>')


def render_body(data: dict) -> tuple[str, str]:
    df = data["all"]
    under = data["under"]
    over = data["over"]
    hed = data["hedonic"]
    meta = data["meta"]

    n_under = (df["verdict"] == "underpriced").sum()
    n_over = (df["verdict"] == "overpriced").sum()
    n_struct = (df["verdict"] == "structural_anomaly").sum()

    stats = "".join([
        _stat("listings", f'{len(df):,}', f'{df["district"].nunique()} districts'),
        _stat("median rent", f'€{df["price"].median():.0f}',
              f'€{df["price_per_m2"].median():.1f}/m²'),
        _stat("underpriced", f'{n_under}', "bargains"),
        _stat("overpriced", f'{n_over}', "asking too much"),
        _stat("structural", f'{n_struct}', "rare, price fair"),
    ])

    under_rows = under.to_dict("records")
    over_rows = over.to_dict("records")

    hed_bin = hed[hed["feature"].str.startswith(("has_", "is_"))].copy()
    hed_bin = hed_bin.reindex(hed_bin["eur_effect_at_median"].abs()
                              .sort_values(ascending=False).index).head(9)
    hed_rows = "".join(
        f'<tr><td>{_esc(_pretty(r["feature"]))}</td>'
        f'<td class="num">{r["eur_effect_at_median"]:+.0f}</td>'
        f'<td class="num">{r["pct_effect"]*100:+.0f}%</td>'
        f'<td class="num muted">{r["t_stat"]:+.1f}</td></tr>'
        for _, r in hed_bin.iterrows())

    body = f"""
<div class="wrap">
<header class="masthead animate-in">
  <a class="logo" href="https://hadimoumeni.com">mad<span class="sl">/</span>rent</a>
  <span class="sub">madrid rental anomaly detector</span>
  <a class="home" href="https://hadimoumeni.com">hadimoumeni.com ↗</a>
</header>

<section class="hero animate-in d1" style="margin-top:0">
  <div class="eyebrow"><span class="sl">/</span> rental price anomaly detection · madrid</div>
  <h1>Which Madrid rentals are genuinely mispriced?</h1>
  <p class="lede">An isolation forest, built from scratch, over {len(df):,} live
  Fotocasa listings, combined with a comparable-based fair-price model and a
  hedonic regression. It flags listings whose price doesn't fit their size,
  location, condition and features. Not the cheapest or the dearest, the
  <em>anomalous</em>.</p>
</section>

<section class="stats animate-in d2">{stats}</section>

<section class="animate-in">
  <h2><span class="sl">/</span> the map of mispricing</h2>
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

<section class="animate-in">
  <h2><span class="sl">/</span> top underpriced - the bargains</h2>
  <p>Ranked by combined mispricing score: distance from comparables × isolation-forest anomaly.</p>
  <div class="cards">{"".join(_listing_card(r, "under") for r in under_rows[:12])}</div>
</section>

<section class="animate-in">
  <h2><span class="sl">/</span> top overpriced - asking too much</h2>
  <div class="cards">{"".join(_listing_card(r, "over") for r in over_rows[:12])}</div>
</section>

<section class="animate-in">
  <h2><span class="sl">/</span> features the market systematically misprices</h2>
  <p>Among flagged listings, which features cluster with bargains vs rip-offs. A
  <b style="color:{C_UNDER}">teal</b> bar means the feature is over-represented
  among <em>underpriced</em> flats, so the market underpays for it (hidden value).
  A <b style="color:{C_OVER}">red</b> bar means it clusters with
  <em>overpriced</em> flats, so the market overpays for it (hype).</p>
  <div class="chart">{data["feature_chart"]}</div>
</section>

<section class="two-col animate-in">
  <div>
    <h2><span class="sl">/</span> what the market pays per feature</h2>
    <p>Hedonic regression: implicit €/month a feature adds to rent, holding size,
    location and condition constant (R² = {hed.attrs.get("r2", float("nan")):.2f}).</p>
    <table>
      <thead><tr><th>feature</th><th class="num">€/mo</th>
      <th class="num">effect</th><th class="num">t</th></tr></thead>
      <tbody>{hed_rows}</tbody>
    </table>
  </div>
  <div>
    <h2><span class="sl">/</span> anomaly score distribution</h2>
    <p>Isolation-forest score for every listing. The right tail
    (<b style="color:{C_OVER}">red</b>, top decile) is where genuine anomalies
    live; most listings cluster as "normal".</p>
    <div class="chart">{data["hist"]}</div>
  </div>
</section>

<section class="method animate-in">
  <h2><span class="sl">/</span> method &amp; honesty note</h2>
  <ul>
    <li><b>Data:</b> {len(df):,} live rental listings scraped from Fotocasa
      ({meta["scraped"]}), across all {df["district"].nunique()} Madrid districts.
      Idealista was intended too but returns HTTP&nbsp;403 to non-browser clients
      (DataDome), so Fotocasa, which embeds listings as JSON, was used.</li>
    <li><b>Location:</b> haversine proximity to 243 real metro stations
      (Wikidata), 15 parks, 10 universities, a nightlife-noise kernel and
      centrality. Not raw lat/lon alone.</li>
    <li><b>Isolation forest:</b> from scratch. Random split trees, path-length +
      c(n) normalisation, s = 2^(-E[h]/c(ψ)), no scikit-learn.</li>
    <li><b>Fair price:</b> median price/m² of each listing's 40 nearest
      comparables in a price-independent characteristic space; residual
      z-scored robustly (MAD), gated on comparable reliability.</li>
    <li><b>Caveats:</b> asking prices, not transacted rents; comparable sets are
      thin in low-inventory districts; condition/antiquity are coded ordinals.
      Flags are leads to inspect, not verdicts.</li>
  </ul>
</section>

<footer>
  built by <a href="https://hadimoumeni.com">hadi moumeni</a> ·
  <a href="https://github.com/hadimoumeni/mad-rent">source on github</a> ·
  data scraped {meta["scraped"]}
</footer>
</div>
"""
    return _STYLE, body


_STYLE = """
:root{
  --bg:#0a0a0a;
  --font-mono:'JetBrains Mono',ui-monospace,monospace;
  --font-display:'Sora',var(--font-mono);
  --t1:rgba(255,255,255,.92); --t2:rgba(255,255,255,.70); --t3:rgba(255,255,255,.50);
  --t4:rgba(255,255,255,.35); --t5:rgba(255,255,255,.22); --t6:rgba(255,255,255,.14);
  --t7:rgba(255,255,255,.08);
  --divider:rgba(255,255,255,.07); --border-subtle:rgba(255,255,255,.05);
  --surface:rgba(255,255,255,.025); --surface-hover:rgba(255,255,255,.05);
  --under:#37b7a4; --over:#e0533f; --struct:#c9a24a; --fair:rgba(255,255,255,.18);
  --max:940px;
}
*{box-sizing:border-box;margin:0;padding:0}
html{-webkit-text-size-adjust:100%}
body{font-family:var(--font-mono);background:var(--bg);color:var(--t3);
  line-height:1.6;-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale}
em{font-style:normal;color:var(--t1)}
a{color:var(--t4);text-decoration:none;border-bottom:1px solid var(--t7);
  transition:color .25s,border-color .25s}
a:hover{color:var(--t2);border-color:var(--t5)}
.wrap{max-width:var(--max);margin:0 auto;padding:2.4rem 2rem 6rem}
.masthead{display:flex;align-items:baseline;gap:.9rem;padding-bottom:2rem;
  border-bottom:1px solid var(--divider)}
.logo{font-family:var(--font-display);font-weight:600;font-size:1.2rem;color:var(--t1);
  letter-spacing:.02em;border:none}
.logo .sl{color:var(--t6)}
.masthead .sub{font-size:.66rem;color:var(--t4);letter-spacing:.03em}
.masthead .home{margin-left:auto;font-size:.64rem;color:var(--t4);border:none;letter-spacing:.02em}
.eyebrow{font-size:.64rem;letter-spacing:.14em;text-transform:uppercase;color:var(--t4)}
.eyebrow .sl{color:var(--t6)}
h1{font-family:var(--font-display);font-weight:600;font-size:clamp(1.7rem,4.2vw,2.5rem);
  color:var(--t1);line-height:1.14;letter-spacing:-.015em;margin:.7rem 0 .8rem;text-wrap:balance}
.lede{font-size:.82rem;color:var(--t3);max-width:66ch;font-weight:300;line-height:1.75}
section{margin-top:3.4rem}
h2{font-family:var(--font-mono);font-size:.78rem;font-weight:600;color:var(--t1);
  text-transform:uppercase;letter-spacing:.07em;margin-bottom:.5rem}
h2 .sl{color:var(--t6);margin-right:.45rem}
section>p{font-size:.76rem;color:var(--t3);font-weight:300;line-height:1.7;
  max-width:74ch;margin-bottom:.5rem}
b{color:var(--t2);font-weight:500}
.stats{display:grid;grid-template-columns:repeat(5,1fr);gap:.55rem;margin-top:2rem}
.stat{border:1px solid var(--border-subtle);background:var(--surface);border-radius:4px;
  padding:.85rem .8rem}
.stat-v{font-family:var(--font-display);font-size:1.45rem;font-weight:600;color:var(--t1);
  letter-spacing:-.01em}
.stat-l{font-size:.63rem;color:var(--t3);letter-spacing:.02em;margin-top:.35rem}
.stat-s{font-size:.58rem;color:var(--t5)}
.chart{border:1px solid var(--border-subtle);background:var(--surface);border-radius:4px;
  padding:1rem;margin-top:.8rem;overflow-x:auto}
.grid{stroke:var(--divider);stroke-width:1}
.baseline{stroke:var(--t6);stroke-width:1.2}
.refline{stroke:var(--t5);stroke-width:1.2;stroke-dasharray:4 4}
.tick{fill:var(--t4);font-size:10px;font-family:var(--font-mono)}
.axis-title{fill:var(--t3);font-size:10.5px;font-family:var(--font-mono);letter-spacing:.03em}
.barlab{fill:var(--t1);font-size:12px;font-family:var(--font-mono)}
.barval{fill:var(--t2);font-size:11px;font-family:var(--font-mono)}
.legend{display:flex;gap:1.1rem;flex-wrap:wrap;font-size:.64rem;color:var(--t3);margin-top:.7rem}
.legend i{width:9px;height:9px;border-radius:2px;display:inline-block;margin-right:.4rem;vertical-align:0}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:.55rem;margin-top:.9rem}
.card{border:1px solid var(--border-subtle);background:var(--surface);border-left:2px solid var(--fair);
  border-radius:4px;padding:.8rem .9rem;transition:background .2s,border-color .2s}
.card:hover{background:var(--surface-hover)}
.card.under{border-left-color:var(--under)}
.card.over{border-left-color:var(--over)}
.card-h{display:flex;justify-content:space-between;gap:.6rem;align-items:baseline}
.card-t{font-size:.72rem;font-weight:600;color:var(--t1)}
.card-p{font-size:.64rem;color:var(--t4);white-space:nowrap;font-variant-numeric:tabular-nums}
.card-b{font-size:.68rem;color:var(--t3);font-weight:300;line-height:1.6;margin:.5rem 0 .6rem}
.card-f{display:flex;gap:.45rem;align-items:center;flex-wrap:wrap;font-size:.62rem}
.pill{padding:.14rem .5rem;border-radius:3px;font-weight:600;font-size:.6rem;color:#0a0a0a}
.pill.under{background:var(--under)} .pill.over{background:var(--over)}
.pill-2{padding:.14rem .5rem;border-radius:3px;background:var(--surface-hover);color:var(--t3)}
.card-f a{color:var(--t4);border:none}
.card-f a:hover{color:var(--under)}
table{width:100%;border-collapse:collapse;font-size:.7rem;margin-top:.6rem}
th,td{text-align:left;padding:.5rem .5rem;border-bottom:1px solid var(--divider)}
th{color:var(--t4);font-weight:500;font-size:.58rem;text-transform:uppercase;letter-spacing:.05em}
td{color:var(--t2);font-weight:300}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.muted{color:var(--t4)}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:2rem}
.method ul{list-style:none}
.method li{font-size:.7rem;color:var(--t3);font-weight:300;line-height:1.65;margin:.55rem 0;
  padding-left:1rem;position:relative}
.method li::before{content:"/";position:absolute;left:0;color:var(--t6)}
footer{margin-top:4rem;padding-top:1.4rem;border-top:1px solid var(--divider);
  font-size:.62rem;color:var(--t5);letter-spacing:.02em;line-height:1.9}
footer a{color:var(--t4);border:none}
footer a:hover{color:var(--t2)}
@keyframes fadeInUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
.animate-in{animation:fadeInUp .6s cubic-bezier(.16,1,.3,1) forwards;opacity:0}
.d1{animation-delay:.06s}.d2{animation-delay:.13s}
@media(prefers-reduced-motion:reduce){.animate-in{animation:none;opacity:1}}
@media(max-width:720px){
  .stats{grid-template-columns:repeat(2,1fr)}
  .cards,.two-col{grid-template-columns:1fr}
  .wrap{padding:1.8rem 1.3rem 4rem}
  .masthead{flex-wrap:wrap;gap:.5rem}
  .masthead .home{margin-left:0}
}
"""


def load_data() -> dict:
    df = pd.read_csv(OUT / "all_scored.csv")
    under = pd.read_csv(OUT / "underpriced.csv")
    over = pd.read_csv(OUT / "overpriced.csv")
    fm = pd.read_csv(OUT / "feature_mispricing.csv")
    hed = pd.read_csv(OUT / "hedonic_coefficients.csv")
    hed.attrs["r2"] = _read_r2()
    return {"all": df, "under": under, "over": over,
            "feature_mispricing": fm, "hedonic": hed,
            "meta": {"scraped": _scrape_date()}}


def _read_r2() -> float:
    try:
        return float((OUT / "hedonic_r2.txt").read_text().strip())
    except Exception:
        return float("nan")


def _scrape_date() -> str:
    try:
        import os, datetime
        ts = os.path.getmtime(ROOT / "data" / "scrape_log.txt")
        return datetime.datetime.fromtimestamp(ts).strftime("%d %b %Y")
    except Exception:
        return "2026"


_FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
          '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
          '<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:'
          'wght@300;400;500;600;700&family=Sora:wght@500;600&display=swap" '
          'rel="stylesheet">')


def build() -> Path:
    data = load_data()
    data["scatter"] = scatter_expected_vs_actual(data["all"])
    data["feature_chart"] = feature_mispricing_chart(data["feature_mispricing"])
    data["hist"] = anomaly_histogram(data["all"])
    style, body = render_body(data)
    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>mad/rent · Madrid rental anomalies</title>'
           f'<meta name="description" content="Isolation-forest anomaly detection '
           f'over live Madrid rental listings: ranked under/overpriced flats and '
           f'the features the market misprices.">{_FONTS}'
           f'<style>{style}</style></head><body>{body}</body></html>')
    out = OUT / "report.html"
    out.write_text(doc, encoding="utf-8")
    (OUT / "artifact_body.html").write_text(
        f"<style>{style}</style>{body}", encoding="utf-8")

    # Also emit the static deploy artifact (Vercel serves public/ as the site).
    pub = ROOT / "public"
    pub.mkdir(exist_ok=True)
    (pub / "index.html").write_text(doc, encoding="utf-8")

    print(f"[report] wrote {out}, public/index.html, artifact_body.html")
    return out


if __name__ == "__main__":
    build()
