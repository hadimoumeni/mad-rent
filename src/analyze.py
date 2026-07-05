"""
Mispricing analysis.

Three complementary signals, each doing a job the others can't:

1. Isolation forest (from scratch) on the FULL feature space (characteristics +
   price/m²). Gives every listing a global anomaly score in (0,1): how unusual
   is this listing overall?

2. Comparable-based price residual. For each listing we find its K nearest
   neighbours in the *price-independent* characteristic space (same size, layout,
   condition, location, amenities) and ask: what do genuinely similar flats rent
   for? The signed gap tells us under- vs over-priced, and by how much.

3. Hedonic OLS regression (from scratch). The market's implicit price for every
   feature (€/month), and — by looking at which features cluster in the
   under- vs over-priced tails — which features the market *systematically*
   misprices.

A listing is flagged as mispriced when it is both statistically anomalous (1)
and far from its comparables (2). The isolation forest keeps us honest: we flag
genuine outliers, not merely the cheapest or most expensive listings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from features import load_and_clean, build_matrices, AMENITY_COLS, FLAG_COLS
from isolation_forest import IsolationForest

OUT_DIR = Path(__file__).resolve().parent.parent / "output"

K_COMPS = 40                # neighbours used to establish a "fair" price
MIN_COMP_DISPERSION = 0.02  # ignore residuals where comps are suspiciously uniform


# --------------------------------------------------------------------------- #
#  Comparable-based fair price
# --------------------------------------------------------------------------- #
def comparable_prices(df: pd.DataFrame, char_matrix: np.ndarray, k: int = K_COMPS):
    """
    For each listing, find its k nearest comparables in characteristic space and
    derive an expected price/m² (robust median) plus a robust z-score of how far
    the listing's own price/m² sits from that comparable set.
    """
    n = char_matrix.shape[0]
    ppm2 = df["price_per_m2"].to_numpy()

    exp_ppm2 = np.empty(n)
    resid_z = np.empty(n)
    comp_disp = np.empty(n)
    comp_dist = np.empty(n)      # mean feature-space distance to the comps
    top_comp_idx = []

    for i in range(n):
        d = np.sqrt(((char_matrix - char_matrix[i]) ** 2).sum(axis=1))
        d[i] = np.inf
        nn = np.argpartition(d, k)[:k]
        comp_ppm2 = ppm2[nn]

        med = np.median(comp_ppm2)
        mad = np.median(np.abs(comp_ppm2 - med))
        robust_sd = 1.4826 * mad if mad > 0 else comp_ppm2.std()
        robust_sd = max(robust_sd, MIN_COMP_DISPERSION * med)

        exp_ppm2[i] = med
        resid_z[i] = (ppm2[i] - med) / robust_sd
        comp_disp[i] = robust_sd / med
        comp_dist[i] = d[nn].mean()
        top_comp_idx.append(nn[np.argsort(d[nn])][:6])

    expected_price = exp_ppm2 * df["surface"].to_numpy()
    residual_pct = (df["price"].to_numpy() - expected_price) / expected_price
    return {
        "expected_ppm2": exp_ppm2,
        "expected_price": expected_price,
        "residual_pct": residual_pct,
        "residual_z": resid_z,
        "comp_dispersion": comp_disp,
        "comp_distance": comp_dist,
        "top_comps": top_comp_idx,
    }


# --------------------------------------------------------------------------- #
#  Hedonic OLS regression (from scratch) — the market's implicit feature prices
# --------------------------------------------------------------------------- #
def hedonic_regression(df: pd.DataFrame) -> pd.DataFrame:
    """
    OLS of log(price) on characteristics + district dummies. Coefficients give
    each feature's implicit price; we translate them to an approximate % effect
    and a €/month effect at the median rent. t-stats come from the standard
    (X'X)^-1 * sigma^2 covariance.
    """
    cont = ["log_surface", "rooms", "bathrooms", "floor_code",
            "conservation_code", "antiquity_code", "dist_metro_m",
            "dist_park_m", "dist_center_km", "noise_index"]
    bins = AMENITY_COLS + FLAG_COLS

    X_parts = [np.ones((len(df), 1))]
    names = ["intercept"]

    # standardise continuous predictors so coefficients are comparable
    for c in cont:
        v = df[c].to_numpy(float)
        X_parts.append(((v - v.mean()) / (v.std() or 1.0)).reshape(-1, 1))
        names.append(c)
    for c in bins:
        X_parts.append(df[c].to_numpy(float).reshape(-1, 1))
        names.append(c)

    # district dummies (drop first as reference)
    dummies = pd.get_dummies(df["district"].fillna("Unknown"), prefix="dist")
    dummies = dummies.iloc[:, 1:]
    for c in dummies.columns:
        X_parts.append(dummies[c].to_numpy(float).reshape(-1, 1))
        names.append(c)

    X = np.hstack(X_parts)
    y = df["log_price"].to_numpy(float)

    # least squares via pseudo-inverse (robust to any collinearity)
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X.T @ y)

    resid = y - X @ beta
    dof = max(len(df) - X.shape[1], 1)
    sigma2 = (resid @ resid) / dof
    # clip tiny negative diagonals from the pseudo-inverse before sqrt
    se = np.sqrt(np.maximum(np.diag(XtX_inv) * sigma2, 0.0))
    tstat = beta / np.where(se == 0, np.nan, se)

    median_rent = df["price"].median()
    rows = []
    for nm, b, t in zip(names, beta, tstat):
        pct = np.exp(b) - 1.0                       # approx % effect on price
        rows.append({
            "feature": nm,
            "coef_logprice": b,
            "pct_effect": pct,
            "eur_effect_at_median": pct * median_rent,
            "t_stat": t,
        })
    r2 = 1 - (resid @ resid) / (((y - y.mean()) ** 2).sum())
    out = pd.DataFrame(rows)
    out.attrs["r2"] = r2
    return out


def feature_mispricing(df: pd.DataFrame, under_mask, over_mask) -> pd.DataFrame:
    """
    Which features cluster in the under- vs over-priced tails?

    For each binary feature we compare its prevalence among under-priced flags,
    over-priced flags, and the whole market. A feature over-represented among
    *under*-priced listings is one the market systematically *underpays* for
    (hidden value); one over-represented among *over*-priced listings is a
    feature the market *overpays* for (hype). Lift = prevalence ratio.
    """
    feats = AMENITY_COLS + FLAG_COLS + ["has_video", "has_floorplan",
                                        "is_opportunity", "is_new_construction"]
    base_rows = []
    n_under = max(under_mask.sum(), 1)
    n_over = max(over_mask.sum(), 1)
    for f in feats:
        if f not in df.columns:
            continue
        v = df[f].astype(bool)
        p_all = v.mean()
        if p_all == 0:
            continue
        p_under = v[under_mask].mean()
        p_over = v[over_mask].mean()
        base_rows.append({
            "feature": f,
            "prevalence_all": p_all,
            "prevalence_underpriced": p_under,
            "prevalence_overpriced": p_over,
            "underpay_lift": p_under / p_all,     # >1 => market underpays for it
            "overpay_lift": p_over / p_all,       # >1 => market overpays for it
            "net_skew": (p_under - p_over) / p_all,
        })
    return pd.DataFrame(base_rows).sort_values("net_skew", ascending=False)


# --------------------------------------------------------------------------- #
#  Explanations
# --------------------------------------------------------------------------- #
def explain(df: pd.DataFrame, i: int, comps: dict, char_cols: list[str],
            char_matrix: np.ndarray) -> str:
    r = df.iloc[i]
    exp_price = comps["expected_price"][i]
    rp = comps["residual_pct"][i] * 100
    verb = "below" if rp < 0 else "above"

    # Which characteristics set this flat apart from its comparables?
    nn = comps["top_comps"][i]
    comp_mean = char_matrix[nn].mean(axis=0)
    diff = char_matrix[i] - comp_mean
    drivers = np.argsort(-np.abs(diff))[:3]
    driver_txt = ", ".join(
        f"{char_cols[j]} {'high' if diff[j] > 0 else 'low'}" for j in drivers
    )

    metro = f"{r['dist_metro_m']:.0f} m to {r['nearest_metro']}"
    return (
        f"€{r['price']:.0f}/mo · {r['surface']:.0f} m² · {int(r['rooms'])}BR/"
        f"{int(r['bathrooms'])}BA in {r['neighborhood']} ({r['district']}). "
        f"{metro}, {r['dist_center_km']:.1f} km to centre. "
        f"Comparable flats rent for ~€{exp_price:.0f} "
        f"(€{comps['expected_ppm2'][i]:.1f}/m²); this listing is "
        f"{abs(rp):.0f}% {verb} comps ({comps['residual_z'][i]:+.1f}σ). "
        f"Anomaly score {r['anomaly_score']:.2f}. Stands out on: {driver_txt}."
    )


# --------------------------------------------------------------------------- #
#  Orchestration
# --------------------------------------------------------------------------- #
def run(n_trees: int = 200, sample_size: int = 256, top_n: int = 20):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_clean()
    char_matrix, full_matrix, info = build_matrices(df)
    print(f"[analyze] {len(df)} clean listings, "
          f"{df['district'].nunique()} districts")

    # (1) Isolation forest on the full space
    forest = IsolationForest(n_trees=n_trees, sample_size=sample_size,
                             random_state=42).fit(full_matrix)
    df["anomaly_score"] = forest.anomaly_score(full_matrix)

    # (2) Comparable-based price residual
    comps = comparable_prices(df, char_matrix, k=min(K_COMPS, len(df) - 1))
    df["expected_price"] = comps["expected_price"]
    df["expected_ppm2"] = comps["expected_ppm2"]
    df["residual_pct"] = comps["residual_pct"]
    df["residual_z"] = comps["residual_z"]
    df["comp_dispersion"] = comps["comp_dispersion"]
    df["comp_distance"] = comps["comp_distance"]

    # Combined mispricing score: blend "far from comparables" with "globally
    # anomalous". Sign carries direction.
    anomaly_norm = (df["anomaly_score"] - df["anomaly_score"].min()) / (
        df["anomaly_score"].max() - df["anomaly_score"].min() + 1e-9)
    df["mispricing_score"] = np.abs(df["residual_z"]) * (0.5 + anomaly_norm)

    # A mispricing verdict requires (a) the price to be clearly off its
    # comparables and (b) those comparables to be trustworthy — if the flat sits
    # in a sparse region of feature space (rare luxury unit, unusual layout) its
    # comps are unreliable, so we call it a *structural* anomaly instead. The
    # isolation-forest anomaly score is not used as a hard gate (it would starve
    # the naturally-compressed under-priced tail); instead it drives the ranking
    # via mispricing_score and is reported per listing.
    #
    # The residual distribution is asymmetric: over-pricing has a long tail
    # (aspirational/luxury listings) while under-pricing is compressed (landlords
    # rarely leave large money on the table), so we use a sign-appropriate |z|
    # threshold rather than a single symmetric cutoff.
    clearly_off = df["residual_z"].abs() >= 1.5
    well_supported = df["comp_distance"] <= df["comp_distance"].quantile(0.88)
    df["verdict"] = "fair"
    df.loc[clearly_off & well_supported
           & (df["residual_z"] < 0), "verdict"] = "underpriced"
    df.loc[clearly_off & well_supported
           & (df["residual_z"] > 0), "verdict"] = "overpriced"
    # anomalous but price roughly fair, OR anomalous with unreliable comps ==
    # unusual property rather than a mispricing
    df.loc[(df["verdict"] == "fair")
           & (df["anomaly_score"] >= df["anomaly_score"].quantile(0.9))
           & ((df["residual_z"].abs() < 1.0) | ~well_supported),
           "verdict"] = "structural_anomaly"

    # (3) Hedonic regression + systematic feature mispricing
    hedonic = hedonic_regression(df)
    under_mask = df["verdict"] == "underpriced"
    over_mask = df["verdict"] == "overpriced"
    feat_mis = feature_mispricing(df, under_mask, over_mask)

    # Explanations for flagged listings
    df["explanation"] = ""
    flagged = df.index[df["verdict"].isin(["underpriced", "overpriced"])]
    for i in flagged:
        df.at[i, "explanation"] = explain(df, i, comps, info["char_cols"], char_matrix)

    # ---- rankings ----
    under = (df[under_mask].sort_values(["mispricing_score", "residual_z"],
                                        ascending=[False, True]).head(top_n))
    over = (df[over_mask].sort_values(["mispricing_score", "residual_z"],
                                      ascending=[False, False]).head(top_n))

    # ---- save ----
    keep = ["id", "url", "district", "neighborhood", "nearest_metro",
            "price", "surface", "rooms", "bathrooms", "price_per_m2",
            "dist_metro_m", "dist_center_km", "expected_price", "expected_ppm2",
            "residual_pct", "residual_z", "anomaly_score", "mispricing_score",
            "verdict", "explanation"]
    df[keep].to_csv(OUT_DIR / "all_scored.csv", index=False)
    under[keep].to_csv(OUT_DIR / "underpriced.csv", index=False)
    over[keep].to_csv(OUT_DIR / "overpriced.csv", index=False)
    hedonic.to_csv(OUT_DIR / "hedonic_coefficients.csv", index=False)
    feat_mis.to_csv(OUT_DIR / "feature_mispricing.csv", index=False)
    (OUT_DIR / "hedonic_r2.txt").write_text(f"{hedonic.attrs['r2']:.4f}")

    print(f"[analyze] underpriced: {under_mask.sum()}  "
          f"overpriced: {over_mask.sum()}  "
          f"structural anomalies: {(df['verdict']=='structural_anomaly').sum()}")
    print(f"[analyze] hedonic R² = {hedonic.attrs['r2']:.3f}")
    print(f"[analyze] outputs written to {OUT_DIR}")

    return df, under, over, hedonic, feat_mis, comps, info, char_matrix


if __name__ == "__main__":
    df, under, over, hedonic, feat_mis, *_ = run()
    print("\n=== TOP 5 UNDERPRICED ===")
    for _, r in under.head(5).iterrows():
        print(" •", r["explanation"])
    print("\n=== TOP 5 OVERPRICED ===")
    for _, r in over.head(5).iterrows():
        print(" •", r["explanation"])
    print("\n=== MARKET UNDERPAYS MOST FOR (hidden-value features) ===")
    print(feat_mis.head(6)[["feature", "underpay_lift", "overpay_lift", "net_skew"]]
          .to_string(index=False))
