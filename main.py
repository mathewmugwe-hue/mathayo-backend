"""
Soccer Match Prediction Engine — v3.0
Fixes applied to the original "Mathayo 154 Genius Engine":
  1. Shin's method for vig removal instead of additive normalization
     (corrects for favorite-longshot bias — Shin 1993)
  2. Dixon-Coles low-score correlation adjustment (Dixon & Coles 1997)
  3. Real edge/value detection: compares model probability against
     market-implied probability instead of just picking the highest
     raw probability
  4. Kelly-criterion stake sizing on genuine positive-EV picks only
  5. Optional independent signal inputs (form xG, injuries) so the
     model isn't structurally just a smoothed copy of the market

HONESTY NOTE (read this before trusting output):
  If you call /predict with ONLY odds (no form_xg / injury inputs),
  this model has NO independent information beyond the market itself.
  It will de-vig and shape the market's own numbers more accurately
  than the original script did, but it CANNOT generate genuine edge
  from odds alone — nothing can. Real edge requires data the market
  hasn't already priced: current form, injuries/suspensions, fixture
  congestion, or a validated statistical anomaly like your PVES
  La Liga home-favorite Elo edge. The `edge_pct` field below will be
  small and mostly noise unless you feed in independent signals.
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import math

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def poisson_pmf(lam: float, k: int) -> float:
    if k < 0 or lam <= 0:
        return 0.0
    return (math.exp(-lam) * (lam ** k)) / math.factorial(k)


def shin_devig(raw_probs: list[float], tol: float = 1e-10, max_iter: int = 100) -> tuple[list[float], float]:
    """
    Shin's (1993) method for removing bookmaker margin.
    Models the overround as partly caused by informed ('insider') money
    rather than assuming margin is spread proportionally across outcomes.
    Returns (true_probabilities, z) where z is the estimated insider
    trading proportion (also a rough proxy for how 'sharp' the market is —
    higher z can indicate a thinner / less efficient market).
    """
    beta = sum(raw_probs)  # overround, > 1
    pi = [p / beta * beta for p in raw_probs]  # keep raw, we normalize inside loop using beta directly

    lo, hi = 0.0, 0.5
    for _ in range(max_iter):
        z = (lo + hi) / 2
        try:
            p = [
                (math.sqrt(z * z + 4 * (1 - z) * (pi_i ** 2) / beta) - z) / (2 * (1 - z))
                if z < 1 else pi_i / beta
                for pi_i in pi
            ]
        except ValueError:
            hi = z
            continue
        total = sum(p)
        if abs(total - 1.0) < tol:
            return p, z
        if total > 1.0:
            lo = z
        else:
            hi = z
    return p, z


def dixon_coles_tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Low-score correlation correction (Dixon & Coles, 1997)."""
    if x == 0 and y == 0:
        return 1 - (lam * mu * rho)
    elif x == 0 and y == 1:
        return 1 + (lam * rho)
    elif x == 1 and y == 0:
        return 1 + (mu * rho)
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0


def score_matrix(h_xg: float, a_xg: float, rho: float, max_goals: int = 9):
    """Dixon-Coles-adjusted Poisson scoreline matrix."""
    matrix = [[0.0] * max_goals for _ in range(max_goals)]
    for h in range(max_goals):
        for a in range(max_goals):
            base = poisson_pmf(h_xg, h) * poisson_pmf(a_xg, a)
            matrix[h][a] = base * dixon_coles_tau(h, a, h_xg, a_xg, rho)

    total = sum(sum(row) for row in matrix)
    if total > 0:
        matrix = [[v / total for v in row] for row in matrix]
    return matrix


def outcome_probs_from_matrix(matrix):
    home = draw = away = 0.0
    for h, row in enumerate(matrix):
        for a, p in enumerate(row):
            if h > a:
                home += p
            elif h == a:
                draw += p
            else:
                away += p
    return home, draw, away


def kelly_fraction(model_prob: float, decimal_odds: float, kelly_multiplier: float = 0.25) -> float:
    """
    Fractional Kelly stake as a percentage of bankroll.
    kelly_multiplier < 1.0 (e.g. 0.25 = quarter-Kelly) is standard practice
    to reduce variance from model uncertainty — never bet full Kelly on a
    model you haven't extensively walk-forward validated.
    """
    b = decimal_odds - 1.0
    q = 1.0 - model_prob
    if b <= 0:
        return 0.0
    f = (b * model_prob - q) / b
    f = max(0.0, f)  # never bet negative EV
    return round(f * kelly_multiplier * 100, 2)


@app.get("/")
def home():
    return {"status": "Prediction Engine v3.0 — Shin de-vig + Dixon-Coles + Value Detection"}


@app.get("/predict")
def predict(
    home_team: str = Query("Home"),
    away_team: str = Query("Away"),
    home_odds: float = Query(2.0),
    draw_odds: float = Query(3.4),
    away_odds: float = Query(3.5),
    home_form_xg: float | None = Query(
        None, description="Independent signal: home team's actual recent xG-for average. "
                           "Without this, the model has no information beyond the odds."
    ),
    away_form_xg: float | None = Query(
        None, description="Independent signal: away team's actual recent xG-for average."
    ),
    home_form_weight: float = Query(
        0.5, ge=0.0, le=1.0,
        description="How much to trust form_xg vs market-implied xG, if form_xg is supplied."
    ),
    rho: float = Query(-0.12, description="Dixon-Coles low-score correlation parameter (typically -0.10 to -0.20)"),
    min_edge_pct: float = Query(3.0, description="Minimum model-vs-market edge (%) required to flag a pick as a play"),
):
    home_odds = max(1.01, home_odds)
    draw_odds = max(1.01, draw_odds)
    away_odds = max(1.01, away_odds)

    # 1. De-vig via Shin's method (corrects favorite-longshot bias)
    raw = [1.0 / home_odds, 1.0 / draw_odds, 1.0 / away_odds]
    margin_pct = (sum(raw) - 1.0) * 100
    p_market, shin_z = shin_devig(raw)
    p_home_market, p_draw_market, p_away_market = p_market

    # 2. xG estimate: market-implied by default, blended with real form data if supplied
    market_ratio = away_odds / home_odds
    base_xg = 1.35
    h_xg_market = max(0.35, base_xg * (market_ratio ** 0.30))
    a_xg_market = max(0.35, base_xg * ((1.0 / market_ratio) ** 0.30))

    has_independent_signal = home_form_xg is not None and away_form_xg is not None
    if has_independent_signal:
        h_xg = home_form_weight * home_form_xg + (1 - home_form_weight) * h_xg_market
        a_xg = home_form_weight * away_form_xg + (1 - home_form_weight) * a_xg_market
    else:
        h_xg, a_xg = h_xg_market, a_xg_market

    # 3. Dixon-Coles-adjusted Poisson matrix
    matrix = score_matrix(h_xg, a_xg, rho)
    model_home, model_draw, model_away = outcome_probs_from_matrix(matrix)

    # 4. Final probabilities:
    #    - With independent signal: trust the model more (it has real info)
    #    - Without independent signal: model is circular w.r.t. market, so
    #      weight it low — it only contributes Dixon-Coles score-correlation
    #      shape, not new information
    model_weight = 0.55 if has_independent_signal else 0.15
    final_home = p_home_market * (1 - model_weight) + model_home * model_weight
    final_draw = p_draw_market * (1 - model_weight) + model_draw * model_weight
    final_away = p_away_market * (1 - model_weight) + model_away * model_weight
    total = final_home + final_draw + final_away
    final_home, final_draw, final_away = (final_home / total, final_draw / total, final_away / total)

    outcomes = [
        {"market": f"Home Win ({home_team})", "code": "1", "model_prob": final_home,
         "market_prob": p_home_market, "odds": home_odds},
        {"market": "Draw", "code": "X", "model_prob": final_draw,
         "market_prob": p_draw_market, "odds": draw_odds},
        {"market": f"Away Win ({away_team})", "code": "2", "model_prob": final_away,
         "market_prob": p_away_market, "odds": away_odds},
    ]

    # 5. Edge = model probability minus market's own de-vigged probability.
    #    This is the actual test of whether there's value, not just which
    #    outcome is most likely.
    for o in outcomes:
        o["edge_pct"] = round((o["model_prob"] - o["market_prob"]) * 100, 2)
        o["kelly_stake_pct"] = kelly_fraction(o["model_prob"], o["odds"])

    outcomes.sort(key=lambda x: x["edge_pct"], reverse=True)
    best = outcomes[0]

    if not has_independent_signal:
        verdict = "NO PLAY — no independent signal supplied; edge shown is noise, not a real edge"
    elif best["edge_pct"] >= min_edge_pct and best["kelly_stake_pct"] > 0:
        verdict = f"PLAY — {best['market']} shows {best['edge_pct']:.1f}pt edge vs market"
    else:
        verdict = "NO PLAY — insufficient edge vs market price"

    return {
        "match": f"{home_team} vs {away_team}",
        "verdict": verdict,
        "has_independent_signal": has_independent_signal,
        "market_margin_pct": round(margin_pct, 2),
        "shin_insider_proportion_z": round(shin_z, 4),
        "calculated_xG": {"home": round(h_xg, 2), "away": round(a_xg, 2)},
        "outcomes": [
            {
                "market": o["market"],
                "model_prob_pct": round(o["model_prob"] * 100, 1),
                "market_prob_pct": round(o["market_prob"] * 100, 1),
                "edge_pct": o["edge_pct"],
                "odds": o["odds"],
                "suggested_stake_pct_bankroll": o["kelly_stake_pct"],
            }
            for o in outcomes
        ],
    }
