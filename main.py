import json
import math
import os
import re
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Mathayo 154 Genius Model API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Load the offline ratings snapshot (built by build_ratings_snapshot.py
# and committed next to this file). If it's missing, the API still runs
# but falls back to pure market-devig, same as before - and says so.
# ---------------------------------------------------------------------
RATINGS_PATH = os.path.join(os.path.dirname(__file__), "team_ratings.json")
_snapshot = {"_meta": {}, "teams": {}}
if os.path.exists(RATINGS_PATH):
    with open(RATINGS_PATH) as f:
        _snapshot = json.load(f)

TEAM_RATINGS = _snapshot.get("teams", {})
AVG_HOME_GOALS = _snapshot.get("_meta", {}).get("avg_home_goals", 1.45)
AVG_AWAY_GOALS = _snapshot.get("_meta", {}).get("avg_away_goals", 1.15)
RATINGS_AS_OF = _snapshot.get("_meta", {}).get("as_of")


def _normalize(name: str) -> str:
    """Loose match so 'Man United', 'Man Utd', 'Manchester United FC' etc.
    have a fighting chance of hitting the same key. This is intentionally
    simple - if it misses, we fall back to a neutral (0.0) rating rather
    than silently matching the wrong team."""
    s = name.lower().strip()
    s = re.sub(r"\b(fc|cf|afc|sc)\b", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


_NORMALIZED_INDEX = {_normalize(t): t for t in TEAM_RATINGS}


def lookup_rating(team_name: str):
    key = _normalize(team_name)
    if key in _NORMALIZED_INDEX:
        real_key = _NORMALIZED_INDEX[key]
        r = TEAM_RATINGS[real_key]
        return r["home_rating"], r["away_rating"], True
    return 0.0, 0.0, False  # unknown team -> neutral rating, flagged


def g_inv(y: float) -> float:
    """Same inverse mapping as pi_ratings.py: rating scale -> goal-diff scale."""
    if y >= 0:
        return 10 ** (y / 3) - 1
    return 1 - 10 ** (-y / 3)


def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def match_outcome_probs(home_exp: float, away_exp: float, max_goals: int = 10):
    """Independent-Poisson scoreline grid -> P(home win), P(draw), P(away win).
    This is a standard, simple baseline (no correlation term) - good enough
    to generate a genuinely independent estimate, not a claim of being
    state-of-the-art. Compare against your backtest.py numbers before
    trusting it for real money."""
    p_home = p_draw = p_away = 0.0
    home_pmf = [poisson_pmf(i, home_exp) for i in range(max_goals + 1)]
    away_pmf = [poisson_pmf(j, away_exp) for j in range(max_goals + 1)]
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = home_pmf[i] * away_pmf[j]
            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p
    total = p_home + p_draw + p_away  # tiny leftover mass beyond max_goals
    return p_home / total, p_draw / total, p_away / total


def kelly_fraction(model_prob: float, decimal_odds: float, cap_fraction: float = 0.25) -> float:
    """Fractional Kelly (25% of full Kelly, a common safety haircut),
    clipped at 0 (never suggest staking on negative edge)."""
    b = decimal_odds - 1
    if b <= 0:
        return 0.0
    full_kelly = (model_prob * decimal_odds - 1) / b
    return max(0.0, full_kelly * cap_fraction)


@app.get("/")
@app.post("/")
def root():
    return {
        "status": "online",
        "model": "Mathayo 154 Genius Model v3.4",
        "ratings_loaded": len(TEAM_RATINGS),
        "ratings_as_of": RATINGS_AS_OF,
    }


@app.get("/predict")
@app.post("/predict")
async def predict(
    request: Request,
    home_team: str = None,
    away_team: str = None,
    home_odds: float = 2.20,
    draw_odds: float = 3.40,
    away_odds: float = 3.10,
):
    try:
        body = await request.json()
        home_team = body.get("home_team", home_team)
        away_team = body.get("away_team", away_team)
        home_odds = float(body.get("home_odds", home_odds))
        draw_odds = float(body.get("draw_odds", draw_odds))
        away_odds = float(body.get("away_odds", away_odds))
    except Exception:
        pass

    h = home_team or "Home Team"
    a = away_team or "Away Team"

    # --- Market side: devigged implied probabilities (unchanged) ---
    h_prob = 1 / home_odds
    d_prob = 1 / draw_odds
    a_prob = 1 / away_odds
    margin = h_prob + d_prob + a_prob
    mkt_h, mkt_d, mkt_a = h_prob / margin, d_prob / margin, a_prob / margin

    # --- Model side: this is the part that was missing before ---
    h_home_rating, _, h_found = lookup_rating(h)
    _, a_away_rating, a_found = lookup_rating(a)
    pred_gd = g_inv(h_home_rating) - g_inv(a_away_rating)

    home_exp = max(0.2, AVG_HOME_GOALS + pred_gd / 2)
    away_exp = max(0.2, AVG_AWAY_GOALS - pred_gd / 2)
    model_h, model_d, model_a = match_outcome_probs(home_exp, away_exp)

    both_known = h_found and a_found
    edge_h = (model_h - mkt_h) * 100
    edge_d = (model_d - mkt_d) * 100
    edge_a = (model_a - mkt_a) * 100

    best_edge = max(edge_h, edge_d, edge_a)
    if not both_known:
        verdict = "NO PLAY (team not in ratings snapshot - see 'ratings_status')"
    elif best_edge > 3.0:  # threshold in percentage points, tune via backtest
        pick = {"H": "HOME", "D": "DRAW", "A": "AWAY"}[
            max([("H", edge_h), ("D", edge_d), ("A", edge_a)], key=lambda t: t[1])[0]
        ]
        verdict = f"RECOMMENDED PLAY: {pick} (edge {best_edge:.1f} pp)"
    else:
        verdict = "NO PLAY"

    return {
        "match": f"{h} vs {a}",
        "verdict": verdict,
        "ratings_status": {
            "home_team_found": h_found,
            "away_team_found": a_found,
            "ratings_as_of": RATINGS_AS_OF,
        },
        "safety_grade": "GRADE A+" if (margin - 1) * 100 < 6 else "GRADE B",
        "market_margin_pct": round((margin - 1) * 100, 2),
        "calculated_xG": [round(home_exp, 2), round(away_exp, 2)],
        "outcomes": [
            {
                "market": "1", "model_prob_pct": round(model_h * 100, 1),
                "market_prob_pct": round(mkt_h * 100, 1), "edge_pct": round(edge_h, 1),
                "suggested_stake_pct_bankroll": round(kelly_fraction(model_h, home_odds) * 100, 2),
            },
            {
                "market": "X", "model_prob_pct": round(model_d * 100, 1),
                "market_prob_pct": round(mkt_d * 100, 1), "edge_pct": round(edge_d, 1),
                "suggested_stake_pct_bankroll": round(kelly_fraction(model_d, draw_odds) * 100, 2),
            },
            {
                "market": "2", "model_prob_pct": round(model_a * 100, 1),
                "market_prob_pct": round(mkt_a * 100, 1), "edge_pct": round(edge_a, 1),
                "suggested_stake_pct_bankroll": round(kelly_fraction(model_a, away_odds) * 100, 2),
            },
        ],
    }
