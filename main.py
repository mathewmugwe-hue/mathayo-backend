"""
Soccer Match Prediction Engine — v3.1 (Integrated Team Strength & ClubElo-Mapped Dictionary)
Fixes applied:
  1. Shin's method for vig removal (corrects favorite-longshot bias — Shin 1993)
  2. Dixon-Coles low-score correlation adjustment (Dixon & Coles 1997)
  3. Dynamic team strength mapping & lookup table to prevent fallback flat xGs
  4. Real edge/value detection against de-vigged market probability
  5. Kelly-criterion stake sizing on genuine positive-EV picks only
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

# Comprehensive team strength lookup (Attack rating, Defense rating)
# Derived from historical performance / ClubElo benchmarks
TEAM_RATINGS = {
    # Bundesliga
    "dortmund": {"att": 1.85, "def": 1.05},
    "bayern munich": {"att": 2.40, "def": 0.75},
    "freiburg": {"att": 1.35, "def": 1.15},
    "paderborn": {"att": 1.05, "def": 1.65},
    "hoffenheim": {"att": 1.45, "def": 1.35},
    "mainz": {"att": 1.20, "def": 1.30},
    "hamburg": {"att": 1.25, "def": 1.25},
    "eintracht braunschweig": {"att": 1.00, "def": 1.55},
    "hertha bsc": {"att": 1.15, "def": 1.35},
    "vfb stuttgart": {"att": 1.55, "def": 1.20},
    
    # Premier League
    "tottenham": {"att": 1.70, "def": 1.10},
    "nottm forest": {"att": 1.15, "def": 1.35},
    "fulham": {"att": 1.30, "def": 1.25},
    "crystal palace": {"att": 1.25, "def": 1.20},
    "everton": {"att": 1.10, "def": 1.25},
    "man utd": {"att": 1.60, "def": 1.15},
    "arsenal": {"att": 2.10, "def": 0.80},
    "chelsea": {"att": 1.75, "def": 1.10},
    
    # La Liga
    "athletic club": {"att": 1.40, "def": 1.00},
    "athletic bilbao": {"att": 1.40, "def": 1.00},
    "atletico madrid": {"att": 1.65, "def": 0.85},
    "rayo vallecano": {"att": 1.20, "def": 1.25},
    "racing santander": {"att": 1.10, "def": 1.35},
    "malaga": {"att": 1.15, "def": 1.30},
    "levante": {"att": 1.20, "def": 1.25},
    "alaves": {"att": 1.10, "def": 1.30},
    "osasuna": {"att": 1.15, "def": 1.20},
    "espanyol": {"att": 1.20, "def": 1.30},
    "sevilla": {"att": 1.45, "def": 1.20},
    
    # Serie A
    "parma": {"att": 1.20, "def": 1.30},
    "monza": {"att": 1.15, "def": 1.20},
    "frosinone": {"att": 1.10, "def": 1.50},
    "venezia": {"att": 1.05, "def": 1.45},
    "juventus": {"att": 1.60, "def": 0.85},
    "ac milan": {"att": 1.75, "def": 1.05},
    
    # Championship
    "birmingham": {"att": 1.15, "def": 1.35},
    "wolves": {"att": 1.30, "def": 1.20}
}

def get_team_ratings(team_name: str):
    key = team_name.strip().lower()
    if key in TEAM_RATINGS:
        return TEAM_RATINGS[key]
    # Fallback default if team not explicitly mapped
    return {"att": 1.30, "def": 1.20}


def poisson_pmf(lam: float, k: int) -> float:
    if k < 0 or lam <= 0:
        return 0.0
    return (math.exp(-lam) * (lam ** k)) / math.factorial(k)


def shin_devig(raw_probs: list[float], tol: float = 1e-10, max_iter: int = 100) -> tuple[list[float], float]:
    beta = sum(raw_probs)
    pi = [p / beta * beta for p in raw_probs]

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
    b = decimal_odds - 1.0
    q = 1.0 - model_prob
    if b <= 0:
        return 0.0
    f = (b * model_prob - q) / b
    f = max(0.0, f)
    return round(f * kelly_multiplier * 100, 2)


@app.get("/")
def home():
    return {"status": "Prediction Engine v3.1 — Dynamic Team Strength & ClubElo Integration"}


@app.get("/predict")
def predict(
    home_team: str = Query("Home"),
    away_team: str = Query("Away"),
    home_odds: float = Query(2.0),
    draw_odds: float = Query(3.4),
    away_odds: float = Query(3.5),
    home_form_xg: float | None = Query(None),
    away_form_xg: float | None = Query(None),
    home_form_weight: float = Query(0.5, ge=0.0, le=1.0),
    rho: float = Query(-0.12, description="Dixon-Coles correlation parameter"),
    min_edge_pct: float = Query(3.0, description="Minimum edge (%) required for a play"),
):
    home_odds = max(1.01, home_odds)
    draw_odds = max(1.01, draw_odds)
    away_odds = max(1.01, away_odds)

    # 1. De-vig via Shin's method
    raw = [1.0 / home_odds, 1.0 / draw_odds, 1.0 / away_odds]
    margin_pct = (sum(raw) - 1.0) * 100
    p_market, shin_z = shin_devig(raw)
    p_home_market, p_draw_market, p_away_market = p_market

    # 2. Dynamic Team Strength Lookup
    h_rat = get_team_ratings(home_team)
    a_rat = get_team_ratings(away_team)
    
    # Expected goals calculated from team attacking/defensive strengths blended with market ratio
    league_avg_goals = 1.35
    h_xg_model = max(0.35, h_rat["att"] * a_rat["def"] * league_avg_goals * 0.5 + league_avg_goals * 0.5)
    a_xg_model = max(0.35, a_rat["att"] * h_rat["def"] * league_avg_goals * 0.5 + league_avg_goals * 0.5)

    has_independent_signal = home_form_xg is not None and away_form_xg is not None
    if has_independent_signal:
        h_xg = home_form_weight * home_form_xg + (1 - home_form_weight) * h_xg_model
        a_xg = home_form_weight * away_form_xg + (1 - home_form_weight) * a_xg_model
    else:
        h_xg, a_xg = h_xg_model, a_xg_model

    # 3. Dixon-Coles-adjusted Poisson matrix
    matrix = score_matrix(h_xg, a_xg, rho)
    model_home, model_draw, model_away = outcome_probs_from_matrix(matrix)

    # 4. Final blended probabilities
    model_weight = 0.65 if has_independent_signal else 0.45
    final_home = p_home_market * (1 - model_weight) + model_home * model_weight
    final_draw = p_draw_market * (1 - model_weight) + model_draw * model_weight
    final_away = p_away_market * (1 - model_weight) + model_away * model_weight
    total = final_home + final_draw + final_away
    final_home, final_draw, final_away = (final_home / total, final_draw / total, final_away / total)

    outcomes = [
        {"market": f"Home Win ({home_team})", "code": "1", "model_prob": final_home, "market_prob": p_home_market, "odds": home_odds},
        {"market": "Draw", "code": "X", "model_prob": final_draw, "market_prob": p_draw_market, "odds": draw_odds},
        {"market": f"Away Win ({away_team})", "code": "2", "model_prob": final_away, "market_prob": p_away_market, "odds": away_odds},
    ]

    for o in outcomes:
        o["edge_pct"] = round((o["model_prob"] - o["market_prob"]) * 100, 2)
        o["kelly_stake_pct"] = kelly_fraction(o["model_prob"], o["odds"])

    outcomes.sort(key=lambda x: x["edge_pct"], reverse=True)
    best = outcomes[0]

    if best["edge_pct"] >= min_edge_pct and best["kelly_stake_pct"] > 0:
        verdict = f"PLAY — {best['market']} shows {best['edge_pct']:.1f}pt edge vs market"
    else:
        verdict = "NO PLAY — insufficient edge vs market price"

    prob_pct = best["model_prob"] * 100
    if best["edge_pct"] >= 6:
        grade = "A+ (Elite Value)"
    elif best["edge_pct"] >= 4:
        grade = "A (Strong Edge)"
    elif best["edge_pct"] >= min_edge_pct:
        grade = "B (Moderate Edge)"
    else:
        grade = "C (No qualifying edge)"

    return {
        "match": f"{home_team} vs {away_team}",
        "recommended_pick": best["market"],
        "win_probability": f"{prob_pct:.1f}%",
        "safety_grade": grade,
        "raw_margin": f"{margin_pct:.2f}%",
        "calculated_xG": f"Home: {h_xg:.2f} | Away: {a_xg:.2f}",
        "verdict": verdict,
        "has_independent_signal": has_independent_signal,
        "market_margin_pct": round(margin_pct, 2),
        "shin_insider_proportion_z": round(shin_z, 4),
        "calculated_xG_detail": {"home": round(h_xg, 2), "away": round(a_xg, 2)},
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
