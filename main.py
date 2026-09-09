"""
Soccer Match Prediction Engine — v3.3 (Unlocked Pure Value & Action Generator)
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
TEAM_RATINGS = {
    # Bundesliga
    "dortmund": {"att": 1.95, "def": 0.95},
    "bayern munich": {"att": 2.50, "def": 0.70},
    "freiburg": {"att": 1.40, "def": 1.10},
    "paderborn": {"att": 0.95, "def": 1.65},
    "hoffenheim": {"att": 1.55, "def": 1.20},
    "mainz": {"att": 1.25, "def": 1.25},
    "hamburg": {"att": 1.20, "def": 1.30},
    "eintracht braunschweig": {"att": 0.90, "def": 1.60},
    "hertha bsc": {"att": 1.10, "def": 1.40},
    "vfb stuttgart": {"att": 1.60, "def": 1.15},
    
    # Premier League
    "tottenham": {"att": 1.80, "def": 1.00},
    "nottm forest": {"att": 1.25, "def": 1.25},
    "fulham": {"att": 1.30, "def": 1.25},
    "crystal palace": {"att": 1.20, "def": 1.25},
    "everton": {"att": 1.10, "def": 1.30},
    "man utd": {"att": 1.70, "def": 1.05},
    "arsenal": {"att": 2.15, "def": 0.75},
    "chelsea": {"att": 1.80, "def": 1.05},
    
    # La Liga
    "athletic club": {"att": 1.50, "def": 0.90},
    "athletic bilbao": {"att": 1.50, "def": 0.90},
    "atletico madrid": {"att": 1.75, "def": 0.75},
    "rayo vallecano": {"att": 1.20, "def": 1.25},
    "racing santander": {"att": 1.05, "def": 1.40},
    "malaga": {"att": 1.10, "def": 1.35},
    "levante": {"att": 1.15, "def": 1.30},
    "alaves": {"att": 1.10, "def": 1.30},
    "osasuna": {"att": 1.15, "def": 1.20},
    "espanyol": {"att": 1.20, "def": 1.30},
    "sevilla": {"att": 1.55, "def": 1.10},
    
    # Serie A
    "parma": {"att": 1.20, "def": 1.30},
    "monza": {"att": 1.15, "def": 1.20},
    "frosinone": {"att": 1.05, "def": 1.55},
    "venezia": {"att": 1.00, "def": 1.50},
    "juventus": {"att": 1.70, "def": 0.75},
    "ac milan": {"att": 1.85, "def": 0.95},
    
    # Championship
    "birmingham": {"att": 1.10, "def": 1.40},
    "wolves": {"att": 1.30, "def": 1.20}
}

def get_team_ratings(team_name: str):
    key = team_name.strip().lower()
    if key in TEAM_RATINGS:
        return TEAM_RATINGS[key]
    return {"att": 1.20, "def": 1.20}

def poisson_pmf(lam: float, k: int) -> float:
    if k < 0 or lam <= 0:
        return 0.0
    return (math.exp(-lam) * (lam ** k)) / math.factorial(k)

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

def kelly_fraction(model_prob: float, decimal_odds: float, kelly_multiplier: float = 0.35) -> float:
    b = decimal_odds - 1.0
    q = 1.0 - model_prob
    if b <= 0:
        return 0.0
    f = (b * model_prob - q) / b
    f = max(0.0, f)
    return round(f * kelly_multiplier * 100, 2)

@app.get("/")
def home():
    return {"status": "Prediction Engine v3.3 — Unlocked & Active"}

@app.get("/predict")
def predict(
    home_team: str = Query("Home"),
    away_team: str = Query("Away"),
    home_odds: float = Query(2.40),
    draw_odds: float = Query(3.30),
    away_odds: float = Query(3.10),
    rho: float = Query(-0.12, description="Dixon-Coles correlation parameter"),
    min_edge_pct: float = Query(0.1, description="Minimum edge (%) required for a play"),
):
    home_odds = max(1.01, home_odds)
    draw_odds = max(1.01, draw_odds)
    away_odds = max(1.01, away_odds)

    # Simple market probabilities from odds (with standard bookmaker margin)
    raw_home = 1.0 / home_odds
    raw_draw = 1.0 / draw_odds
    raw_away = 1.0 / away_odds
    total_raw = raw_home + raw_draw + raw_away
    margin_pct = (total_raw - 1.0) * 100

    p_home_market = raw_home / total_raw
    p_draw_market = raw_draw / total_raw
    p_away_market = raw_away / total_raw

    # Pure Model Calculation using Team Strength Ratings
    h_rat = get_team_ratings(home_team)
    a_rat = get_team_ratings(away_team)
    
    league_avg_goals = 1.40
    h_xg = max(0.5, h_rat["att"] * a_rat["def"] * league_avg_goals * 1.10) # Home advantage factor
    a_xg = max(0.5, a_rat["att"] * h_rat["def"] * league_avg_goals * 0.90)

    matrix = score_matrix(h_xg, a_xg, rho)
    model_home, model_draw, model_away = outcome_probs_from_matrix(matrix)

    outcomes = [
        {"market": f"Home Win ({home_team})", "code": "1", "model_prob": model_home, "market_prob": p_home_market, "odds": home_odds},
        {"market": "Draw", "code": "X", "model_prob": model_draw, "market_prob": p_draw_market, "odds": draw_odds},
        {"market": f"Away Win ({away_team})", "code": "2", "model_prob": model_away, "market_prob": p_away_market, "odds": away_odds},
    ]

    for o in outcomes:
        o["edge_pct"] = round((o["model_prob"] - o["market_prob"]) * 100, 2)
        o["kelly_stake_pct"] = kelly_fraction(o["model_prob"], o["odds"])

    outcomes.sort(key=lambda x: x["edge_pct"], reverse=True)
    best = outcomes[0]

    # Force a play whenever edge is positive or meets the minimal threshold
    if best["edge_pct"] >= min_edge_pct and best["odds"] > 1.05:
        verdict = f"RECOMMENDED PLAY — {best['market']} offers a {best['edge_pct']:.1f}% value edge."
    else:
        verdict = "NO PLAY — market price is fully efficient."

    prob_pct = best["model_prob"] * 100
    if best["edge_pct"] >= 4:
        grade = "A+ (High Value Play)"
    elif best["edge_pct"] >= 2:
        grade = "A (Solid Edge)"
    else:
        grade = "B (Marginal Value)"

    return {
        "match": f"{home_team} vs {away_team}",
        "recommended_pick": best["market"],
        "win_probability": f"{prob_pct:.1f}%",
        "safety_grade": grade,
        "calculated_xG": f"Home: {h_xg:.2f} | Away: {a_xg:.2f}",
        "verdict": verdict,
        "market_margin_pct": round(margin_pct, 2),
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
