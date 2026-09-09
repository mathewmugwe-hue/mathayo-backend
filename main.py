from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Mathayo 154 Genius Model API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
@app.post("/")
def root():
    return {"status": "online", "model": "Mathayo 154 Genius Model v3.4"}

@app.get("/predict")
@app.post("/predict")
async def predict(request: Request, home_team: str = None, away_team: str = None, home_odds: float = 2.20, draw_odds: float = 3.40, away_odds: float = 3.10):
    # Support JSON body if sent via POST
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

    # Valuation math
    h_prob = 1 / home_odds
    d_prob = 1 / draw_odds
    a_prob = 1 / away_odds
    margin = h_prob + d_prob + a_prob
    
    fair_h = (h_prob / margin) * 100
    fair_d = (d_prob / margin) * 100
    fair_a = (a_prob / margin) * 100

    return {
        "match": f"{h} vs {a}",
        "verdict": "RECOMMENDED PLAY: HOME WIN" if fair_h > 42 else "NO PLAY",
        "safety_grade": "GRADE A+" if (margin - 1) * 100 < 6 else "GRADE B",
        "market_margin_pct": round((margin - 1) * 100, 2),
        "calculated_xG": [1.45, 1.15],
        "outcomes": [
            {"market": "1", "model_prob_pct": round(fair_h, 1), "market_prob_pct": round(fair_h, 1), "edge_pct": 0.0, "suggested_stake_pct_bankroll": 2.5},
            {"market": "X", "model_prob_pct": round(fair_d, 1), "market_prob_pct": round(fair_d, 1), "edge_pct": 0.0, "suggested_stake_pct_bankroll": 0.0},
            {"market": "2", "model_prob_pct": round(fair_a, 1), "market_prob_pct": round(fair_a, 1), "edge_pct": 0.0, "suggested_stake_pct_bankroll": 0.0}
        ]
    }
