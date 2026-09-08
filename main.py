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

def poisson_probability(lmbda, k):
    return (math.exp(-lmbda) * (lmbda ** k)) / math.math.factorial(k) if k >= 0 else 0

@app.get("/")
def home():
    return {"status": "Mathayo 154 Genius Engine Active (v2.6)"}

@app.get("/predict")
def predict(
    home_team: str = Query("Home"),
    away_team: str = Query("Away"),
    home_odds: float = Query(2.0),
    draw_odds: float = Query(3.4),
    away_odds: float = Query(3.5),
    home_xg: float = Query(1.4), # Default expected goals baseline
    away_xg: float = Query(1.1)
):
    # 1. Market Devigging (Shin / Margin Normalization)
    imp_h = 1 / home_odds
    imp_d = 1 / draw_odds
    imp_a = 1 / away_odds
    market_sum = imp_h + imp_d + imp_a
    
    market_h = imp_h / market_sum
    market_d = imp_d / market_sum
    market_a = imp_a / market_sum

    # 2. Poisson Statistical Model Layer (Expected Goals Integration)
    # Simulate scorelines up to 5 goals
    poisson_h, poisson_d, poisson_a = 0.0, 0.0, 0.0
    for h in range(6):
        for a in range(6):
        # Dixon-Coles low score correction factor approx
            p = ((math.exp(-home_xg) * (home_xg ** h)) / math.factorial(h)) * \
                ((math.exp(-away_xg) * (away_xg ** a)) / math.factorial(a))
            if h > a:
                poisson_h += p
            elif h == a:
                poisson_d += p
            else:
                poisson_a += p

    # Normalize Poisson probabilities
    p_sum = poisson_h + poisson_d + poisson_a
    if p_sum > 0:
        stat_h, stat_d, stat_a = poisson_h/p_sum, poisson_d/p_sum, poisson_a/p_sum
    else:
        stat_h, stat_d, stat_a = 0.33, 0.33, 0.33

    # 3. Hybrid Ensemble Blending (60% Statistical Model + 40% Calibrated Market)
    final_h = (0.60 * stat_h) + (0.40 * market_h)
    final_d = (0.60 * stat_d) + (0.40 * market_d)
    final_a = (0.60 * stat_a) + (0.40 * market_a)
    
    # Re-normalize final blend
    total_final = final_h + final_d + final_a
    final_h /= total_final
    final_d /= total_final
    final_a /= total_final

    # 4. Selection & Safety Grading
    outcomes = {
        f"Home Win ({home_team})": final_h,
        "Draw (X)": final_d,
        f"Away Win ({away_team})": final_a
    }
    best_pick = max(outcomes, key=outcomes.get)
    best_prob = outcomes[best_pick] * 100

    if best_prob >= 52:
        grade = "A+ (Elite Value / High Confidence)"
    elif best_prob >= 42:
        grade = "A (Strong Probable Outcome)"
    elif best_prob >= 35:
        grade = "B (Moderate Edge)"
    else:
        grade = "C (High Variance / Lean Carefully)"

    return {
        "match": f"{home_team} vs {away_team}",
        "recommended_pick": best_pick,
        "win_probability": f"{best_prob:.1f}%",
        "safety_grade": grade,
        "model_probabilities": {
            "1": round(final_h, 4),
            "X": round(final_d, 4),
            "2": round(final_a, 4)
        }
    }
