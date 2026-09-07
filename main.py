from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import math

app = FastAPI()

# Allow frontend dashboard calls from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "Mathayo Engine Online", "version": "1.0"}

@app.get("/predict")
def predict_match(home_team: str, away_team: str, home_odds: float, draw_odds: float, away_odds: float):
    # Example: Pure mathematical devigging (Removing Bookmaker Margin)
    implied_h = 1 / home_odds
    implied_d = 1 / draw_odds
    implied_a = 1 / away_odds
    total_margin = implied_h + implied_d + implied_a

    # True Fair Probabilities
    fair_h = round(implied_h / total_margin, 4)
    fair_d = round(implied_d / total_margin, 4)
    fair_a = round(implied_a / total_margin, 4)

    return {
        "match": f"{home_team} vs {away_team}",
        "raw_margin": f"{round((total_margin - 1) * 100, 2)}%",
        "fair_probabilities": {
            "1": fair_h,
            "X": fair_d,
            "2": fair_a
        }
    }
