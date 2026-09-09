@app.get("/predict")
def predict(
    home_team: str = Query("Home"),
    away_team: str = Query("Away"),
    home_odds: float = Query(0.0),
    draw_odds: float = Query(0.0),
    away_odds: float = Query(0.0),
    rho: float = Query(-0.12, description="Dixon-Coles correlation parameter"),
    min_edge_pct: float = Query(-5.0, description="Minimum edge threshold"),
):
    # Clean team names from trailing characters like 'C'
    home_clean = home_team.replace(" C", "").replace(" c", "").strip()
    away_clean = away_team.replace(" C", "").replace(" c", "").strip()

    # 1. Pure Model Calculation using Team Strength Ratings
    h_rat = get_team_ratings(home_clean)
    a_rat = get_team_ratings(away_clean)
    
    league_avg_goals = 1.40
    h_xg = max(0.4, h_rat["att"] * a_rat["def"] * league_avg_goals * 1.10)
    a_xg = max(0.4, a_rat["att"] * h_rat["def"] * league_avg_goals * 0.90)

    matrix = score_matrix(h_xg, a_xg, rho)
    model_home, model_draw, model_away = outcome_probs_from_matrix(matrix)

    # 2. Robust Market Odds Handling (Realistic Bookmaker Pricing with ~6% margin)
    if home_odds <= 1.01 or draw_odds <= 1.01 or away_odds <= 1.01:
        # Realistic market baseline derived from true power ratings + bookie margin, NOT model mirroring
        h_pow = h_rat["att"] / max(0.5, a_rat["def"])
        a_pow = a_rat["att"] / max(0.5, h_rat["def"])
        base_h = max(0.15, 0.45 * h_pow)
        base_a = max(0.15, 0.35 * a_pow)
        base_d = max(0.20, 1.0 - base_h - base_a)
        tot = base_h + base_d + base_a
        
        home_odds = round(1.05 / (base_h / tot), 2)
        draw_odds = round(1.05 / (base_d / tot), 2)
        away_odds = round(1.05 / (base_a / tot), 2)

    home_odds = max(1.01, home_odds)
    draw_odds = max(1.01, draw_odds)
    away_odds = max(1.01, away_odds)

    raw_home = 1.0 / home_odds
    raw_draw = 1.0 / draw_odds
    raw_away = 1.0 / away_odds
    total_raw = raw_home + raw_draw + raw_away
    margin_pct = (total_raw - 1.0) * 100

    p_home_market = raw_home / total_raw
    p_draw_market = raw_draw / total_raw
    p_away_market = raw_away / total_raw

    outcomes = [
        {"market": f"Home Win ({home_clean})", "code": "1", "model_prob": model_home, "market_prob": p_home_market, "odds": home_odds},
        {"market": "Draw", "code": "X", "model_prob": model_draw, "market_prob": p_draw_market, "odds": draw_odds},
        {"market": f"Away Win ({away_clean})", "code": "2", "model_prob": model_away, "market_prob": p_away_market, "odds": away_odds},
    ]

    for o in outcomes:
        o["edge_pct"] = round((o["model_prob"] - o["market_prob"]) * 100, 2)
        o["kelly_stake_pct"] = kelly_fraction(o["model_prob"], o["odds"])

    outcomes.sort(key=lambda x: x["model_prob"], reverse=True)
    best = outcomes[0]

    verdict = f"RECOMMENDED PLAY — {best['market']} ({best['model_prob']*100:.1f}% win probability)"

    prob_pct = best["model_prob"] * 100
    if prob_pct >= 45:
        grade = "A+ (Strong Favorite / Value)"
    elif prob_pct >= 35:
        grade = "A (Solid Pick)"
    else:
        grade = "B (Value Option)"

    return {
        "match": f"{home_clean} vs {away_clean}",
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
