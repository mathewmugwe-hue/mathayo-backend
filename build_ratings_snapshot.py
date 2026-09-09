"""
Run this LOCALLY, not on Render. It walks your full historical match file
through pi_ratings.py exactly like backtest_v2.py does, but instead of
scoring a model it just keeps the FINAL rating per team (i.e. "as of the
most recent match in the file") and a couple of league-average constants,
and dumps that to team_ratings.json.

That JSON is what main.py will load at boot - a live web service can't
re-run this walk-forward computation on every request (or even on every
boot, cheaply), so you commit the snapshot and refresh it periodically
(weekly is plenty during a season) by re-running this script and pushing
the updated JSON.

Usage:
    python build_ratings_snapshot.py
    # writes team_ratings.json in the current directory - copy/commit it
    # into the mathayo-backend repo next to main.py
"""
import json
import pandas as pd
from pi_ratings import PiRatings

TOP5 = ['E0', 'D1', 'SP1', 'I1', 'F1']

df = pd.read_csv('matches_full.csv', low_memory=False)
df = df[df['Division'].isin(TOP5)].copy()
df['MatchDate'] = pd.to_datetime(df['MatchDate'], errors='coerce')
df = df.dropna(subset=['MatchDate', 'HomeTeam', 'AwayTeam', 'FTHome', 'FTAway'])
df = df.sort_values('MatchDate').reset_index(drop=True)

pr = PiRatings(lr_home=0.06, cross_ratio=0.35)
for row in df.itertuples():
    pr.update(row.HomeTeam, row.AwayTeam, row.FTHome, row.FTAway)
    # (no need to read predictions here - we only want ratings AFTER the
    # last match, i.e. current form, for use on upcoming fixtures)

teams = set(pr.home_rating) | set(pr.away_rating)
ratings = {
    team: {
        "home_rating": pr.home_rating.get(team, 0.0),
        "away_rating": pr.away_rating.get(team, 0.0),
    }
    for team in teams
}

snapshot = {
    "_meta": {
        "as_of": str(df['MatchDate'].max().date()),
        "n_matches_used": len(df),
        # league-average goals, used by main.py as the Poisson base rate
        "avg_home_goals": round(float(df['FTHome'].mean()), 3),
        "avg_away_goals": round(float(df['FTAway'].mean()), 3),
    },
    "teams": ratings,
}

with open("team_ratings.json", "w") as f:
    json.dump(snapshot, f, indent=2)

print(f"Wrote ratings for {len(ratings)} teams, as of {snapshot['_meta']['as_of']}, "
      f"to team_ratings.json")
print("Copy this file into the mathayo-backend repo root, next to main.py, "
      "and re-run this script + re-commit every week or so to keep it current.")
