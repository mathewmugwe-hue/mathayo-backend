"""
Understat xG fetcher - RUN THIS ON YOUR OWN MACHINE OR RENDER/VERCEL SERVER,
NOT in this sandbox. understat.com is not reachable from this environment's
network allowlist, so I could not fetch or backtest this data myself - I'm
giving you the pipeline honestly rather than fabricating results for data
I never actually touched.

Install first:
    pip install understat aiohttp pandas

This pulls team-level and match-level xG for the top-5 leagues + RFPL,
2014/15 onward, and produces a CSV of rolling pre-match xG form you can
merge into the Matches.csv feature set by (Division, MatchDate, HomeTeam,
AwayTeam).

IMPORTANT - to keep this leakage-free:
  Only use each team's xG from matches STRICTLY BEFORE the match you're
  predicting (a rolling average over their last N matches). Do not use the
  xG of the match itself as a feature for predicting that match's outcome
  - that's the exact leakage trap that produces the fake 80-90% accuracy
  claims we already discussed.
"""
import asyncio
import aiohttp
import pandas as pd
from understat import Understat

LEAGUES = ["epl", "la_liga", "bundesliga", "serie_a", "ligue_1"]
SEASONS = list(range(2014, 2026))  # 2014/15 through 2025/26


async def fetch_league_season(understat, league, season):
    try:
        results = await understat.get_league_results(league, season)
        rows = []
        for m in results:
            rows.append({
                "league": league,
                "season": season,
                "date": m["datetime"],
                "home_team": m["h"]["title"],
                "away_team": m["a"]["title"],
                "home_xg": float(m["xG"]["h"]),
                "away_xg": float(m["xG"]["a"]),
                "home_goals": int(m["goals"]["h"]),
                "away_goals": int(m["goals"]["a"]),
            })
        return rows
    except Exception as e:
        print(f"  skipped {league} {season}: {e}")
        return []


async def main():
    all_rows = []
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        for league in LEAGUES:
            for season in SEASONS:
                print(f"Fetching {league} {season}...")
                rows = await fetch_league_season(understat, league, season)
                all_rows.extend(rows)
                await asyncio.sleep(1)  # be polite to the source

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv("understat_xg_raw.csv", index=False)
    print(f"\nSaved {len(df)} matches with xG to understat_xg_raw.csv")

    # --- Build LEAKAGE-FREE rolling pre-match xG form (last 5 matches) ---
    df["team_home_row"] = df.index
    rolling_rows = []
    team_history = {}  # team -> list of (date, xg_for, xg_against)

    for row in df.itertuples():
        h, a = row.home_team, row.away_team
        h_hist = team_history.get(h, [])
        a_hist = team_history.get(a, [])

        h_xg_form = sum(x[1] for x in h_hist[-5:]) / len(h_hist[-5:]) if h_hist else None
        h_xga_form = sum(x[2] for x in h_hist[-5:]) / len(h_hist[-5:]) if h_hist else None
        a_xg_form = sum(x[1] for x in a_hist[-5:]) / len(a_hist[-5:]) if a_hist else None
        a_xga_form = sum(x[2] for x in a_hist[-5:]) / len(a_hist[-5:]) if a_hist else None

        rolling_rows.append({
            "date": row.date, "league": row.league,
            "home_team": h, "away_team": a,
            "home_xg_form5": h_xg_form, "home_xga_form5": h_xga_form,
            "away_xg_form5": a_xg_form, "away_xga_form5": a_xga_form,
        })

        team_history.setdefault(h, []).append((row.date, row.home_xg, row.away_xg))
        team_history.setdefault(a, []).append((row.date, row.away_xg, row.home_xg))

    rolling_df = pd.DataFrame(rolling_rows)
    rolling_df.to_csv("understat_xg_rolling_features.csv", index=False)
    print(f"Saved leakage-free rolling xG features to understat_xg_rolling_features.csv")
    print("\nNext step: merge this on (date, home_team, away_team) with your")
    print("Matches.csv pipeline (main backtest script), add home_xg_form5 /")
    print("home_xga_form5 / away_xg_form5 / away_xga_form5 to feature_cols,")
    print("and re-run the SAME walk-forward backtest methodology - not a new")
    print("one - so the result is comparable to the 53.6-53.7% baseline.")


if __name__ == "__main__":
    asyncio.run(main())
