"""
External team-rating fetcher - RUN THIS ON YOUR OWN MACHINE, NOT in this
sandbox. Both clubelo.com and the soccerdata package's other backends are
not reachable from this environment's network allowlist, so - same as the
Understat fetcher - I have not run this myself and am not fabricating any
numbers from it.

Install first:
    pip install soccerdata pandas

WHY THESE TWO SOURCES:
  1. clubelo.com  - free public CSV API, built to be machine-read, updated
     after every match, no ToS ambiguity because that's the site's stated
     purpose. Docs: http://clubelo.com/API
  2. soccerdata (github.com/probberechts/soccerdata) - actively maintained
     wrapper that gives you ClubElo AND football-data.co.uk (your likely
     existing odds/results source) through one consistent pandas interface,
     so when either site's page layout changes, you upgrade one pip package
     instead of hand-patching a scraper.

I deliberately did NOT point this at Betika/Mozzart/any bookmaker site.
Those aren't data providers, scraping them programmatically is a ToS risk,
and there's no stable "current" open-source scraper for them because their
markup changes constantly. Use paste_odds_parser.py (the other file) for
that data instead - it works off text you copy yourself.

LEAKAGE WARNING (same rule as everywhere else in this pipeline):
  Only ever use a team's rating AS OF THE DAY BEFORE the match you're
  predicting. ClubElo's per-date CSV (http://api.clubelo.com/{date}) gives
  you the rating as of that date, so pull it dated on the match date itself
  and treat it as "rating entering this match" - never use a rating dated
  after the match.
"""
import time
import pandas as pd
import soccerdata as sd

LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
]

# Maps soccerdata/ClubElo league names to the Division codes used in
# matches_full.csv (football-data.co.uk convention) so this merges cleanly.
LEAGUE_TO_DIVISION = {
    "ENG-Premier League": "E0",
    "ESP-La Liga": "SP1",
    "GER-Bundesliga": "D1",
    "ITA-Serie A": "I1",
    "FRA-Ligue 1": "F1",
}


def fetch_clubelo_history() -> pd.DataFrame:
    """Full historical Elo time series for every club ClubElo tracks."""
    elo = sd.ClubElo()
    # read_by_date() with no args returns *today's* snapshot; for full
    # history you pull per-team series, which is what we do below.
    all_teams = elo.read_by_date().reset_index()["team"].unique()
    frames = []
    for team in all_teams:
        try:
            hist = elo.read_team_history(team)
            hist["team"] = team
            frames.append(hist)
        except Exception as e:
            print(f"  skipped {team}: {e}")
        time.sleep(0.2)  # be polite to the source
    return pd.concat(frames, ignore_index=True)


def fetch_football_data_results() -> pd.DataFrame:
    """Historical results/odds via football-data.co.uk, same source your
    matches_full.csv likely already comes from - useful as a refresh/
    cross-check rather than a new source."""
    frames = []
    for league in LEAGUES:
        try:
            fd = sd.FootballData(leagues=league, seasons="all")
            games = fd.read_games()
            games["Division"] = LEAGUE_TO_DIVISION[league]
            frames.append(games)
        except Exception as e:
            print(f"  skipped {league}: {e}")
    return pd.concat(frames, ignore_index=True)


def main():
    print("Fetching ClubElo historical ratings...")
    elo_hist = fetch_clubelo_history()
    elo_hist.to_csv("clubelo_history_raw.csv", index=False)
    print(f"Saved {len(elo_hist)} rating rows to clubelo_history_raw.csv")

    print("\nFetching football-data.co.uk results/odds (cross-check)...")
    fd_results = fetch_football_data_results()
    fd_results.to_csv("football_data_raw.csv", index=False)
    print(f"Saved {len(fd_results)} match rows to football_data_raw.csv")

    print(
        "\nNext step: for each row in matches_full.csv, look up each team's "
        "ClubElo 'Elo' value as of (MatchDate - 1 day) from "
        "clubelo_history_raw.csv (columns: team, from, to, Elo - the rating "
        "was valid for the [from, to) window), and add it as an extra "
        "feature (e.g. ClubEloHome / ClubEloAway) alongside your existing "
        "HomeElo/AwayElo. Then re-run the SAME walk-forward backtest, since "
        "the only fair test of 'does this help' is whether it moves the "
        "accuracy/log-loss/RPS numbers on identical folds."
    )


if __name__ == "__main__":
    main()
