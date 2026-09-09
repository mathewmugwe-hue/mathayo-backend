"""
Paste-odds parser.

WHAT THIS IS FOR:
  You copy-paste the visible text of an upcoming-fixtures/jackpot page
  (Betika, Mozzart, or any other site's raw odds dump - the formats are
  all different and inconsistent, which is exactly why this is heuristic,
  not a strict format reader) and this pulls out
  (date, league, home_team, away_team, odd_home, odd_draw, odd_away) rows.

WHAT THIS IS NOT:
  It is not a scraper. It does not visit any bookmaker site itself. You
  still do the copying. That keeps you clear of any site's anti-scraping
  ToS, and it's more robust anyway - if you paste it, I don't have to keep
  up with every site's layout changing.

  Screenshots specifically: OCR-ing a screenshot of an odds table is much
  less reliable than this (font/column misalignment, decimal points
  dropped, etc.). If you only have a screenshot, select-all/copy the
  underlying page text instead if you can - that's what this expects.

HOW IT WORKS (heuristic, not a strict grammar - inspect the output):
  1. Scan line by line for two consecutive "team-name-shaped" lines
     (letters/spaces/periods only, not a label like "Draw" or "NOT
     STARTED", not itself a number).
  2. Once found, scan forward collecting the first three decimal-odds
     tokens (##.## format), ignoring any label lines in between (Draw,
     1/X/2, Home/Away, market names, "+84 Markets", etc.).
  3. Those three odds are assumed to be [home, draw, away] in that order -
     true for every format you showed me, but ALWAYS spot check a few
     rows against the source before trusting this at scale.
  4. The nearest preceding date/time line and league-name line are
     attached as context.

This handles the Betika jackpot layout, the Mozzart daily-jackpot layout,
and the "3 Way" / compact-table odds-portal-style layout you pasted, all
in one pass - see the worked example at the bottom.
"""
import re
import pandas as pd

ODDS_RE = re.compile(r"^\d{1,3}\.\d{2}$")
ODDS_FINDALL_RE = re.compile(r"\d{1,3}\.\d{2}")
DATE_RE = re.compile(
    r"(\d{1,2}[./]\d{1,2}(?:[./]\d{2,4})?)\s*[-,|]?\s*(\d{1,2}:\d{2})?"
)

EXCLUDE_EXACT = {
    "draw", "home", "away", "1", "x", "2", "1 or x", "x or 2", "1 or 2",
    "not started", "teams", "apply pass", "auto pick", "clear all",
    "double chance", "over/under", "both teams to score", "odd/even",
    "3 way", "draw no bet", "draw no bet - full time",
    "half with most goals", "first team to score", "3 way - first half",
    "both teams to score - first half", "hot", "🔥 hot",
}

LEAGUE_TO_DIVISION = {
    "premier league": "E0", "epl": "E0",
    "la liga": "SP1", "laliga": "SP1",
    "bundesliga": "D1",
    "serie a": "I1",
    "ligue 1": "F1",
}


def _is_team_line(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 40:
        return False
    if s.lower() in EXCLUDE_EXACT:
        return False
    if ODDS_RE.match(s):
        return False
    if s.startswith(("#", "+", "ID:", "|")):
        return False
    if re.fullmatch(r"[+\-]?\d+", s):  # bare integers like "25", "#1625"
        return False
    if not re.fullmatch(r"[A-Za-z0-9ÀÁÂÃÄÅàáâãäåÈÉÊËèéêë .'&\-]+", s):
        return False
    if not re.search(r"[A-Za-z]", s):
        return False
    return True


def _guess_division(league_line: str):
    if not league_line:
        return None
    key = league_line.lower()
    for name, code in LEAGUE_TO_DIVISION.items():
        if name in key:
            return code
    return None


def parse_slip_text(text: str) -> pd.DataFrame:
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln != ""]  # drop blank lines, keep order

    rows = []
    last_date, last_time, last_league = None, None, None
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # Track most recent date/time context
        m = DATE_RE.search(line)
        if m and any(ch.isdigit() for ch in line) and ("/" in line or "." in line):
            last_date = m.group(1)
            if m.group(2):
                last_time = m.group(2)
            # a league name often sits on the same or previous line
            prefix = line[: m.start()].strip(" |")
            if prefix and not _is_team_line(prefix):
                last_league = prefix
            elif i > 0 and not _is_team_line(lines[i - 1]) and lines[i - 1].lower() not in EXCLUDE_EXACT:
                candidate = lines[i - 1]
                if re.search(r"[A-Za-z]", candidate) and not ODDS_RE.match(candidate):
                    last_league = candidate
            i += 1
            continue

        # Look for a team-pair (two consecutive team-shaped lines)
        if _is_team_line(line) and i + 1 < n and _is_team_line(lines[i + 1]):
            home, away = line, lines[i + 1]
            j = i + 2
            collected = []
            steps = 0
            while j < n and len(collected) < 3 and steps < 40:
                cand = lines[j]
                if DATE_RE.search(cand) and ("/" in cand or "." in cand) and re.search(r"\d{1,2}:\d{2}", cand):
                    break  # hit the next fixture's date line - stop early
                # handle both "each odd on its own line" and
                # "1  2.34   X  4.00   2  2.75" all on one line
                for tok in ODDS_FINDALL_RE.findall(cand):
                    if len(collected) < 3:
                        collected.append(float(tok))
                j += 1
                steps += 1

            if len(collected) == 3:
                rows.append({
                    "date": last_date,
                    "time": last_time,
                    "league": last_league,
                    "Division": _guess_division(last_league or ""),
                    "HomeTeam": home,
                    "AwayTeam": away,
                    "OddHome": collected[0],
                    "OddDraw": collected[1],
                    "OddAway": collected[2],
                })
                i = j  # skip past the consumed block
                continue

        i += 1

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Worked example using your pasted text (mixed Betika/odds-portal formats)
    sample = """
09/09/26 - 19:45 | ID: 3336
 Barcelona
 Feyenoord

25
3 Way   " >    
Barcelona
1.08
Draw
13.00
Feyenoord
25.00
Double Chance   " >    
1 or X
1.02
X or 2
8.20
1 or 2
1.04

09/09/26 - 22:00 | ID: 5793
 Napoli
 Arsenal

25
3 Way   " >    
Napoli
5.80
Draw
4.10
Arsenal
1.60

International Clubs • UEFA Champions League
09/09, 19:45
Stuttgart
Viking Fk

1.23

8.00

11.00
+84 Markets

International Clubs • UEFA Champions League
09/09, 19:45
Barcelona
Feyenoord

1.08

15.00

28.00
+81 Markets

International Clubs / UEFA Champions League (6)
🔥 HOT
AEK Athens
LASK Linz
NOT STARTED
#1625
+149 more

1
1.69

X
4.10

2
4.30
🔥 HOT
Club Brugge
Aston Villa
NOT STARTED
#2305
+149 more

1
2.50

X
3.45

2
2.60

Soccer • Bundesliga
12/09, 16:30
Hoffenheim
Stuttgart
1  2.34   X  4.00   2  2.75
"""
    out = parse_slip_text(sample)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    print(out)
