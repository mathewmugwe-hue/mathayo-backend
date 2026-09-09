"""
Pi-ratings (simplified implementation following Constantinou & Fenton, 2013,
"Determining the level of ability of football teams by dynamic ratings
based on the relative discrepancies in scores between adversaries").

Unlike Elo (which only tracks a single overall strength number updated on
win/draw/loss), pi-ratings track separate HOME and AWAY ratings per team,
updated by the SIZE of the goal-difference surprise rather than just
win/loss - and each match updates both a team's primary rating strongly and
their other-venue rating weakly (cross-learning), since home form partially
predicts away ability and vice versa.

This is computed walk-forward, strictly using only goals from matches that
happened BEFORE the one being rated - there is no leakage, and no external
data is required. This can be computed entirely from the same match-results
table you already have.

Reference values (lr_home=0.06, cross_ratio=0.35) follow the ranges
reported in the original paper and in Hubacek et al.'s use of pi-ratings
for the 2017 Soccer Prediction Challenge. Treat these as a starting point,
not gospel - the backtest below is what actually validates them, not the
citation.
"""
import math


def g(x: float) -> float:
    """Maps a goal-difference error onto rating scale (compressive, like log-odds)."""
    if x >= 0:
        return 3 * math.log10(1 + x)
    return -3 * math.log10(1 - x)


def g_inv(y: float) -> float:
    """Inverse of g(): maps a rating value back onto expected goal-difference scale."""
    if y >= 0:
        return 10 ** (y / 3) - 1
    return 1 - 10 ** (-y / 3)


class PiRatings:
    def __init__(self, lr_home: float = 0.06, cross_ratio: float = 0.35):
        self.lr_home = lr_home          # learning rate for the primary (surprised) rating
        self.lr_cross = lr_home * cross_ratio  # smaller cross-update to the OTHER venue rating
        self.home_rating = {}  # team -> float, "how strong at home"
        self.away_rating = {}  # team -> float, "how strong away"

    def _get(self, d, team):
        return d.get(team, 0.0)

    def predict_goal_diff(self, home_team: str, away_team: str) -> float:
        """Expected (home_goals - away_goals) BEFORE this match, using only past data."""
        return g_inv(self._get(self.home_rating, home_team)) - g_inv(self._get(self.away_rating, away_team))

    def get_ratings(self, team: str) -> tuple[float, float]:
        return self._get(self.home_rating, team), self._get(self.away_rating, team)

    def update(self, home_team: str, away_team: str, home_goals: int, away_goals: int):
        """Call AFTER predict_goal_diff / get_ratings have been used for this match."""
        actual_gd = home_goals - away_goals
        expected_gd = self.predict_goal_diff(home_team, away_team)
        error = actual_gd - expected_gd
        adj = g(error)

        rh = self._get(self.home_rating, home_team)
        ra_home = self._get(self.away_rating, home_team)
        ra_away = self._get(self.away_rating, away_team)
        rh_away = self._get(self.home_rating, away_team)

        # Primary updates: home team's home rating rises/falls with the surprise;
        # away team's away rating moves the opposite way (same match, same error).
        self.home_rating[home_team] = rh + self.lr_home * adj
        self.away_rating[away_team] = ra_away - self.lr_home * adj

        # Cross updates: a team's OTHER-venue rating moves a little too, since
        # a team that outperforms at home is somewhat more likely to also be
        # underrated away, and vice versa.
        self.away_rating[home_team] = ra_home + self.lr_cross * adj
        self.home_rating[away_team] = rh_away - self.lr_cross * adj
