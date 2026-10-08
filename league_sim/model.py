"""Match model: turn two Elo ratings into expected goals, then sample scorelines.

Goals for each side follow a Poisson distribution. The mean (expected goals)
rises exponentially with the rating gap, and the home side gets a fixed Elo
bonus. It's a simple model, but it reproduces realistic scorelines
(lots of 1-0, 1-1, 2-1) and realistic home win / draw / away win rates.
"""
from __future__ import annotations

import numpy as np

# Tuned so two equal teams give ~45% home win / 26% draw / 29% away win,
# close to long-run averages in top European leagues.
HOME_ADVANTAGE = 40.0      # Elo points
BASE_HOME_GOALS = 1.35
BASE_AWAY_GOALS = 1.20
BETA = 0.0018              # how strongly the rating gap moves expected goals


def expected_goals(home_elo, away_elo):
    """Expected goals (home, away). Works with floats or numpy arrays."""
    d = np.asarray(home_elo, dtype=float) + HOME_ADVANTAGE - np.asarray(away_elo, dtype=float)
    return BASE_HOME_GOALS * np.exp(BETA * d), BASE_AWAY_GOALS * np.exp(-BETA * d)


def sample_scores(home_elo, away_elo, rng: np.random.Generator):
    """Sample one scoreline per element of the rating arrays."""
    lam_h, lam_a = expected_goals(home_elo, away_elo)
    return rng.poisson(lam_h), rng.poisson(lam_a)


def outcome_probabilities(home_elo: float, away_elo: float, max_goals: int = 10):
    """Exact (home win, draw, away win) probabilities under the Poisson model."""
    from math import exp, factorial

    lam_h, lam_a = (float(x) for x in expected_goals(home_elo, away_elo))
    ph = [exp(-lam_h) * lam_h**k / factorial(k) for k in range(max_goals + 1)]
    pa = [exp(-lam_a) * lam_a**k / factorial(k) for k in range(max_goals + 1)]
    win = sum(ph[i] * pa[j] for i in range(len(ph)) for j in range(len(pa)) if i > j)
    draw = sum(ph[i] * pa[i] for i in range(len(ph)))
    loss = sum(ph[i] * pa[j] for i in range(len(ph)) for j in range(len(pa)) if i < j)
    total = win + draw + loss
    return win / total, draw / total, loss / total


def elo_update(home_elo, away_elo, home_goals, away_goals, k: float = 20.0):
    """Standard football Elo update with a goal-difference multiplier.

    Returns the rating change for the home side (the away side gets the negative).
    """
    home_elo = np.asarray(home_elo, dtype=float)
    away_elo = np.asarray(away_elo, dtype=float)
    gd = np.asarray(home_goals) - np.asarray(away_goals)
    result = np.where(gd > 0, 1.0, np.where(gd == 0, 0.5, 0.0))
    expected = 1.0 / (1.0 + 10 ** (-(home_elo + HOME_ADVANTAGE - away_elo) / 400.0))
    margin = np.where(np.abs(gd) <= 1, 1.0, np.where(np.abs(gd) == 2, 1.5, (11 + np.abs(gd)) / 8))
    return k * margin * (result - expected)
