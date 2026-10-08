"""Load a league (teams + fixtures) and simulate the rest of the season many times.

All simulations run *in parallel* with numpy: every fixture is sampled for all
N seasons at once, so 10,000 seasons of a 20-team league take about a second.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .model import elo_update, sample_scores


@dataclass
class Fixture:
    home: str
    away: str
    home_goals: int | None = None
    away_goals: int | None = None

    @property
    def played(self) -> bool:
        return self.home_goals is not None and self.away_goals is not None


@dataclass
class League:
    teams: dict[str, float]                      # name -> Elo rating
    fixtures: list[Fixture] = field(default_factory=list)

    @classmethod
    def from_csv(cls, teams_csv: str | Path, fixtures_csv: str | Path | None = None) -> "League":
        with open(teams_csv, newline="", encoding="utf-8") as fh:
            teams = {row["team"].strip(): float(row["elo"]) for row in csv.DictReader(fh)}
        if len(teams) < 2:
            raise ValueError("need at least two teams")
        if fixtures_csv is None:
            return cls(teams, double_round_robin(list(teams)))
        fixtures = []
        with open(fixtures_csv, newline="", encoding="utf-8") as fh:
            for i, row in enumerate(csv.DictReader(fh), start=2):
                home, away = row["home"].strip(), row["away"].strip()
                for t in (home, away):
                    if t not in teams:
                        raise ValueError(f"line {i}: unknown team {t!r}")
                hg, ag = (row.get("home_goals") or "").strip(), (row.get("away_goals") or "").strip()
                fixtures.append(Fixture(home, away, int(hg) if hg else None, int(ag) if ag else None))
        return cls(teams, fixtures)


def double_round_robin(teams: list[str]) -> list[Fixture]:
    """Every team plays every other team home and away (circle method)."""
    names = list(teams)
    if len(names) % 2:
        names.append(None)  # bye
    n = len(names)
    rounds = []
    for r in range(n - 1):
        pairs = []
        for i in range(n // 2):
            a, b = names[i], names[n - 1 - i]
            if a is not None and b is not None:
                pairs.append((a, b) if (r + i) % 2 == 0 else (b, a))
        rounds.append(pairs)
        names = [names[0]] + [names[-1]] + names[1:-1]
    first = [Fixture(h, a) for rnd in rounds for h, a in rnd]
    second = [Fixture(a, h) for rnd in rounds for h, a in rnd]
    return first + second


@dataclass
class SimulationResult:
    teams: list[str]
    points: np.ndarray        # (sims, teams)
    goal_diff: np.ndarray     # (sims, teams)
    positions: np.ndarray     # (sims, teams), 1 = champion

    @property
    def n_sims(self) -> int:
        return self.points.shape[0]

    def position_probabilities(self) -> np.ndarray:
        """(teams, positions) matrix of probabilities."""
        n = len(self.teams)
        out = np.zeros((n, n))
        for pos in range(1, n + 1):
            out[:, pos - 1] = (self.positions == pos).mean(axis=0)
        return out

    def summary(self, top: int = 4, relegated: int = 3) -> list[dict]:
        n = len(self.teams)
        rows = []
        for i, name in enumerate(self.teams):
            pos = self.positions[:, i]
            rows.append({
                "team": name,
                "avg_points": float(self.points[:, i].mean()),
                "avg_position": float(pos.mean()),
                "title": float((pos == 1).mean()),
                "top": float((pos <= top).mean()),
                "relegation": float((pos > n - relegated).mean()),
            })
        return sorted(rows, key=lambda r: (r["avg_position"], -r["avg_points"]))


def simulate(league: League, n_sims: int = 10_000, seed: int | None = None, dynamic: bool = True) -> SimulationResult:
    """Simulate the unplayed fixtures `n_sims` times.

    dynamic=True updates each team's Elo after every simulated match, so a team
    on a hot streak in one simulated season keeps that form — this widens the
    spread of outcomes in a realistic way.
    """
    rng = np.random.default_rng(seed)
    names = list(league.teams)
    idx = {t: i for i, t in enumerate(names)}
    T = len(names)

    points = np.zeros((n_sims, T), dtype=np.int32)
    gf = np.zeros((n_sims, T), dtype=np.int32)
    ga = np.zeros((n_sims, T), dtype=np.int32)
    elo = np.tile(np.array([league.teams[t] for t in names], dtype=float), (n_sims, 1))

    def record(h, a, hg, ag):
        gf[:, h] += hg; ga[:, h] += ag
        gf[:, a] += ag; ga[:, a] += hg
        points[:, h] += np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
        points[:, a] += np.where(ag > hg, 3, np.where(hg == ag, 1, 0))

    for fx in league.fixtures:
        h, a = idx[fx.home], idx[fx.away]
        if fx.played:
            hg = np.full(n_sims, fx.home_goals); ag = np.full(n_sims, fx.away_goals)
        else:
            hg, ag = sample_scores(elo[:, h], elo[:, a], rng)
        record(h, a, hg, ag)
        if dynamic:
            delta = elo_update(elo[:, h], elo[:, a], hg, ag)
            elo[:, h] += delta
            elo[:, a] -= delta

    gd = gf - ga
    # rank by points, then goal difference, then goals scored, then a random draw of lots
    key = points * 1_000_000 + (gd + 500) * 1_000 + gf + rng.random((n_sims, T))
    order = np.argsort(-key, axis=1)
    positions = np.empty_like(order)
    rows = np.arange(n_sims)[:, None]
    positions[rows, order] = np.arange(1, T + 1)
    return SimulationResult(names, points, gd, positions)


def current_table(league: League) -> list[dict]:
    """The real table from fixtures that have already been played."""
    table = {t: {"team": t, "p": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "pts": 0} for t in league.teams}
    for fx in league.fixtures:
        if not fx.played:
            continue
        h, a = table[fx.home], table[fx.away]
        for row, f, ag in ((h, fx.home_goals, fx.away_goals), (a, fx.away_goals, fx.home_goals)):
            row["p"] += 1; row["gf"] += f; row["ga"] += ag
            if f > ag: row["w"] += 1; row["pts"] += 3
            elif f == ag: row["d"] += 1; row["pts"] += 1
            else: row["l"] += 1
    return sorted(table.values(), key=lambda r: (-r["pts"], -(r["gf"] - r["ga"]), -r["gf"], r["team"]))
