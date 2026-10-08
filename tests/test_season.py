import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from league_sim import Fixture, League, current_table, double_round_robin, outcome_probabilities, simulate
from league_sim.cli import main

DATA = Path(__file__).resolve().parent.parent / "data"


class RoundRobinTests(unittest.TestCase):
    def test_every_pair_meets_home_and_away(self):
        teams = [f"T{i}" for i in range(6)]
        fx = double_round_robin(teams)
        self.assertEqual(len(fx), 6 * 5)
        self.assertEqual(len({(f.home, f.away) for f in fx}), 30)
        self.assertTrue(all(f.home != f.away for f in fx))

    def test_odd_number_of_teams(self):
        fx = double_round_robin(["A", "B", "C"])
        self.assertEqual(Counter((f.home, f.away) for f in fx), Counter({p: 1 for p in
                         [("A", "B"), ("B", "A"), ("A", "C"), ("C", "A"), ("B", "C"), ("C", "B")]}))


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.league = League({"Strong": 1800, "Mid": 1600, "Weak": 1400}, double_round_robin(["Strong", "Mid", "Weak"]))

    def test_positions_are_a_permutation_each_season(self):
        r = simulate(self.league, n_sims=500, seed=1)
        for row in r.positions:
            self.assertEqual(sorted(row.tolist()), [1, 2, 3])

    def test_position_probabilities_rows_and_columns_sum_to_one(self):
        m = simulate(self.league, n_sims=2000, seed=2).position_probabilities()
        np.testing.assert_allclose(m.sum(axis=0), 1.0)
        np.testing.assert_allclose(m.sum(axis=1), 1.0)

    def test_strong_team_usually_wins_the_league(self):
        rows = simulate(self.league, n_sims=4000, seed=3).summary(top=1, relegated=1)
        self.assertEqual(rows[0]["team"], "Strong")
        self.assertGreater(rows[0]["title"], 0.5)

    def test_same_seed_same_result(self):
        a = simulate(self.league, n_sims=300, seed=42)
        b = simulate(self.league, n_sims=300, seed=42)
        np.testing.assert_array_equal(a.points, b.points)

    def test_static_simulation_matches_exact_expected_points(self):
        r = simulate(self.league, n_sims=40_000, seed=4, dynamic=False)
        expected = {t: 0.0 for t in self.league.teams}
        for f in self.league.fixtures:
            w, d, l = outcome_probabilities(self.league.teams[f.home], self.league.teams[f.away])
            expected[f.home] += 3 * w + d
            expected[f.away] += 3 * l + d
        for i, t in enumerate(r.teams):
            self.assertAlmostEqual(r.points[:, i].mean(), expected[t], delta=0.05)

    def test_played_results_are_respected(self):
        fx = [Fixture("Strong", "Weak", 0, 3)] + [Fixture("Mid", "Weak", 0, 4)] + [Fixture("Strong", "Mid", 0, 2)]
        league = League({"Strong": 1800, "Mid": 1600, "Weak": 1400}, fx)
        r = simulate(league, n_sims=50, seed=5)
        self.assertTrue((r.positions[:, r.teams.index("Weak")] == 1).all())
        self.assertEqual(current_table(league)[0]["team"], "Weak")


class CsvAndCliTests(unittest.TestCase):
    def test_sample_data_loads(self):
        league = League.from_csv(DATA / "teams.csv", DATA / "fixtures.csv")
        self.assertEqual(len(league.teams), 20)
        self.assertEqual(len(league.fixtures), 380)

    def test_unknown_team_in_fixtures_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            teams, fixtures = Path(d) / "t.csv", Path(d) / "f.csv"
            teams.write_text("team,elo\nA,1500\nB,1500\n")
            fixtures.write_text("home,away,home_goals,away_goals\nA,Z,,\n")
            with self.assertRaises(ValueError):
                League.from_csv(teams, fixtures)

    def test_cli_runs_and_writes_json(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out.json"
            code = main(["--teams", str(DATA / "teams.csv"), "--fixtures", str(DATA / "fixtures.csv"),
                         "-n", "200", "--seed", "1", "--no-color", "--json", str(out)])
            self.assertEqual(code, 0)
            self.assertIn("teams", out.read_text())

    def test_cli_odds(self):
        self.assertEqual(main(["--teams", str(DATA / "teams.csv"), "--odds", "Code FC", "Bugs FC", "--no-color"]), 0)
        self.assertEqual(main(["--teams", str(DATA / "teams.csv"), "--odds", "Code FC", "Nope FC"]), 2)


if __name__ == "__main__":
    unittest.main()
