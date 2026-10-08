import unittest

import numpy as np

from league_sim.model import elo_update, expected_goals, outcome_probabilities, sample_scores


class ModelTests(unittest.TestCase):
    def test_probabilities_sum_to_one(self):
        for h, a in [(1500, 1500), (1800, 1400), (1400, 1800)]:
            self.assertAlmostEqual(sum(outcome_probabilities(h, a)), 1.0, places=9)

    def test_equal_teams_look_like_real_football(self):
        w, d, l = outcome_probabilities(1550, 1550)
        self.assertTrue(0.42 < w < 0.48, w)
        self.assertTrue(0.23 < d < 0.29, d)
        self.assertTrue(0.26 < l < 0.32, l)

    def test_stronger_team_scores_more(self):
        strong, weak = expected_goals(1800, 1400)
        self.assertGreater(strong, weak)
        self.assertGreater(outcome_probabilities(1800, 1400)[0], outcome_probabilities(1600, 1600)[0])

    def test_sampled_goals_match_expected_goals(self):
        rng = np.random.default_rng(0)
        hg, ag = sample_scores(np.full(200_000, 1600.0), np.full(200_000, 1500.0), rng)
        lam_h, lam_a = expected_goals(1600, 1500)
        self.assertAlmostEqual(hg.mean(), float(lam_h), delta=0.02)
        self.assertAlmostEqual(ag.mean(), float(lam_a), delta=0.02)

    def test_elo_update_is_zero_sum_and_directional(self):
        win = float(elo_update(1500, 1500, 2, 0))
        loss = float(elo_update(1500, 1500, 0, 1))
        self.assertGreater(win, 0)
        self.assertLess(loss, 0)
        # bigger wins move ratings more
        self.assertGreater(float(elo_update(1500, 1500, 5, 0)), float(elo_update(1500, 1500, 1, 0)))


if __name__ == "__main__":
    unittest.main()
