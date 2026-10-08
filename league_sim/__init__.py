"""league-sim: Monte Carlo football season simulator."""
from .model import expected_goals, outcome_probabilities, sample_scores, elo_update
from .season import Fixture, League, SimulationResult, current_table, double_round_robin, simulate

__all__ = [
    "expected_goals", "outcome_probabilities", "sample_scores", "elo_update",
    "Fixture", "League", "SimulationResult", "current_table", "double_round_robin", "simulate",
]
__version__ = "1.0.0"
