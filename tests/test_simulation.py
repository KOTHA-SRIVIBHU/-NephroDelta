"""
Unit tests for stochastic Monte Carlo simulation and transition model.
"""

import pytest
import numpy as np

from src.simulation import (
    validate_transition_matrix,
    run_monte_carlo_simulation,
    DEFAULT_TRANSITION_MATRIX
)


def test_transition_matrix_validation():
    # Valid default matrix
    assert validate_transition_matrix(DEFAULT_TRANSITION_MATRIX) is True

    # Invalid row sum
    invalid_matrix = np.array([
        [0.80, 0.08, 0.02, 0.00],  # Sums to 0.90
        [0.03, 0.85, 0.10, 0.02],
        [0.00, 0.05, 0.82, 0.13],
        [0.00, 0.00, 0.05, 0.95]
    ])
    with pytest.raises(ValueError, match="Transition matrix rows must sum to 1.0"):
        validate_transition_matrix(invalid_matrix)

    # Invalid shape
    with pytest.raises(ValueError, match="Transition matrix must be square 2D array"):
        validate_transition_matrix(np.array([0.5, 0.5]))


def test_monte_carlo_simulation_output():
    res = run_monte_carlo_simulation(
        initial_state=0,
        initial_egfr=90.0,
        num_simulations=500,
        time_horizon=10,
        seed=42
    )

    assert len(res["time_steps"]) == 11  # t=0 to t=10
    assert len(res["median"]) == 11
    assert len(res["p10"]) == 11
    assert len(res["p90"]) == 11
    assert res["egfr_trajectories"].shape == (500, 11)
    assert res["initial_egfr"] == 90.0

    # Percentile ordering assertion: p10 <= median <= p90
    for t in range(11):
        assert res["p10"][t] <= res["median"][t] <= res["p90"][t]


def test_monte_carlo_simulation_invalid_inputs():
    with pytest.raises(ValueError, match="num_simulations must be greater than 0"):
        run_monte_carlo_simulation(initial_state=0, initial_egfr=90.0, num_simulations=0)

    with pytest.raises(ValueError, match="time_horizon must be greater than 0"):
        run_monte_carlo_simulation(initial_state=0, initial_egfr=90.0, time_horizon=-5)
