"""
Stochastic Monte Carlo Simulation Module for NephroDelta.

Implements a Markov chain state transition model and continuous Monte Carlo
trajectory simulation for research analytics.

IMPORTANT DISCLAIMER:
These transition matrices and simulation trajectories represent hypothetical,
model-generated research scenarios. They are NOT clinical probabilities or predictions
of an individual's medical future.
"""

from typing import Dict, Any, Tuple
import numpy as np


DEFAULT_TRANSITION_MATRIX = np.array([
    [0.90, 0.08, 0.02, 0.00],
    [0.03, 0.85, 0.10, 0.02],
    [0.00, 0.05, 0.82, 0.13],
    [0.00, 0.00, 0.05, 0.95]
], dtype=float)


def validate_transition_matrix(matrix: np.ndarray) -> bool:
    """
    Validates that a 2D transition matrix is square and that each row sums to 1.0.

    Args:
        matrix: 2D NumPy array representing state transition probabilities.

    Returns:
        True if valid.

    Raises:
        ValueError: If shape is invalid or row sums deviate from 1.0.
    """
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Transition matrix must be square 2D array, got shape {matrix.shape}")

    row_sums = np.sum(matrix, axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError(f"Transition matrix rows must sum to 1.0. Row sums: {row_sums}")

    if np.any(matrix < 0.0) or np.any(matrix > 1.0):
        raise ValueError("Transition probabilities must be between 0.0 and 1.0.")

    return True


def run_monte_carlo_simulation(
    initial_state: int,
    initial_egfr: float,
    num_simulations: int = 1000,
    time_horizon: int = 12,
    seed: int = 42,
    transition_matrix: np.ndarray = DEFAULT_TRANSITION_MATRIX
) -> Dict[str, Any]:
    """
    Runs a Monte Carlo stochastic simulation over a specified time horizon.

    Args:
        initial_state: Analytical state at t=0 (0, 1, 2, or 3).
        initial_egfr: Initial eGFR value at t=0.
        num_simulations: Number of Monte Carlo iterations (must be > 0).
        time_horizon: Number of future time steps to simulate (must be > 0).
        seed: Random seed for reproducibility.
        transition_matrix: 4x4 Markov state transition matrix.

    Returns:
        Dict containing time_steps, median, p10, p90, and full trajectory arrays.
    """
    if num_simulations <= 0:
        raise ValueError("num_simulations must be greater than 0.")
    if time_horizon <= 0:
        raise ValueError("time_horizon must be greater than 0.")

    # Validate transition matrix
    validate_transition_matrix(transition_matrix)

    # Validate state
    initial_state = int(np.clip(initial_state, 0, 3))
    initial_egfr = float(np.clip(initial_egfr, 5.0, 140.0))

    np.random.seed(seed)

    # State-dependent drift and noise parameters for continuous eGFR change
    # Drift per step in mL/min/1.73m2 depending on current state
    state_drifts = {0: -0.2, 1: -0.7, 2: -1.4, 3: -2.2}
    state_stds = {0: 1.2, 1: 1.8, 2: 2.5, 3: 3.2}

    # Initialize trajectories matrix (shape: num_simulations x (time_horizon + 1))
    egfr_trajectories = np.zeros((num_simulations, time_horizon + 1))
    state_trajectories = np.zeros((num_simulations, time_horizon + 1), dtype=int)

    egfr_trajectories[:, 0] = initial_egfr
    state_trajectories[:, 0] = initial_state

    # Cumulative probabilities for efficient choice sampling
    cum_probs = np.cumsum(transition_matrix, axis=1)

    for step in range(1, time_horizon + 1):
        prev_states = state_trajectories[:, step - 1]
        prev_egfr = egfr_trajectories[:, step - 1]

        # Draw random numbers for state transitions
        r_state = np.random.rand(num_simulations)
        next_states = np.zeros(num_simulations, dtype=int)

        for s in range(4):
            mask = (prev_states == s)
            if np.any(mask):
                # Find indices where r_state <= cum_probs[s]
                next_s = np.searchsorted(cum_probs[s], r_state[mask])
                next_states[mask] = np.clip(next_s, 0, 3)

        state_trajectories[:, step] = next_states

        # Continuous eGFR update based on transition state
        drifts = np.array([state_drifts[st] for st in next_states])
        stds = np.array([state_stds[st] for st in next_states])

        noise = np.random.normal(0, 1, size=num_simulations) * stds
        next_egfr = prev_egfr + drifts + noise
        next_egfr = np.clip(next_egfr, 2.0, 140.0)

        egfr_trajectories[:, step] = next_egfr

    # Calculate percentiles across all simulations at each time step
    time_steps = list(range(time_horizon + 1))
    median_trajectory = np.percentile(egfr_trajectories, 50, axis=0).round(2).tolist()
    p10_trajectory = np.percentile(egfr_trajectories, 10, axis=0).round(2).tolist()
    p90_trajectory = np.percentile(egfr_trajectories, 90, axis=0).round(2).tolist()

    return {
        "time_steps": time_steps,
        "median": median_trajectory,
        "p10": p10_trajectory,
        "p90": p90_trajectory,
        "egfr_trajectories": egfr_trajectories,
        "state_trajectories": state_trajectories,
        "initial_state": initial_state,
        "initial_egfr": initial_egfr,
        "num_simulations": num_simulations,
        "time_horizon": time_horizon,
        "disclaimer": "Hypothetical stochastic simulation. This is not a clinical prediction."
    }
