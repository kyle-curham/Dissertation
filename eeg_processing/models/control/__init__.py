"""
Control theory utilities for state-space models.

This module provides functions for solving control-theoretic problems
such as Riccati equations and optimal control.
"""

# Import control functions to make them available at the module level
from .riccati_solver import (
    compute_optimal_K,
    implicit_euler_step,
    build_augmented_system,
    solve_care_smith
)

from .matrix_initialization import (
    create_controllable_B,
    initialize_riccati_p
) 