"""
Control theory utilities for state-space models.

This module provides functions for solving control-theoretic problems
such as Riccati equations and optimal control.
"""

# Import control classes and functions to make them available at the module level
from .riccati_solver import RiccatiSolver

from .matrix_initialization import (
    create_controllable_B,
    initialize_riccati_p
) 