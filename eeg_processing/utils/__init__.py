"""
EEG Processing Utilities

This module contains numerical tools for state-space modeling and analysis.
"""

# Numerical differentiation functions
from .numerical_differentiation import (
    central_difference,
    savitzky_golay_derivative,
    discrete_to_continuous,
    continuous_to_discrete,
    estimate_derivatives
)

# Numerical stability functions
from .numerical_stability import (
    enforce_cholesky_structure,
    ensure_positive_definite,
    stabilize_covariance,
    safe_cholesky
)

__all__ = [
    # Differentiation
    'central_difference',
    'savitzky_golay_derivative',
    'discrete_to_continuous',
    'continuous_to_discrete',
    'estimate_derivatives',
    
    # Stability
    'enforce_cholesky_structure',
    'ensure_positive_definite', 
    'stabilize_covariance',
    'safe_cholesky'
] 