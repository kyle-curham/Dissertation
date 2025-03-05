"""
EEG Visualization Module

This module provides functions for visualizing state-space models and observations.
"""

# Primary visualization functions for state-space analysis
from .plots import (
    plot_states_and_observations,
    plot_state_couplings,
    plot_state_evolution
)

__all__ = [
    'plot_states_and_observations',
    'plot_state_couplings',
    'plot_state_evolution'
] 