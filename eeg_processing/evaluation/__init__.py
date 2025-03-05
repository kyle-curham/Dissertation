"""
Evaluation module for analyzing model performance.

This module provides tools for evaluating, visualizing, and analyzing
state-space models and their outputs.
"""

from .visualization import (
    plot_state_trajectories,
    plot_state_distributions,
    plot_reconstruction,
    plot_system_matrices,
    plot_training_history,
    plot_eigenvalue_analysis,
    visualize_coupled_model
) 