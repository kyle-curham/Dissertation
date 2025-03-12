"""
Utility functions for EEG processing and visualization.
This package contains helper functions for data processing, visualization, and analysis.
"""

from eeg_processing.utils.plotting import (
    plot_training_results,
    plot_matrix_analysis,
    plot_training_history,
    plot_prediction_comparison,
    plot_latent_dynamics
)

__all__ = [
    'plot_training_results',
    'plot_matrix_analysis',
    'plot_training_history',
    'plot_prediction_comparison',
    'plot_latent_dynamics'
] 