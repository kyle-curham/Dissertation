"""
State Space Models for EEG Data

This package provides state-space models for analyzing EEG data, 
including linear and nonlinear models, variational inference methods,
and control-theoretic approaches.
"""

from .coupled_state_space_vi import CoupledStateSpaceVI

__all__ = [
    'CoupledStateSpaceVI',
] 