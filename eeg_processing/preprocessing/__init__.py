"""
EEG Preprocessing Module

This module provides functions for preprocessing EEG data, including filtering,
artifact removal, epoching, and feature extraction.
"""

from .filtering import bandpass_filter, notch_filter

__all__ = [
    'bandpass_filter',
    'notch_filter',
] 