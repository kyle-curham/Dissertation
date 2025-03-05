"""
EEG Filtering Utilities

This module implements preprocessing filters using MNE-python.
"""

import mne
import numpy as np

def bandpass_filter(raw_data, sfreq, l_freq, h_freq, method='iir'):
    """
    Apply bandpass filter to EEG data using MNE.
    
    Parameters:
    raw_data (np.ndarray): EEG data array (channels x samples)
    sfreq (float): Sampling frequency
    l_freq (float): Low cutoff frequency
    h_freq (float): High cutoff frequency
    method (str): Filtering method ('iir' or 'fir')
    
    Returns:
    np.ndarray: Filtered EEG data
    """
    # Create raw object from numpy array
    raw = mne.io.RawArray(raw_data, mne.create_info(raw_data.shape[0], sfreq, 'eeg'))
    
    # Apply bandpass filter
    filtered_raw = raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method=method,
        fir_design='firwin'
    )
    
    return filtered_raw.get_data()

def notch_filter(raw_data, sfreq, freq, notch_widths=3):
    """
    Apply notch filter to remove line noise.
    
    Parameters:
    raw_data (np.ndarray): EEG data array (channels x samples)
    sfreq (float): Sampling frequency
    freq (float): Frequency to notch filter (e.g. 50Hz or 60Hz)
    notch_widths (float): Width of the notch filter
    
    Returns:
    np.ndarray: Filtered EEG data
    """
    raw = mne.io.RawArray(raw_data, mne.create_info(raw_data.shape[0], sfreq, 'eeg'))
    
    # Apply notch filter
    filtered_raw = raw.notch_filter(
        freqs=freq,
        notch_widths=notch_widths,
        method='iir'
    )
    
    return filtered_raw.get_data() 