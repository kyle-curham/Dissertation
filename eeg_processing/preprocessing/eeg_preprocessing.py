"""
EEG Preprocessing Functions

This module contains functions for preprocessing EEG data, including
filtering, artifact removal, and other common preprocessing steps.
"""

import numpy as np
import mne
from typing import Tuple, Optional, List, Dict, Any, Union


def preprocess_eeg(raw: mne.io.Raw, 
                   line_freq: float = 50.0,
                   highpass_freq: float = 1.0, 
                   lowpass_freq: float = 40.0,
                   notch_freqs: Optional[List[float]] = None,
                   apply_ica: bool = True,
                   n_components: Optional[int] = None,
                   random_state: int = 42) -> mne.io.Raw:
    """
    Preprocess EEG data following best practices.
    
    Steps:
    1. Set channel types
    2. Filter line noise (50 Hz by default)
    3. High-pass filter to remove slow drifts
    4. Low-pass filter to remove high-frequency noise
    5. Re-reference to average reference
    6. Remove artifacts using ICA (optional)
    
    Args:
        raw: Raw MNE object
        line_freq: Line frequency for notch filter (default: 50.0 Hz)
        highpass_freq: High-pass filter cutoff frequency (default: 1.0 Hz)
        lowpass_freq: Low-pass filter cutoff frequency (default: 40.0 Hz)
        notch_freqs: List of frequencies for notch filter (default: None, uses line_freq harmonics)
        apply_ica: Whether to apply ICA for artifact removal (default: True)
        n_components: Number of ICA components (default: None, automatically determined)
        random_state: Random state for ICA (default: 42)
        
    Returns:
        Preprocessed MNE Raw object
    """
    # Copy the raw object
    raw = raw.copy()
    
    print("\nPreprocessing Steps:")
    
    # Print initial channel info
    print("\nInitial channel types:", raw.get_channel_types())
    
    # 1. Set all channels to EEG type if not already set
    print("1. Setting channel types...")
    if not any(ch_type == 'eeg' for ch_type in raw.get_channel_types()):
        raw.set_channel_types({ch_name: 'eeg' for ch_name in raw.ch_names})
        print("   Set all channels to EEG type")
    
    # 2. Notch filter for line noise (line_freq Hz and harmonics)
    print(f"2. Applying notch filter for line noise ({line_freq} Hz)...")
    if notch_freqs is None:
        notch_freqs = np.arange(line_freq, 201, line_freq)
    raw.notch_filter(notch_freqs, picks='eeg', verbose=False)
    
    # 3. High-pass filter to remove slow drifts
    print(f"3. Applying high-pass filter ({highpass_freq} Hz)...")
    raw.filter(l_freq=highpass_freq, h_freq=None, picks='eeg', verbose=False)
    
    # 4. Low-pass filter to remove high-frequency noise
    print(f"4. Applying low-pass filter ({lowpass_freq} Hz)...")
    raw.filter(l_freq=None, h_freq=lowpass_freq, picks='eeg', verbose=False)
    
    # 5. Re-reference to average reference
    print("5. Re-referencing to average reference...")
    raw.set_eeg_reference('average', projection=True, verbose=False)
    
    # 6. Apply ICA for artifact removal (optional)
    if apply_ica:
        print("6. Applying ICA for artifact removal...")
        # Create ICA object
        ica = mne.preprocessing.ICA(n_components=n_components, random_state=random_state)
        
        # Fit ICA on filtered data
        ica.fit(raw, picks='eeg')
        
        # Find and remove EOG artifacts
        eog_indices, eog_scores = ica.find_bads_eog(raw)
        if eog_indices:
            print(f"   Found {len(eog_indices)} EOG-related components")
            ica.exclude = eog_indices
            
            # Apply ICA to remove artifacts
            raw = ica.apply(raw, exclude=ica.exclude)
            print("   Applied ICA to remove artifacts")
        else:
            print("   No EOG-related components found")
    
    print("\nPreprocessing complete!")
    return raw


def epoch_data(raw: mne.io.Raw, 
               events: np.ndarray, 
               event_id: Dict[str, int], 
               tmin: float = -0.2, 
               tmax: float = 0.5, 
               baseline: Tuple[Optional[float], Optional[float]] = (None, 0),
               reject: Optional[Dict[str, float]] = None) -> mne.Epochs:
    """
    Create epochs from continuous EEG data.
    
    Args:
        raw: Preprocessed MNE Raw object
        events: Events array from mne.find_events
        event_id: Dictionary mapping event names to event codes
        tmin: Start time of epoch relative to event (default: -0.2 s)
        tmax: End time of epoch relative to event (default: 0.5 s)
        baseline: Baseline period for baseline correction (default: (None, 0))
        reject: Rejection parameters for artifact rejection (default: None)
        
    Returns:
        MNE Epochs object
    """
    print("\nCreating epochs...")
    
    # Create default rejection parameters if none provided
    if reject is None:
        reject = {'eeg': 100e-6}  # 100 µV
    
    # Create epochs
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                       baseline=baseline, reject=reject,
                       preload=True, verbose=True)
    
    print(f"Created {len(epochs)} epochs")
    return epochs 