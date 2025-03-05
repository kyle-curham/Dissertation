"""
EEG Visualization Functions

This module provides functions for visualizing EEG data and analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
from typing import Optional, Dict, List, Tuple, Union, Any
from pathlib import Path


def plot_raw_eeg(raw: mne.io.Raw, 
                 duration: float = 10.0, 
                 start: float = 0.0,
                 n_channels: Optional[int] = None,
                 show_scalebars: bool = True,
                 title: Optional[str] = None,
                 show: bool = True,
                 save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Plot raw EEG data.
    
    Args:
        raw: MNE Raw object to plot
        duration: Duration of data to plot in seconds (default: 10.0)
        start: Start time in seconds (default: 0.0)
        n_channels: Number of channels to plot (default: None, plot all)
        show_scalebars: Whether to show scale bars (default: True)
        title: Title for the plot (default: None)
        show: Whether to show the plot (default: True)
        save_path: Path to save the plot to (default: None, don't save)
        
    Returns:
        Matplotlib Figure object
    """
    # Create a copy of the raw object to avoid modifying the original
    raw_plot = raw.copy()
    
    # Set title if not provided
    if title is None:
        title = f"Raw EEG Data ({raw.info['sfreq']} Hz)"
    
    # Create the plot
    fig = raw_plot.plot(
        duration=duration,
        start=start,
        n_channels=n_channels,
        scalings='auto',
        title=title,
        show_scrollbars=True,
        show_scalebars=show_scalebars,
        block=False,
        show=show
    )
    
    # Save the plot if a path is provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    return fig


def plot_psd(raw: mne.io.Raw,
             fmin: float = 0.0,
             fmax: float = 50.0,
             tmin: Optional[float] = None,
             tmax: Optional[float] = None,
             picks: Optional[Union[str, List[str]]] = 'eeg',
             show: bool = True,
             save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Plot power spectral density (PSD) of EEG data.
    
    Args:
        raw: MNE Raw object to plot
        fmin: Minimum frequency to include (default: 0.0 Hz)
        fmax: Maximum frequency to include (default: 50.0 Hz)
        tmin: Start time for PSD calculation (default: None, use all data)
        tmax: End time for PSD calculation (default: None, use all data)
        picks: Channels to include (default: 'eeg', all EEG channels)
        show: Whether to show the plot (default: True)
        save_path: Path to save the plot to (default: None, don't save)
        
    Returns:
        Matplotlib Figure object
    """
    # Create the plot
    fig = raw.plot_psd(
        fmin=fmin,
        fmax=fmax,
        tmin=tmin,
        tmax=tmax,
        picks=picks,
        show=show
    )
    
    # Save the plot if a path is provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"PSD plot saved to {save_path}")
    
    return fig


def plot_topomap(data: np.ndarray,
                info: mne.Info,
                times: Optional[Union[float, List[float]]] = None,
                title: Optional[Union[str, List[str]]] = None,
                colorbar: bool = True,
                vmin: Optional[float] = None,
                vmax: Optional[float] = None,
                cmap: str = 'RdBu_r',
                show: bool = True,
                save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Plot topographic maps of EEG data.
    
    Args:
        data: Data to plot (channels x times)
        info: MNE Info object with channel positions
        times: Time point(s) to plot (default: None, average across time)
        title: Title(s) for the plot(s) (default: None)
        colorbar: Whether to show a colorbar (default: True)
        vmin: Minimum value for color scaling (default: None, automatic)
        vmax: Maximum value for color scaling (default: None, automatic)
        cmap: Colormap to use (default: 'RdBu_r')
        show: Whether to show the plot (default: True)
        save_path: Path to save the plot to (default: None, don't save)
        
    Returns:
        Matplotlib Figure object
    """
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the topomap
    im, _ = mne.viz.plot_topomap(
        data,
        info,
        times=times,
        axes=ax,
        show=False,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        colorbar=colorbar
    )
    
    # Set title if provided
    if title is not None:
        ax.set_title(title)
    
    # Show the plot if requested
    if show:
        plt.show()
    
    # Save the plot if a path is provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Topomap saved to {save_path}")
    
    return fig


def plot_epochs_image(epochs: mne.Epochs,
                     picks: Optional[Union[str, List[str]]] = None,
                     sigma: float = 0.0,
                     vmin: Optional[float] = None,
                     vmax: Optional[float] = None,
                     colorbar: bool = True,
                     order: Optional[List[int]] = None,
                     show: bool = True,
                     title: Optional[str] = None,
                     save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Plot epochs as an image.
    
    Args:
        epochs: MNE Epochs object to plot
        picks: Channels to plot (default: None, plot first channel)
        sigma: Amount of smoothing to apply (default: 0.0, no smoothing)
        vmin: Minimum value for color scaling (default: None, automatic)
        vmax: Maximum value for color scaling (default: None, automatic)
        colorbar: Whether to show a colorbar (default: True)
        order: Order in which to plot epochs (default: None, chronological)
        show: Whether to show the plot (default: True)
        title: Title for the plot (default: None)
        save_path: Path to save the plot to (default: None, don't save)
        
    Returns:
        Matplotlib Figure object
    """
    # Create the plot
    fig = epochs.plot_image(
        picks=picks,
        sigma=sigma,
        vmin=vmin,
        vmax=vmax,
        colorbar=colorbar,
        order=order,
        show=show,
        title=title
    )
    
    # Save the plot if a path is provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Epochs image saved to {save_path}")
    
    return fig 