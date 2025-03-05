"""
Visualization utilities for model evaluation.

This module provides functions for visualizing model outputs and evaluation metrics.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Tuple, List, Dict, Optional, Union, Any
import seaborn as sns
from pathlib import Path


def plot_state_trajectories(
    states: torch.Tensor,
    time_axis: Optional[np.ndarray] = None,
    title: str = "State Trajectories",
    n_states_to_plot: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot state trajectories over time.
    
    Args:
        states: Tensor of shape (time_steps, n_states) or (batch, time_steps, n_states)
        time_axis: Optional time axis values
        title: Plot title
        n_states_to_plot: Number of states to plot (default: all)
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    # Handle batched input
    if states.dim() == 3:
        # Average over batch dimension
        states = states.mean(dim=0)
    
    # Convert to numpy if it's a torch tensor
    if isinstance(states, torch.Tensor):
        states = states.detach().cpu().numpy()
    
    time_steps, n_states = states.shape
    
    # Use sequence indices if time_axis not provided
    if time_axis is None:
        time_axis = np.arange(time_steps)
    
    # Determine number of states to plot
    if n_states_to_plot is None:
        n_states_to_plot = min(n_states, 12)  # Plot at most 12 states
    
    # Create figure
    fig, axes = plt.subplots(n_states_to_plot, 1, figsize=figsize, sharex=True)
    if n_states_to_plot == 1:
        axes = [axes]  # Make sure axes is always a list
    
    # Plot each state
    for i in range(n_states_to_plot):
        axes[i].plot(time_axis, states[:, i])
        axes[i].set_ylabel(f"State {i+1}")
        axes[i].grid(True)
    
    # Set title and labels
    plt.suptitle(title)
    plt.xlabel("Time")
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_state_distributions(
    states: torch.Tensor,
    covariances: Optional[torch.Tensor] = None,
    title: str = "State Distributions",
    n_states_to_plot: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot distributions of state values.
    
    Args:
        states: Tensor of shape (time_steps, n_states) or (batch, time_steps, n_states)
        covariances: Optional tensor of state covariances
        title: Plot title
        n_states_to_plot: Number of states to plot (default: all)
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    # Handle batched input
    if states.dim() == 3:
        # Reshape to (batch * time_steps, n_states)
        batch, time_steps, n_states = states.shape
        states = states.reshape(batch * time_steps, n_states)
    
    # Convert to numpy if it's a torch tensor
    if isinstance(states, torch.Tensor):
        states = states.detach().cpu().numpy()
    
    n_states = states.shape[1]
    
    # Determine number of states to plot
    if n_states_to_plot is None:
        n_states_to_plot = min(n_states, 12)  # Plot at most 12 states
    
    # Create figure
    fig, axes = plt.subplots(n_states_to_plot, 1, figsize=figsize)
    if n_states_to_plot == 1:
        axes = [axes]  # Make sure axes is always a list
    
    # Plot each state distribution
    for i in range(n_states_to_plot):
        sns.histplot(states[:, i], ax=axes[i], kde=True)
        axes[i].set_xlabel(f"State {i+1}")
        axes[i].grid(True)
    
    # Set title
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_reconstruction(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    time_axis: Optional[np.ndarray] = None,
    title: str = "Original vs Reconstructed",
    n_channels_to_plot: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot original data against reconstructed data.
    
    Args:
        original: Original data tensor of shape (time_steps, n_channels)
        reconstructed: Reconstructed data tensor of shape (time_steps, n_channels)
        time_axis: Optional time axis values
        title: Plot title
        n_channels_to_plot: Number of channels to plot (default: all)
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy if they're torch tensors
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()
    
    time_steps, n_channels = original.shape
    
    # Use sequence indices if time_axis not provided
    if time_axis is None:
        time_axis = np.arange(time_steps)
    
    # Determine number of channels to plot
    if n_channels_to_plot is None:
        n_channels_to_plot = min(n_channels, 6)  # Plot at most 6 channels
    
    # Create figure
    fig, axes = plt.subplots(n_channels_to_plot, 1, figsize=figsize, sharex=True)
    if n_channels_to_plot == 1:
        axes = [axes]  # Make sure axes is always a list
    
    # Plot each channel
    for i in range(n_channels_to_plot):
        axes[i].plot(time_axis, original[:, i], label="Original")
        axes[i].plot(time_axis, reconstructed[:, i], label="Reconstructed", alpha=0.7)
        axes[i].set_ylabel(f"Channel {i+1}")
        axes[i].grid(True)
        axes[i].legend()
    
    # Set title and labels
    plt.suptitle(title)
    plt.xlabel("Time")
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_system_matrices(
    A: torch.Tensor,
    C: torch.Tensor,
    Q: Optional[torch.Tensor] = None,
    R: Optional[torch.Tensor] = None,
    title: str = "System Matrices",
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize system matrices (A, C, Q, R) of a state-space model.
    
    Args:
        A: State transition matrix
        C: Observation matrix
        Q: Process noise covariance (optional)
        R: Observation noise covariance (optional)
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy if they're torch tensors
    if isinstance(A, torch.Tensor):
        A = A.detach().cpu().numpy()
    if isinstance(C, torch.Tensor):
        C = C.detach().cpu().numpy()
    if Q is not None and isinstance(Q, torch.Tensor):
        Q = Q.detach().cpu().numpy()
    if R is not None and isinstance(R, torch.Tensor):
        R = R.detach().cpu().numpy()
    
    # Determine number of plots needed
    n_plots = 2
    if Q is not None:
        n_plots += 1
    if R is not None:
        n_plots += 1
    
    # Create figure
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    # Plot A matrix
    im0 = axes[0].imshow(A, cmap="coolwarm")
    axes[0].set_title("A: State Transition")
    axes[0].set_xlabel("From State")
    axes[0].set_ylabel("To State")
    plt.colorbar(im0, ax=axes[0])
    
    # Plot C matrix
    im1 = axes[1].imshow(C, cmap="coolwarm")
    axes[1].set_title("C: Observation")
    axes[1].set_xlabel("State")
    axes[1].set_ylabel("Channel")
    plt.colorbar(im1, ax=axes[1])
    
    plot_idx = 2
    
    # Plot Q matrix if provided
    if Q is not None:
        im2 = axes[plot_idx].imshow(Q, cmap="coolwarm")
        axes[plot_idx].set_title("Q: Process Noise Cov")
        axes[plot_idx].set_xlabel("State")
        axes[plot_idx].set_ylabel("State")
        plt.colorbar(im2, ax=axes[plot_idx])
        plot_idx += 1
    
    # Plot R matrix if provided
    if R is not None:
        im3 = axes[plot_idx].imshow(R, cmap="coolwarm")
        axes[plot_idx].set_title("R: Observation Noise Cov")
        axes[plot_idx].set_xlabel("Channel")
        axes[plot_idx].set_ylabel("Channel")
        plt.colorbar(im3, ax=axes[plot_idx])
    
    # Set title
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    metrics: Optional[List[str]] = None,
    title: str = "Training History",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training history metrics.
    
    Args:
        history: Dictionary of training metrics
        metrics: List of metrics to plot (default: all)
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    # Determine metrics to plot
    if metrics is None:
        metrics = list(history.keys())
    
    # Create figure
    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)
    if len(metrics) == 1:
        axes = [axes]  # Make sure axes is always a list
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        axes[i].plot(history[metric])
        axes[i].set_ylabel(metric)
        axes[i].grid(True)
    
    # Set title and labels
    plt.suptitle(title)
    plt.xlabel("Epoch")
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_eigenvalue_analysis(
    A: torch.Tensor,
    title: str = "Eigenvalue Analysis",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Analyze and plot eigenvalues of the state transition matrix A.
    
    Args:
        A: State transition matrix
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(A, torch.Tensor):
        A = A.detach().cpu().numpy()
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(A)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
    
    # Plot eigenvalues in complex plane
    ax.scatter(eigenvalues.real, eigenvalues.imag, s=50)
    
    # Add labels for each eigenvalue
    for i, eig in enumerate(eigenvalues):
        ax.annotate(
            f"{i+1}",
            (eig.real, eig.imag),
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    # Add magnitude = 1 circle for stability analysis
    ax.grid(True)
    ax.set_xlabel("Real Part")
    ax.set_ylabel("Imaginary Part")
    ax.set_title(f"{title}\nEigenvalues of A")
    
    # Make axes equal and add origin lines
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.set_aspect('equal')
    
    # Add stability information
    max_abs = np.max(np.abs(eigenvalues))
    is_stable = max_abs < 1
    stability_text = f"System is {'stable' if is_stable else 'unstable'}"
    max_abs_text = f"Max eigenvalue magnitude: {max_abs:.4f}"
    ax.text(
        0.05, 0.05, 
        f"{stability_text}\n{max_abs_text}",
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def visualize_coupled_model(
    model,
    data: torch.Tensor,
    n_samples: int = 10,
    device: Optional[torch.device] = None,
    save_dir: Optional[str] = None
):
    """
    Comprehensive visualization of a CoupledStateSpaceVI model.
    
    Args:
        model: CoupledStateSpaceVI model instance
        data: Input data tensor of shape (time_steps, n_channels)
        n_samples: Number of samples to draw from the model
        device: PyTorch device
        save_dir: Directory to save figures
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    # Move data to device if specified
    if device is not None:
        data = data.to(device)
    
    # Get model matrices
    A = model.A
    C = model.C
    
    # Sample states and inputs
    with torch.no_grad():
        x_samples, u_samples = model.sample_augmented_state(data, n_samples=n_samples)
    
    # Compute mean states and inputs
    x_mean = x_samples.mean(dim=0)
    u_mean = u_samples.mean(dim=0)
    
    # Compute standard deviations
    x_std = x_samples.std(dim=0)
    u_std = u_samples.std(dim=0)
    
    # Compute reconstruction
    reconstruction = C @ x_mean.T
    reconstruction = reconstruction.T
    
    # Plot states
    fig1 = plot_state_trajectories(
        x_mean,
        title="Latent State Trajectories",
        save_path=os.path.join(save_dir, "state_trajectories.png") if save_dir else None
    )
    
    # Plot inputs
    fig2 = plot_state_trajectories(
        u_mean,
        title="Latent Input Trajectories",
        save_path=os.path.join(save_dir, "input_trajectories.png") if save_dir else None
    )
    
    # Plot reconstruction
    fig3 = plot_reconstruction(
        data.cpu(), 
        reconstruction.cpu(),
        title="Original vs Reconstructed Data",
        save_path=os.path.join(save_dir, "reconstruction.png") if save_dir else None
    )
    
    # Plot system matrices
    fig4 = plot_system_matrices(
        A.cpu(), 
        C.cpu(),
        title="System Matrices",
        save_path=os.path.join(save_dir, "system_matrices.png") if save_dir else None
    )
    
    # Plot eigenvalue analysis
    fig5 = plot_eigenvalue_analysis(
        A.cpu(),
        title="Dynamics Analysis",
        save_path=os.path.join(save_dir, "eigenvalue_analysis.png") if save_dir else None
    )
    
    # Plot state uncertainty
    fig6, axs = plt.subplots(min(4, x_mean.shape[1]), 1, figsize=(12, 8), sharex=True)
    if min(4, x_mean.shape[1]) == 1:
        axs = [axs]
    
    for i in range(min(4, x_mean.shape[1])):
        axs[i].plot(x_mean[:, i].cpu().numpy())
        axs[i].fill_between(
            np.arange(len(x_mean)),
            (x_mean[:, i] - x_std[:, i]).cpu().numpy(),
            (x_mean[:, i] + x_std[:, i]).cpu().numpy(),
            alpha=0.3
        )
        axs[i].set_ylabel(f"State {i+1}")
        axs[i].grid(True)
    
    plt.suptitle("State Trajectories with Uncertainty")
    plt.xlabel("Time")
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, "state_uncertainty.png"), dpi=300, bbox_inches="tight")
    
    # Return all figures
    return {
        "state_trajectories": fig1,
        "input_trajectories": fig2,
        "reconstruction": fig3,
        "system_matrices": fig4,
        "eigenvalue_analysis": fig5,
        "state_uncertainty": fig6
    } 