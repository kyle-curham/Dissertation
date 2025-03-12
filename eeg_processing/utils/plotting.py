"""
Plotting utilities for visualizing state space model results.
These functions are designed to work with the CoupledStateSpace model for EEG data.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
from pathlib import Path


def plot_training_history(history: Dict[str, List[float]], n_epochs: int, 
                          has_validation: bool = False) -> plt.Figure:
    """
    Plot training metrics over epochs.
    
    Args:
        history: Dictionary containing training history
        n_epochs: Number of epochs
        has_validation: Whether validation data was used
        
    Returns:
        Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Log-Likelihood
    axes[0].plot(range(1, n_epochs + 1), history['log_likelihood'], label='Train Log-Likelihood')
    if has_validation:
        axes[0].plot(range(1, n_epochs + 1), history['val_log_likelihood'], label='Val Log-Likelihood')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Log-Likelihood')
    axes[0].set_title('Log-Likelihood Over Time')
    axes[0].legend()
    
    # Plot Control Cost
    axes[1].plot(range(1, n_epochs + 1), history['control_cost'], label='Train Control Cost')
    if has_validation:
        axes[1].plot(range(1, n_epochs + 1), history['val_control_cost'], label='Val Control Cost')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Control Cost')
    axes[1].set_title('Control Cost Over Time')
    axes[1].legend()
    
    plt.tight_layout()
    return fig


def plot_prediction_comparison(predictions: Dict[int, Dict[str, np.ndarray]], 
                               plot_epochs: List[int], n_epochs: int, 
                               time_array: np.ndarray, channel_idx: int = 0,
                               val_time_array: Optional[np.ndarray] = None) -> plt.Figure:
    """
    Plot prediction comparisons across training epochs and between train/validation.
    
    Args:
        predictions: Dictionary of predictions at different epochs
        plot_epochs: List of epochs to plot
        n_epochs: Total number of epochs (for accessing final predictions)
        time_array: Time array for x-axis (training data)
        channel_idx: Channel index to plot (default: 0)
        val_time_array: Time array for validation data (optional)
        
    Returns:
        Figure object
    """
    has_validation = val_time_array is not None and 'val_true' in predictions[n_epochs]
    
    # Create a 1x3 layout: Training | Validation | Combined Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Training Data Progression
    # Plot training predictions at different epochs
    for epoch in plot_epochs:
        if epoch in predictions:
            axes[0].plot(time_array, predictions[epoch]['train_pred'][0, :, channel_idx], 
                        label=f'Epoch {epoch}', alpha=0.6)
    
    # Plot ground truth once
    axes[0].plot(time_array, predictions[n_epochs]['train_true'][0, :, channel_idx], 
                'k--', linewidth=2, label='True', alpha=0.9)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Normalized Amplitude')
    axes[0].set_title(f'TRAINING: Channel {channel_idx+1}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Validation Data Progression
    if has_validation:
        # Plot validation predictions at different epochs
        for epoch in plot_epochs:
            if epoch in predictions and 'val_pred' in predictions[epoch]:
                axes[1].plot(val_time_array, predictions[epoch]['val_pred'][0, :, channel_idx], 
                            label=f'Epoch {epoch}', alpha=0.6)
        
        # Plot ground truth once
        axes[1].plot(val_time_array, predictions[n_epochs]['val_true'][0, :, channel_idx], 
                    'k--', linewidth=2, label='True', alpha=0.9)
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Normalized Amplitude')
        axes[1].set_title(f'VALIDATION: Channel {channel_idx+1}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No Validation Data Available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=axes[1].transAxes, fontsize=14)
        axes[1].set_title('Validation Predictions')
        axes[1].axis('off')
    
    # Plot 3: Final Epoch Direct Comparison
    # For final epoch, show error/improvement visually
    
    # Training error
    train_true = predictions[n_epochs]['train_true'][0, :, channel_idx]
    train_pred = predictions[n_epochs]['train_pred'][0, :, channel_idx]
    axes[2].plot(time_array, train_true, 'b--', label='Train True', alpha=0.6)
    axes[2].plot(time_array, train_pred, 'b-', label='Train Pred', alpha=0.6)
    
    # Fill the error region between true and predicted for visualization
    axes[2].fill_between(time_array, train_true, train_pred, color='blue', alpha=0.1)
    
    # Validation error if available
    if has_validation:
        val_true = predictions[n_epochs]['val_true'][0, :, channel_idx]
        val_pred = predictions[n_epochs]['val_pred'][0, :, channel_idx]
        
        # Offset validation time to show separately
        offset = len(time_array) + len(val_time_array) * 0.1  # 10% gap
        offset_time = val_time_array + offset
        
        axes[2].plot(offset_time, val_true, 'r--', label='Val True', alpha=0.6)
        axes[2].plot(offset_time, val_pred, 'r-', label='Val Pred', alpha=0.6)
        axes[2].fill_between(offset_time, val_true, val_pred, color='red', alpha=0.1)
        
        # Add a vertical line to separate training and validation
        axes[2].axvline(x=offset - len(val_time_array) * 0.05, color='gray', linestyle='--', alpha=0.5)
        
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Normalized Amplitude')
    axes[2].set_title(f'Final Epoch Comparison: Channel {channel_idx+1}')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_latent_dynamics(states: np.ndarray, controls: np.ndarray, 
                        time_array: np.ndarray, max_dims: int = 4) -> plt.Figure:
    """
    Plot latent state dynamics and optimal control inputs.
    
    Args:
        states: Latent state trajectories (shape: [time, state_dim])
        controls: Control input trajectories (shape: [time, control_dim])
        time_array: Time array for x-axis
        max_dims: Maximum number of dimensions to plot
        
    Returns:
        Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot states
    num_states_to_plot = min(max_dims, states.shape[1])
    for i in range(num_states_to_plot):
        axes[0].plot(time_array, states[:, i], label=f'State {i+1}')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('State Value')
    axes[0].set_title('Latent State Trajectories')
    axes[0].legend()
    
    # Plot controls
    num_controls_to_plot = min(max_dims, controls.shape[1])
    for i in range(num_controls_to_plot):
        axes[1].plot(time_array, controls[:, i], label=f'Control {i+1}')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Control Value')
    axes[1].set_title('Optimal Control Inputs')
    axes[1].legend()
    
    plt.tight_layout()
    return fig


def plot_matrix_analysis(A: np.ndarray, B: np.ndarray, train_data: np.ndarray, 
                         train_pred: np.ndarray, val_data: Optional[np.ndarray] = None, 
                         val_pred: Optional[np.ndarray] = None, 
                         channel_idx: int = 0) -> plt.Figure:
    """
    Plot matrix analysis including matrix visualizations, eigenvalues, and predictions.
    
    Args:
        A: System dynamics matrix
        B: Control input matrix
        train_data: Training data (shape: [batch, time, channels] or [time, channels])
        train_pred: Predictions on training data (shape: [batch, time, channels] or [time, channels])
        val_data: Validation data (optional) (shape: [batch, time, channels] or [time, channels])
        val_pred: Predictions on validation data (optional) (shape: [batch, time, channels] or [time, channels])
        channel_idx: Channel index to plot
        
    Returns:
        Figure object
    """
    fig = plt.figure(figsize=(15, 15))
    
    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvals(A)
    
    # Plot matrix A heatmap
    ax1 = fig.add_subplot(2, 2, 1)
    im1 = ax1.imshow(A, cmap='coolwarm', aspect='auto')
    plt.colorbar(im1, ax=ax1, label='Value')
    ax1.set_title('Dynamics Matrix A')
    ax1.set_xlabel('State Index')
    ax1.set_ylabel('State Index')
    
    # Plot matrix B heatmap
    ax2 = fig.add_subplot(2, 2, 2)
    im2 = ax2.imshow(B, cmap='coolwarm', aspect='auto')
    plt.colorbar(im2, ax=ax2, label='Value')
    ax2.set_title('Control Matrix B')
    ax2.set_xlabel('Control Index')
    ax2.set_ylabel('State Index')
    
    # Plot eigenvalues in complex plane
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(eigenvalues.real, eigenvalues.imag)
    ax3.axvline(x=0, color='r', linestyle='--', alpha=0.3)
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    ax3.grid(True)
    ax3.set_title('Eigenvalues of A in Complex Plane')
    ax3.set_xlabel('Real Part')
    ax3.set_ylabel('Imaginary Part')
    
    # Plot comparison of predicted vs true for a specific channel
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Handle dimensionality of input data
    # If 3D [batch, time, channels], extract first batch
    # If 2D [time, channels], use as is
    if train_data.ndim == 3:
        train_data_plot = train_data[0, :, channel_idx]
    else:  # 2D
        train_data_plot = train_data[:, channel_idx]
        
    if train_pred.ndim == 3:
        train_pred_plot = train_pred[0, :, channel_idx]
    else:  # 2D
        train_pred_plot = train_pred[:, channel_idx]
    
    # Training data
    train_time = np.arange(train_data_plot.shape[0])
    ax4.plot(train_time, train_data_plot, 'b-', alpha=0.5, label='Train True')
    ax4.plot(train_time, train_pred_plot, 'r-', label='Train Pred')
    
    # Validation data if available
    if val_data is not None and val_pred is not None:
        # Handle dimensionality for validation data
        if val_data.ndim == 3:
            val_data_plot = val_data[0, :, channel_idx]
        else:  # 2D
            val_data_plot = val_data[:, channel_idx]
            
        if val_pred.ndim == 3:
            val_pred_plot = val_pred[0, :, channel_idx]
        else:  # 2D
            val_pred_plot = val_pred[:, channel_idx]
            
        val_time = np.arange(val_data_plot.shape[0]) + len(train_time)
        ax4.plot(val_time, val_data_plot, 'g-', alpha=0.5, label='Val True')
        ax4.plot(val_time, val_pred_plot, 'm-', label='Val Pred')
    
    ax4.set_title(f'Channel {channel_idx+1}: True vs Predicted')
    ax4.set_xlabel('Time Points')
    ax4.set_ylabel('Amplitude')
    ax4.legend()
    
    plt.tight_layout()
    return fig


def plot_training_results(output: Dict[str, Any], history: Dict[str, List[float]], 
                         predictions: Dict[int, Dict[str, np.ndarray]], 
                         n_epochs: int, plot_epochs: List[int], 
                         y: np.ndarray, validation_data: Optional[np.ndarray] = None) -> plt.Figure:
    """
    Plot comprehensive training results including metrics, predictions, states, and controls.
    
    Args:
        output: Output from the forward pass at the final epoch
        history: Training history dictionary
        predictions: Predictions at different epochs
        n_epochs: Total number of epochs
        plot_epochs: List of epochs to plot
        y: Training data
        validation_data: Validation data (optional)
        
    Returns:
        Figure object
    """
    # Create a 3x2 layout with:
    # Row 1: Training and Validation Metrics (LL and Control Cost)
    # Row 2: Training Predictions | Validation Predictions
    # Row 3: Latent States | Control Inputs
    
    has_validation = validation_data is not None
    fig = plt.figure(figsize=(18, 12))
    
    # === ROW 1: Training and Validation Metrics ===
    
    # Plot 1: Log-Likelihood
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(range(1, n_epochs + 1), history['log_likelihood'], 'b-', label='Train')
    if has_validation:
        ax1.plot(range(1, n_epochs + 1), history['val_log_likelihood'], 'r-', label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Log-Likelihood')
    ax1.set_title('Log-Likelihood Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Control Cost
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(range(1, n_epochs + 1), history['control_cost'], 'b-', label='Train')
    if has_validation:
        ax2.plot(range(1, n_epochs + 1), history['val_control_cost'], 'r-', label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Control Cost')
    ax2.set_title('Control Cost Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # === ROW 2: Training and Validation Predictions ===
    time = np.arange(y.shape[1])
    channel_idx = 0  # First channel by default
    
    # Plot 3: Training Data Predictions (Across Epochs)
    ax3 = fig.add_subplot(3, 2, 3)
    # Plot training predictions at different epochs to show improvement
    for epoch in plot_epochs:
        if epoch in predictions:
            ax3.plot(time, predictions[epoch]['train_pred'][0, :, channel_idx], 
                   label=f'Epoch {epoch}', alpha=0.6)
    # Plot ground truth once
    ax3.plot(time, predictions[n_epochs]['train_true'][0, :, channel_idx], 
           'k--', linewidth=2, label='True', alpha=0.9)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Normalized Amplitude')
    ax3.set_title(f'TRAINING: Channel {channel_idx+1} Over Epochs')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Validation Data Predictions
    ax4 = fig.add_subplot(3, 2, 4)
    if has_validation and 'val_true' in predictions[n_epochs]:
        val_time = np.arange(validation_data.shape[1])
        
        # Plot validation true data
        ax4.plot(val_time, predictions[n_epochs]['val_true'][0, :, channel_idx], 
               'k--', linewidth=2, label='True', alpha=0.9)
        
        # If we have multiple epochs, show progression on validation too
        for epoch in plot_epochs:
            if epoch in predictions and 'val_pred' in predictions[epoch]:
                ax4.plot(val_time, predictions[epoch]['val_pred'][0, :, channel_idx], 
                       label=f'Epoch {epoch}', alpha=0.6)
        
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Normalized Amplitude')
        ax4.set_title(f'VALIDATION: Channel {channel_idx+1}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No Validation Data Available', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax4.transAxes, fontsize=14)
        ax4.set_title('Validation Predictions')
        ax4.axis('off')
    
    # === ROW 3: Model Internals ===
    
    # Plot 5: Latent States
    ax5 = fig.add_subplot(3, 2, 5)
    states = output['states'].cpu().numpy()[0]  # First batch
    num_states_to_plot = min(4, states.shape[1])
    for i in range(num_states_to_plot):
        ax5.plot(time, states[:, i], label=f'State {i+1}')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('State Value')
    ax5.set_title('Latent State Trajectories')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Control Inputs
    ax6 = fig.add_subplot(3, 2, 6)
    controls = output['controls'].cpu().numpy()[0]  # First batch
    num_controls_to_plot = min(4, controls.shape[1])
    for i in range(num_controls_to_plot):
        ax6.plot(time, controls[:, i], label=f'Control {i+1}')
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Control Value')
    ax6.set_title('Optimal Control Inputs')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_direct_comparison(y_true: np.ndarray, y_pred: np.ndarray, 
                           channel_idx: int = 0, title: str = None,
                           time_array: Optional[np.ndarray] = None) -> plt.Figure:
    """
    Directly compare predicted and observed time series for a single channel.
    
    Args:
        y_true: True/observed data (shape: [batch, time, channels] or [time, channels])
        y_pred: Predicted data (shape: [batch, time, channels] or [time, channels])  
        channel_idx: The channel index to plot (default: 0)
        title: Optional title for the plot
        time_array: Optional time array for x-axis (default: np.arange(len(data)))
        
    Returns:
        Figure object
    """
    # Handle different input shapes
    if y_true.ndim == 3:
        y_true = y_true[0]  # Take first batch
    if y_pred.ndim == 3:
        y_pred = y_pred[0]  # Take first batch
        
    # Create time array if not provided
    if time_array is None:
        time_array = np.arange(y_true.shape[0])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the true and predicted time series
    ax.plot(time_array, y_true[:, channel_idx], 'b-', label='Observed', linewidth=1.5, alpha=0.7)
    ax.plot(time_array, y_pred[:, channel_idx], 'r-', label='Predicted', linewidth=1.5)
    
    # Add labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Channel {channel_idx+1}: Observed vs Predicted')
    
    # Add legend and grid
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig 