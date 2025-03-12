#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example script for running the CoupledStateSpaceVI model for subject 001.
This script loads the leadfield matrix (C) and EEG data, trains the model to learn
A, B, Q, R, and P, and then saves the trained model.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import mne
from mne.io import read_raw_edf
from pathlib import Path
import sys
import argparse
from scipy.sparse.linalg import svds
import math

# Add the project root to the path so we can import from eeg_processing
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from eeg_processing.models.coupled_state_space_vi import CoupledStateSpaceVI

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description='Run Coupled State Space VI model for EEG data')
    parser.add_argument('--subject', type=str, default='sub-001', help='Subject ID (default: sub-001)')
    parser.add_argument('--session', type=str, default='ses-t1', help='Session ID (default: ses-t1)')
    parser.add_argument('--task', type=str, default='resteyesc', help='Task ID (default: resteyesc)')
    parser.add_argument('--x_dim', type=int, default=8, help='Latent state dimension (default: 8)')
    parser.add_argument('--n_epochs_to_use', type=int, default=10, 
                        help='Number of EEG epochs to use for training (default: 10)')
    parser.add_argument('--training_epochs', type=int, default=100, 
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--beta', type=float, default=0.1, 
                        help='KL divergence weight (default: 0.1)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Check if GPU should be used
    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Paths
    sub_id = args.subject
    ses_id = args.session
    task_id = args.task
    data_root = project_root / "data"
    leadfield_dir = project_root / "leadfield"
    derivatives_dir = data_root / "derivatives" / "cleaned_epochs"
    output_dir = project_root / "eeg_processing" / "results"
    os.makedirs(output_dir, exist_ok=True)


    edf_file = data_root / sub_id / ses_id / "eeg" / f"{sub_id}_{ses_id}_task-{task_id}_eeg.edf"
        
    print(f"Loading continuous EEG data from: {edf_file}")
    
    # Load the raw EDF file
    raw = read_raw_edf(edf_file, preload=True)
    print(f"Raw data loaded: {len(raw.ch_names)} channels, {raw.n_times} time points")
    
    # Extract data and convert to array
    raw_data = raw.get_data()
    
    # Convert from volts to microvolts for better numerical stability
    raw_data = raw_data * 1e3
    print(f"Converting EEG data from millivolts to microvolts (μV)")
    
    # Reshape to (n_timepoints, n_channels) for the model
    eeg_data = raw_data.T

    # Limit to the first 10000 timepoints
    eeg_data = eeg_data[:1000, :]
    print(f"Using only the first 10000 timepoints of EEG data")
    
    # Convert to PyTorch tensor
    eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32, device=device)
    print(f"EEG data shape: {eeg_tensor.shape}")
    
    # Load leadfield 
    leadfield_file = leadfield_dir / f"{sub_id}_{ses_id}_leadfield.npy"
    leadfield = np.load(leadfield_file)
    print(f"Leadfield matrix shape: {leadfield.shape}")
    
    # Convert leadfield to μV/(nAm) for consistent units with EEG data in microvolts
    leadfield = leadfield * 1e-3
    print(f"Converting leadfield to μV/(nAm) for unit consistency")
    
    # Define model parameters
    y_dim = eeg_tensor.shape[1]  # Number of EEG channels
    x_dim = args.x_dim  # Dimension of latent state
    
    print(f"Using latent state dimension: {x_dim}")
    print(f"Observation dimension: {y_dim}")
    
    # Check for NaN or infinite values in leadfield
    if np.any(np.isnan(leadfield)):
        print("Warning: Lead field matrix contains NaN values")
        # Replace NaN with zeros
        leadfield = np.nan_to_num(leadfield, nan=0.0)
    
    if np.any(np.isinf(leadfield)):
        print("Warning: Lead field matrix contains infinite values")
        # Replace inf with large values
        leadfield = np.nan_to_num(leadfield, posinf=1e6, neginf=-1e6)
    
    # Use SVD-based dimensionality reduction for the leadfield
    # This is a more principled approach than simple column selection
    
    # Perform truncated SVD
    n_components = min(x_dim, min(leadfield.shape) - 1)
    U, s, Vh = svds(leadfield, k=n_components)
    
    # Sort in descending order of singular values
    idx = np.argsort(s)[::-1]
    s = s[idx]
    U = U[:, idx]
    Vh = Vh[idx, :]
    
    # Create reduced matrices
    C_downsampled = U @ np.diag(s)
    V = Vh.T  # For back-projection
    
    # Calculate explained variance
    explained_variance = np.sum(s**2) / np.sum(s**2) * 100
    
    print(f"\nDimensionality reduction:")
    print(f"Original leadfield dimensions: {leadfield.shape}")
    print(f"Reduced leadfield dimensions: {C_downsampled.shape}")
    print(f"Number of components kept: {n_components}")
    print(f"Variance explained: {explained_variance:.2f}%")
    
    # Create validation set (20% of data)
    total_timepoints = eeg_tensor.shape[0]
    val_size = int(0.2 * total_timepoints)
    train_size = total_timepoints - val_size
    
    # Important: When using continuous data, split in a way that preserves continuity
    # For training data, use a continuous segment
    train_data = eeg_tensor[:train_size]
    # For validation, use the next continuous segment
    val_data = eeg_tensor[train_size:]
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    
    # Initialize model
    model = CoupledStateSpaceVI(
        x_dim=x_dim,
        y_dim=y_dim,
        C=C_downsampled,
        u_dim=x_dim,  # Set control input dimension equal to state dimension
        beta=args.beta,
        prior_std=1.0,
        dt=1/raw.info['sfreq'],  # Use actual sampling frequency from raw data
        eps=1e-4
    )
    
    # Store V matrix for back-projection (requires model to be updated)
    model.register_buffer('V', torch.tensor(V, dtype=torch.float32))
    
    # Move model to device
    model = model.to(device)
    
    # Set training parameters
    n_epochs = args.training_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    
    # Print training data info
    print("\nTraining data info:")
    print(f"Training data shape: {train_data.shape}")
    print(f"Training data - min: {train_data.min().item()}, max: {train_data.max().item()}, mean: {train_data.mean().item()}")
    print(f"First few values: {train_data[0, 0:3]}")
    
    # Train model
    print(f"\nTraining model for {n_epochs} epochs with batch size {batch_size}...")
    history = model.fit(
        y=train_data,
        n_epochs=n_epochs,
        #batch_size=batch_size,
        learning_rate=learning_rate,
        validation_data=val_data,
        verbose=True
    )
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history['elbo'], label='Training ELBO')
    if history['val_elbo'] is not None:
        plt.plot(history['val_elbo'], label='Validation ELBO')
    plt.xlabel('Epoch')
    plt.ylabel('ELBO')
    plt.title(f'Training History for {sub_id}')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(output_dir / f"{sub_id}_training_history.png")
    plt.close()
    
    # Forward pass on validation data to get predictions and latent states
    print("\nValidation data info:")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Validation data - min: {val_data.min().item()}, max: {val_data.max().item()}, mean: {val_data.mean().item()}")
    print(f"First few values: {val_data[0, 0:3]}")
    
    # Run prediction on validation data
    print("\nRunning forward pass on validation data...")
    output = model.predict(val_data, project_to_sources=True)
    
    # Extract states and predictions
    latent_states = output['x'].squeeze(0)  # Remove batch dimension
    predicted_obs = output['y_pred'].squeeze(0)  # Remove batch dimension
    
    # Print prediction info
    print("\nPrediction results:")
    print(f"Predicted observations shape: {predicted_obs.shape}")
    print(f"Predicted obs - min: {predicted_obs.min().item()}, max: {predicted_obs.max().item()}, mean: {predicted_obs.mean().item()}")
    print(f"First few predicted values: {predicted_obs[0, 0:3]}")
    
    # Compare scales directly
    obs_max = val_data.max().item()
    pred_max = predicted_obs.max().item()
    print(f"\nScale comparison:")
    print(f"Observation max: {obs_max:.6e}")
    print(f"Prediction max: {pred_max:.6e}")
    print(f"Ratio (pred/obs): {pred_max/obs_max:.6e}")
    
    # Run prediction on training data to check for overfitting
    print("\nRunning forward pass on training data...")
    train_output = model.predict(train_data, project_to_sources=True)
    train_predicted_obs = train_output['y_pred'].squeeze(0)  # Remove batch dimension
    
    # Print training prediction info
    print("\nTraining prediction results:")
    print(f"Training predicted observations shape: {train_predicted_obs.shape}")
    print(f"Training predicted obs - min: {train_predicted_obs.min().item()}, max: {train_predicted_obs.max().item()}, mean: {train_predicted_obs.mean().item()}")
    
    # Compare training prediction vs observed
    train_obs_max = train_data.max().item()
    train_pred_max = train_predicted_obs.max().item()
    print(f"\nTraining data scale comparison:")
    print(f"Training observation max: {train_obs_max:.6e}")
    print(f"Training prediction max: {train_pred_max:.6e}")
    print(f"Ratio (pred/obs): {train_pred_max/train_obs_max:.6e}")
    
    # Plot latent states
    plt.figure(figsize=(12, 8))
    for i in range(min(x_dim, 8)):  # Plot up to 8 latent states
        plt.subplot(4, 2, i+1)
        plt.plot(latent_states[:, i].cpu().numpy())
        plt.title(f'Latent State {i+1}')
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"{sub_id}_latent_states.png")
    plt.close()
    
    # Plot a subset of observed vs predicted EEG channels for training data
    channels_to_plot = min(6, y_dim)  # Plot up to 6 channels
    plt.figure(figsize=(15, 10))
    for i in range(channels_to_plot):
        plt.subplot(3, 2, i+1)
        plt.plot(train_data[:, i].cpu().numpy(), label='Observed', alpha=0.7)
        plt.plot(train_predicted_obs[:, i].cpu().numpy(), label='Predicted', alpha=0.7)
        plt.title(f'Training Channel {i+1}')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"{sub_id}_training_predictions.png")
    plt.close()
    
    # Plot a subset of observed vs predicted EEG channels for validation data
    plt.figure(figsize=(15, 10))
    for i in range(channels_to_plot):
        plt.subplot(3, 2, i+1)
        plt.plot(val_data[:, i].cpu().numpy(), label='Observed', alpha=0.7)
        plt.plot(predicted_obs[:, i].cpu().numpy(), label='Predicted', alpha=0.7)
        plt.title(f'Validation Channel {i+1}')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"{sub_id}_validation_predictions.png")
    plt.close()
    
    # If source activity is available, plot it
    if 'source_activity' in output:
        source_activity = output['source_activity'].squeeze(0)
        # Plot a subset of source activities
        sources_to_plot = min(6, source_activity.shape[1])
        plt.figure(figsize=(15, 10))
        for i in range(sources_to_plot):
            plt.subplot(3, 2, i+1)
            plt.plot(source_activity[:, i].cpu().numpy())
            plt.title(f'Source {i+1}')
            plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / f"{sub_id}_source_activity.png")
        plt.close()
    
    # Get learned matrices
    A = model.A.detach().cpu().numpy()
    B = model.B.detach().cpu().numpy()
    Q = model.Q.detach().cpu().numpy()
    R = model.R.detach().cpu().numpy()
    
    # Print matrix information
    print("\nLearned model parameters:")
    print(f"A shape: {A.shape}")
    print(f"B shape: {B.shape}")
    print(f"Q shape: {Q.shape}")
    print(f"R shape: {R.shape}")
    
    # Save model
    model_save_path = output_dir / f"{sub_id}_coupled_state_space_model.pt"
    model.save(str(model_save_path))
    print(f"Model saved to {model_save_path}")
    
    # Also save the learned matrices
    matrices_save_path = output_dir / f"{sub_id}_learned_matrices.npz"
    
    # Prepare matrices dict
    matrices_dict = {
        'A': A,
        'B': B,
        'C': model.C.detach().cpu().numpy(),
        'Q': Q,
        'R': R,
        'V': model.V.detach().cpu().numpy(),  # Save V matrix for back-projection
        'leadfield_full': leadfield  # Save the full leadfield
    }
    
    np.savez(
        matrices_save_path,
        **matrices_dict
    )
    print(f"Learned matrices saved to {matrices_save_path}")
    
    return model

if __name__ == "__main__":
    model = main() 