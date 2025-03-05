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
from mne.io import read_epochs_eeglab
from pathlib import Path
import sys
import argparse

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
    parser.add_argument('--learning_rate', type=float, default=1e-3, 
                        help='Learning rate (default: 0.001)')
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

    # Load leadfield matrix
    leadfield_path = leadfield_dir / f"{sub_id}_{ses_id}_leadfield.npy"
    leadfield = np.load(leadfield_path)
    
    # Load leadfield info (optional, for reference)
    leadfield_info_path = leadfield_dir / f"{sub_id}_{ses_id}_leadfield_info.npy"
    leadfield_info = np.load(leadfield_info_path, allow_pickle=True).item()
    
    print(f"Leadfield matrix shape: {leadfield.shape}")
    
    # Load cleaned EEG epochs
    epochs_path = derivatives_dir / sub_id / ses_id / "eeg" / f"{sub_id}_{ses_id}_task-{task_id}_desc-epochs_eeg.set"
    epochs = read_epochs_eeglab(epochs_path)
    
    # Get EEG data and convert to torch tensor
    eeg_data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    
    # Take the specified number of epochs
    n_epochs_to_use = min(args.n_epochs_to_use, eeg_data.shape[0])
    eeg_subset = eeg_data[:n_epochs_to_use, :, :]
    
    # Reshape for model input: (time_points, channels)
    # Concatenate epochs along the time dimension
    eeg_data_reshaped = eeg_subset.reshape(-1, eeg_subset.shape[1]).T  # Now (n_channels, n_timepoints)
    eeg_tensor = torch.tensor(eeg_data_reshaped.T, dtype=torch.float32, device=device)  # Convert to (n_timepoints, n_channels)
    
    print(f"EEG data shape: {eeg_tensor.shape}")
    
    # Define model parameters
    y_dim = eeg_tensor.shape[1]  # Number of EEG channels
    x_dim = args.x_dim  # Dimension of latent state
    
    print(f"Using latent state dimension: {x_dim}")
    print(f"Observation dimension: {y_dim}")
    
    # Downsample the leadfield to match the state dimension
    # This is a simple approach - in practice, you might want a more sophisticated method
    # such as ROI-based averaging or PCA
    n_sources = leadfield.shape[1]
    source_indices = np.linspace(0, n_sources-1, x_dim, dtype=int)
    C_downsampled = leadfield[:, source_indices]
    
    print(f"Downsampled leadfield (C) shape: {C_downsampled.shape}")
    
    # Create validation set (20% of data)
    total_timepoints = eeg_tensor.shape[0]
    val_size = int(0.2 * total_timepoints)
    train_size = total_timepoints - val_size
    
    train_data = eeg_tensor[:train_size]
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
        dt=1/epochs.info['sfreq'],  # Use actual sampling frequency
        eps=1e-4
    )
    
    # Move model to device
    model = model.to(device)
    
    # Set training parameters
    n_epochs = args.training_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    
    # Train model
    print(f"Training model for {n_epochs} epochs with batch size {batch_size}...")
    history = model.fit(
        y=train_data,
        n_epochs=n_epochs,
        batch_size=batch_size,
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
    with torch.no_grad():
        output = model.forward(val_data)
    
    # Extract states and predictions
    latent_states = output['latent_states']  # (time, state_dim)
    predicted_obs = output['predicted_observations']  # (time, obs_dim)
    control_inputs = output['control_inputs']  # (time, input_dim)
    
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
    
    # Plot a subset of observed vs predicted EEG channels
    channels_to_plot = min(6, y_dim)  # Plot up to 6 channels
    plt.figure(figsize=(15, 10))
    for i in range(channels_to_plot):
        plt.subplot(3, 2, i+1)
        plt.plot(val_data[:, i].cpu().numpy(), label='Observed', alpha=0.7)
        plt.plot(predicted_obs[:, i].cpu().numpy(), label='Predicted', alpha=0.7)
        plt.title(f'Channel {i+1}')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"{sub_id}_predictions.png")
    plt.close()
    
    # Get learned matrices
    A = model.A.detach().cpu().numpy()
    B = model.B.detach().cpu().numpy()
    Q = model.Q.detach().cpu().numpy()
    R = model.R.detach().cpu().numpy()
    P = model.P.detach().cpu().numpy()
    
    # Print matrix information
    print("\nLearned model parameters:")
    print(f"A shape: {A.shape}")
    print(f"B shape: {B.shape}")
    print(f"Q shape: {Q.shape}")
    print(f"R shape: {R.shape}")
    print(f"P shape: {P.shape}")
    
    # Save model
    model_save_path = output_dir / f"{sub_id}_coupled_state_space_model.pt"
    model.save(str(model_save_path))
    print(f"Model saved to {model_save_path}")
    
    # Also save the learned matrices
    matrices_save_path = output_dir / f"{sub_id}_learned_matrices.npz"
    np.savez(
        matrices_save_path,
        A=A,
        B=B,
        C=model.C.detach().cpu().numpy(),
        Q=Q,
        R=R,
        P=P
    )
    print(f"Learned matrices saved to {matrices_save_path}")
    
    return model

if __name__ == "__main__":
    model = main() 