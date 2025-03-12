#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for the chunking strategy in CoupledStateSpaceVI.
This script creates a synthetic dataset and tests training with different chunk sizes.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import time

# Add the project root to the path so we can import from eeg_processing
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from eeg_processing.models.coupled_state_space_vi import CoupledStateSpaceVI

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description='Test chunking strategy for CoupledStateSpaceVI')
    parser.add_argument('--x_dim', type=int, default=8, help='Latent state dimension (default: 8)')
    parser.add_argument('--y_dim', type=int, default=32, help='Observation dimension (default: 32)')
    parser.add_argument('--seq_len', type=int, default=1024, help='Sequence length (default: 1024)')
    parser.add_argument('--training_epochs', type=int, default=5, help='Number of training epochs (default: 5)')
    parser.add_argument('--chunk_sizes', type=str, default='32,64,128,256', help='Comma-separated list of chunk sizes to test')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    
    return parser.parse_args()

def generate_synthetic_data(seq_len, x_dim, y_dim):
    """Generate synthetic data for testing."""
    # Generate a random linear dynamical system
    A = 0.95 * torch.randn(x_dim, x_dim)
    # Make A stable by scaling its eigenvalues
    u, s, v = torch.svd(A)
    A = u @ torch.diag(0.95 * s / torch.max(s)) @ v.t()
    
    # Generate random observation matrix
    C = torch.randn(y_dim, x_dim)
    
    # Generate latent states using the dynamical system
    x = torch.zeros(seq_len, x_dim)
    x[0] = torch.randn(x_dim)
    for t in range(1, seq_len):
        x[t] = x[t-1] @ A.t() + 0.1 * torch.randn(x_dim)
    
    # Generate observations
    y = x @ C.t() + 0.1 * torch.randn(seq_len, y_dim)
    
    return y, C.numpy()

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Set device
    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Parse chunk sizes
    chunk_sizes = [int(cs) for cs in args.chunk_sizes.split(',')]
    print(f"Testing chunk sizes: {chunk_sizes}")
    
    # Generate synthetic data
    print(f"Generating synthetic data with sequence length {args.seq_len}...")
    data, C = generate_synthetic_data(args.seq_len, args.x_dim, args.y_dim)
    data = data.to(device)
    
    # Split into train and validation
    train_size = int(0.8 * args.seq_len)
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    
    # Results storage
    results = {
        'chunk_sizes': chunk_sizes,
        'training_times': [],
        'final_elbos': [],
        'val_elbos': []
    }
    
    # Output directory
    output_dir = project_root / "eeg_processing" / "results" / "chunking_tests"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test each chunk size
    for chunk_size in chunk_sizes:
        print(f"\n--- Testing chunk size: {chunk_size} ---")
        
        # Create model
        model = CoupledStateSpaceVI(
            x_dim=args.x_dim,
            y_dim=args.y_dim,
            C=C,
            u_dim=args.x_dim,
            beta=0.1,
            prior_std=1.0,
            dt=1.0/30.0,  # Assuming 30 Hz
            eps=1e-4,
            device=device
        )
        
        # Train model
        start_time = time.time()
        history = model.fit(
            y=train_data,
            n_epochs=args.training_epochs,
            batch_size=64,
            learning_rate=1e-3,
            validation_data=val_data,
            verbose=True,
            chunk_size=chunk_size
        )
        end_time = time.time()
        
        # Record results
        training_time = end_time - start_time
        results['training_times'].append(training_time)
        results['final_elbos'].append(history['elbo'][-1])
        if history['val_elbo'] is not None:
            results['val_elbos'].append(history['val_elbo'][-1])
        
        print(f"Chunk size {chunk_size}: Training time = {training_time:.2f}s, Final ELBO = {history['elbo'][-1]:.4f}")
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history['elbo'], label=f'Training ELBO')
        if history['val_elbo'] is not None:
            plt.plot(history['val_elbo'], label=f'Validation ELBO')
        plt.xlabel('Epoch')
        plt.ylabel('ELBO')
        plt.title(f'Training History for Chunk Size {chunk_size}')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / f"chunk_size_{chunk_size}_history.png")
        plt.close()
    
    # Create comparison plots
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(range(len(chunk_sizes)), results['training_times'], tick_label=[str(cs) for cs in chunk_sizes])
    plt.title('Training Time Comparison')
    plt.xlabel('Chunk Size')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(chunk_sizes)), results['final_elbos'], tick_label=[str(cs) for cs in chunk_sizes])
    plt.title('Final ELBO Comparison')
    plt.xlabel('Chunk Size')
    plt.ylabel('ELBO')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / "chunking_comparison.png")
    plt.close()
    
    print("\nResults summary:")
    for i, chunk_size in enumerate(chunk_sizes):
        print(f"Chunk size {chunk_size}: Time = {results['training_times'][i]:.2f}s, ELBO = {results['final_elbos'][i]:.4f}")
    
    # Save results to file
    with open(output_dir / "chunking_results.txt", "w") as f:
        f.write("Chunk Size Tests Results\n")
        f.write("=======================\n\n")
        f.write(f"Sequence Length: {args.seq_len}\n")
        f.write(f"Number of Epochs: {args.training_epochs}\n\n")
        
        f.write("Chunk Size | Training Time (s) | Final ELBO\n")
        f.write("-----------------------------------------\n")
        for i, chunk_size in enumerate(chunk_sizes):
            f.write(f"{chunk_size:^10} | {results['training_times'][i]:^16.2f} | {results['final_elbos'][i]:^10.4f}\n")

if __name__ == "__main__":
    main() 