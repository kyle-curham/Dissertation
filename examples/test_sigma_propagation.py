#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for Sigma propagation in the CoupledStateSpaceVI model.

This script tests the propagation of the covariance matrix Sigma through the symplectic integrator
and ensures that it maintains its non-diagonal structure.
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
from eeg_processing.models.control.riccati_solver import RiccatiSolver

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description='Test Sigma propagation in CoupledStateSpaceVI')
    parser.add_argument('--x_dim', type=int, default=8, help='Latent state dimension (default: 8)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    
    return parser.parse_args()

def get_random_linear_system(x_dim, stable=True):
    """Generate a random linear dynamical system."""
    A = torch.randn(x_dim, x_dim)
    
    # Make A stable if requested
    if stable:
        u, s, v = torch.svd(A)
        A = u @ torch.diag(0.95 * s / torch.max(s)) @ v.T
    
    B = torch.randn(x_dim, x_dim)
    Q = torch.randn(x_dim, x_dim)
    Q = torch.matmul(Q, Q.T) + torch.eye(x_dim)  # Make Q positive definite
    R = torch.randn(x_dim, x_dim)
    R = torch.matmul(R, R.T) + torch.eye(x_dim)  # Make R positive definite
    
    return A, B, Q, R

def test_symplectic_integration(device, x_dim=8, num_steps=50):
    """Test the symplectic integrator directly to see if it preserves structure."""
    riccati_solver = RiccatiSolver(device=device)
    
    # Generate random system matrices
    A, B, Q, R = get_random_linear_system(x_dim)
    A, B, Q, R = A.to(device), B.to(device), Q.to(device), R.to(device)
    
    # Build augmented system
    A_aug = riccati_solver.build_augmented_system(A, B, Q, R)
    
    # Initialize Sigma with a non-diagonal structure
    Sigma = torch.randn(x_dim, x_dim).to(device)
    Sigma = torch.matmul(Sigma, Sigma.T)  # Ensure it's symmetric
    Sigma = Sigma / torch.norm(Sigma)     # Normalize for numerical stability
    
    print("\n===== DIRECT SYMPLECTIC INTEGRATION TEST =====")
    print(f"Initial Sigma diagonal ratio: {torch.norm(torch.diag(Sigma))/torch.norm(Sigma):.4f}")
    
    # Track Sigma's properties over time
    diagonal_ratios = []
    traces = []
    min_eigenvalues = []
    
    # Run symplectic integration for multiple steps
    for i in range(num_steps):
        # Calculate diagnostics
        diag_ratio = torch.norm(torch.diag(Sigma)) / torch.norm(Sigma)
        diagonal_ratios.append(diag_ratio.item())
        traces.append(torch.trace(Sigma).item())
        min_eigenvalues.append(torch.min(torch.linalg.eigvalsh(Sigma)).item())
        
        # Propagate Sigma
        Sigma = riccati_solver.symplectic_integrator_step(A_aug, Sigma, dt=0.01)
        
        # Print periodic diagnostics
        if i % 10 == 0 or i == num_steps - 1:
            print(f"Step {i}: Diagonal ratio = {diag_ratio:.4f}, "
                  f"Trace = {traces[-1]:.4f}, Min eigenvalue = {min_eigenvalues[-1]:.4f}")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(diagonal_ratios)
    plt.title('Diagonal Ratio (higher = more diagonal)')
    plt.ylabel('Ratio')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(traces)
    plt.title('Trace of Sigma')
    plt.ylabel('Trace')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(min_eigenvalues)
    plt.title('Minimum Eigenvalue')
    plt.xlabel('Integration Step')
    plt.ylabel('Min Eigenvalue')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Create output directory
    output_dir = project_root / "eeg_processing" / "results" / "sigma_tests"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(output_dir / "symplectic_integration_test.png")
    plt.close()
    
    # Visualize final Sigma matrix
    plt.figure(figsize=(8, 8))
    plt.imshow(Sigma.cpu().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title(f'Final Sigma Matrix (Diagonal Ratio: {diagonal_ratios[-1]:.4f})')
    plt.savefig(output_dir / "final_sigma_matrix.png")
    plt.close()

def test_parameter_update_effect(device, x_dim=8, num_steps=10):
    """Test how parameter updates affect Sigma propagation."""
    print("\n===== PARAMETER UPDATE EFFECT TEST =====")
    
    # Generate synthetic data
    seq_len = 100
    y_dim = x_dim
    data = torch.randn(seq_len, y_dim).to(device) * 0.1
    
    # Create initial system matrices
    A, B, Q, R = get_random_linear_system(x_dim)
    C = torch.eye(y_dim, x_dim).numpy()  # Simple observation matrix
    
    # Initialize model with these matrices
    model = CoupledStateSpaceVI(
        x_dim=x_dim,
        y_dim=y_dim,
        C=C,
        u_dim=x_dim,
        beta=0.1,
        prior_std=1.0,
        dt=0.01,
        device=device
    )
    
    # Force our A, B, Q, R values
    with torch.no_grad():
        model.A.copy_(A.to(device))
        model.B.copy_(B.to(device))
        model.Q.copy_(Q.to(device))
        model.R.copy_(R.to(device))
    
    # Initialize Sigma with a non-diagonal structure
    Sigma = torch.randn(x_dim, x_dim).to(device)
    Sigma = 0.5 * (Sigma + Sigma.T)  # Ensure it's symmetric
    Sigma = Sigma / torch.norm(Sigma) + torch.eye(x_dim, device=device) * 0.1  # Add some diagonal
    
    print(f"Initial Sigma diagonal ratio: {torch.norm(torch.diag(Sigma))/torch.norm(Sigma):.4f}")
    
    # Build initial augmented system
    A_aug_initial = model._riccati_solver.build_augmented_system(model.A, model.B, model.Q, model.R)
    
    # Propagate with initial parameters
    print("\nPropagating with initial parameters...")
    for i in range(5):
        Sigma = model._riccati_solver.symplectic_integrator_step(A_aug_initial, Sigma, model.dt)
        diag_ratio = torch.norm(torch.diag(Sigma)) / torch.norm(Sigma)
        print(f"Step {i}: Diagonal ratio = {diag_ratio:.4f}, Trace = {torch.trace(Sigma).item():.4f}")
    
    # Create optimizer and update parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Make a significant parameter change
    print("\nUpdating parameters...")
    with torch.enable_grad():
        # Create a loss that will change parameters significantly
        loss = torch.sum(model.A) + torch.sum(model.B)
        loss.backward()
        optimizer.step()
    
    # Build new augmented system with UPDATED parameters
    A_aug_updated = model._riccati_solver.build_augmented_system(model.A, model.B, model.Q, model.R)
    
    # Check if parameters actually changed
    param_diff = torch.norm(A_aug_updated - A_aug_initial)
    print(f"Parameter change magnitude: {param_diff.item():.4f}")
    
    # Store Sigma before update propagation
    Sigma_before_update_propagation = Sigma.clone()
    
    # Propagate with updated parameters
    print("\nPropagating with UPDATED parameters...")
    for i in range(5):
        Sigma = model._riccati_solver.symplectic_integrator_step(A_aug_updated, Sigma, model.dt)
        diag_ratio = torch.norm(torch.diag(Sigma)) / torch.norm(Sigma)
        print(f"Step {i}: Diagonal ratio = {diag_ratio:.4f}, Trace = {torch.trace(Sigma).item():.4f}")
    
    # Compute difference between propagation with old vs new parameters
    diff_magnitude = torch.norm(Sigma - Sigma_before_update_propagation)
    print(f"\nDifference due to parameter update: {diff_magnitude.item():.4f}")
    
    # Create output directory
    output_dir = project_root / "eeg_processing" / "results" / "sigma_tests"
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize matrices
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(Sigma_before_update_propagation.cpu().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title('Sigma Before Parameter Update')
    
    plt.subplot(1, 3, 2)
    plt.imshow(Sigma.cpu().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title('Sigma After Parameter Update')
    
    plt.subplot(1, 3, 3)
    plt.imshow((Sigma - Sigma_before_update_propagation).cpu().numpy(), cmap='RdBu')
    plt.colorbar()
    plt.title('Difference Due to Parameter Update')
    
    plt.tight_layout()
    plt.savefig(output_dir / "parameter_update_effect.png")
    plt.close()
    
    if diff_magnitude.item() < 1e-4:
        print("WARNING: Parameter updates have very little effect on Sigma!")
    elif diff_magnitude.item() > 0.1:
        print("Parameter updates significantly change Sigma, as expected.")
    
    return diff_magnitude.item()

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Set device
    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Test symplectic integration directly
    test_symplectic_integration(device, x_dim=args.x_dim)
    
    # Test how parameter updates affect Sigma propagation
    diff_magnitude = test_parameter_update_effect(device, x_dim=args.x_dim)
    
    print(f"\nSigma propagation test completed! Parameter update effect: {diff_magnitude:.6f}")
    print("Check the results directory for plots.")

if __name__ == "__main__":
    main() 