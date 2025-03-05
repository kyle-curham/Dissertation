"""
Riccati Equation Solvers

This module provides functions for solving continuous algebraic Riccati equations (CARE)
and associated control problems. It includes implementations of numerical methods
for solving matrix Riccati equations, such as the Smith iteration and implicit Euler methods.
"""

import numpy as np
import torch
from scipy import linalg
from typing import Tuple, Optional


def compute_optimal_K(A: torch.Tensor, B: torch.Tensor, Q: torch.Tensor, R: torch.Tensor, device: torch.device) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], bool]:
    """
    Compute the optimal feedback gain K via the continuous algebraic Riccati equation (CARE).
    
    Args:
        A: System matrix
        B: Input matrix
        Q: State cost matrix
        R: Input cost matrix
        device: PyTorch device to use
        
    Returns:
        Tuple containing:
        - Optimal feedback gain K (or None if computation fails)
        - Solution P of CARE (or None if computation fails)
        - Boolean indicating closed-loop stability
    """
    A_np = A.detach().cpu().numpy()
    B_np = B.detach().cpu().numpy()
    Q_np = Q.detach().cpu().numpy()
    R_np = R.detach().cpu().numpy()

    try:
        P = linalg.solve_continuous_are(A_np, B_np, Q_np, R_np)
        R_inv = np.linalg.inv(R_np)
        K = R_inv @ B_np.T @ P
        closed_loop_A = A_np - B_np @ K
        eigenvals = np.linalg.eigvals(closed_loop_A)
        is_stable = np.all(eigenvals.real < 0)

        K_torch = torch.FloatTensor(K).to(device)
        P_torch = torch.FloatTensor(P).to(device)
        return K_torch, P_torch, is_stable
    except Exception as e:
        print(f"Error in compute_optimal_K: {e}")
        return None, None, False


def implicit_euler_step(A: torch.Tensor, Q: torch.Tensor, X_current: torch.Tensor, h: float, device: torch.device, tol: float = 1e-9, max_iter: int = 100) -> torch.Tensor:
    """
    Perform an implicit Euler update for the matrix Riccati (Lyapunov) equation:
    X_dot = A.T @ X + X @ A + Q
    
    Args:
        A: System matrix
        Q: State cost matrix
        X_current: Current state of the Riccati equation
        h: Time step
        device: PyTorch device to use
        tol: Tolerance for convergence (default: 1e-9)
        max_iter: Maximum number of iterations (default: 100)
        
    Returns:
        Updated state of the Riccati equation
    """
    RHS = X_current + h * Q
    X_new = X_current.clone()
    for _ in range(max_iter):
        X_prev = X_new
        X_new = RHS + h * (A.T @ X_new + X_new @ A)
        if torch.norm(X_new - X_prev) / (torch.norm(X_prev) + 1e-4) < tol:
            break
    return 0.5 * (X_new + X_new.T)


def build_augmented_system(A: torch.Tensor, B: torch.Tensor, Q: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    Build the augmented system matrix from system matrices A, B, Q, R.
    
    This function constructs the Hamiltonian matrix for the linear-quadratic regulator problem,
    which is used in the Riccati equation solution.
    
    Args:
        A: System matrix
        B: Input matrix
        Q: State cost matrix
        R: Input cost matrix
        
    Returns:
        Augmented system matrix (Hamiltonian matrix)
    """
    # Compute R inverse with added stability
    eps = 1e-6 * torch.eye(R.shape[0], device=R.device)
    R_with_eps = R + eps
    R_inv = torch.inverse(R_with_eps)
    
    # Create the Hamiltonian matrix
    BR_inv_BT = B @ R_inv @ B.T
    A_top = torch.cat([A, -BR_inv_BT], dim=1)
    A_bottom = torch.cat([-Q, -A.T], dim=1)
    return torch.cat([A_top, A_bottom], dim=0)


def solve_care_smith(A: torch.Tensor, B: torch.Tensor, Q: torch.Tensor, R: torch.Tensor, W: torch.Tensor, V: torch.Tensor, device: torch.device, max_iter: int = 50000, tol: float = 1e-8, X_init: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Solve the continuous-time algebraic Riccati equation using Smith iteration.
    
    The Smith method is an iterative approach to solving Riccati equations, which uses
    a continuous-time system to evolve the solution to steady state.
    
    Args:
        A: System matrix
        B: Input matrix
        Q: State cost matrix
        R: Input cost matrix
        W: Process noise covariance
        V: Measurement noise covariance
        device: PyTorch device to use
        max_iter: Maximum number of iterations (default: 50000)
        tol: Tolerance for convergence (default: 1e-8)
        X_init: Initial guess for the solution (default: None)
        
    Returns:
        Solution of the continuous-time algebraic Riccati equation
    """
    # Initialize with a reasonable starting point if not provided
    if X_init is None:
        X = torch.block_diag(Q, W, W).to(device)
    else:
        X = X_init.clone()
    
    # Time step for implicit Euler method
    dt = (1.0/1024.0)
    
    # Iteratively solve the Riccati equation
    for i in range(max_iter):
        # Build the augmented system matrix
        A_aug = build_augmented_system(A, B, Q, R)
        
        # Perform an implicit Euler step
        X_new = implicit_euler_step(A_aug, Q, X, dt, device, tol=tol)
        
        # Check for convergence
        rel_error = torch.norm(X_new - X) / (torch.norm(X) + 1e-4)
        if rel_error < tol:
            return X_new
        
        # Update for next iteration
        X = X_new.clone()
    
    # Return the best approximation if max_iter is reached
    return X 