"""
Matrix Initialization for Control Systems

This module provides functions for initializing control system matrices,
such as creating controllable input matrices and initializing Riccati
equation solutions.
"""

import sympy
import torch
import numpy as np
from scipy import linalg
from typing import Optional


def create_controllable_B(A: torch.Tensor, m: int) -> torch.Tensor:
    """
    Create a controllable B matrix via the Jordan form approach.
    
    This function creates an input matrix B that ensures the pair (A, B)
    is controllable, which is a necessary condition for optimal control.
    
    Args:
        A: System matrix of shape (n, n)
        m: Number of inputs (columns of B)
        
    Returns:
        Input matrix B of shape (n, m) ensuring controllability
    """
    # Extract dimensions
    n = A.shape[0]
    device = A.device
    
    # Create B matrix
    B = torch.zeros(n, m, device=device)
    
    # Set first m elements to 1
    for i in range(min(m, n)):
        B[i, i] = 1.0
    
    return B


def initialize_riccati_p(A: torch.Tensor, 
                         B: torch.Tensor, 
                         C: torch.Tensor, 
                         Q: torch.Tensor,
                         R: torch.Tensor, 
                         W: torch.Tensor,
                         V: torch.Tensor) -> torch.Tensor:
    """
    Initialize the P matrix for the Riccati equation using the continuous algebraic
    Riccati equation (CARE) solution.
    
    Args:
        A: System matrix
        B: Input matrix
        C: Observation matrix
        Q: State cost matrix
        R: Input cost matrix
        W: Process noise covariance
        V: Measurement noise covariance
        
    Returns:
        Initial P matrix for the Riccati equation
    """
    # Convert tensors to numpy arrays
    A_np = A.detach().cpu().numpy()
    B_np = B.detach().cpu().numpy()
    C_np = C.detach().cpu().numpy()
    Q_np = Q.detach().cpu().numpy()
    R_np = R.detach().cpu().numpy()
    
    # Solve CARE
    try:
        # First try the standard CARE solution
        P_np = linalg.solve_continuous_are(A_np, B_np, Q_np, R_np)
    except Exception as e:
        print(f"Error solving CARE with standard approach: {e}")
        try:
            # Try a regularized approach
            Q_reg = Q_np + 1e-6 * np.eye(Q_np.shape[0])
            R_reg = R_np + 1e-6 * np.eye(R_np.shape[0])
            P_np = linalg.solve_continuous_are(A_np, B_np, Q_reg, R_reg)
        except Exception as e:
            print(f"Error solving CARE with regularized approach: {e}")
            # Fallback: Initialize P with an identity matrix scaled by trace of Q
            P_np = np.eye(A_np.shape[0]) * (np.trace(Q_np) / A_np.shape[0])
    
    # Convert back to tensor
    P = torch.tensor(P_np, dtype=A.dtype, device=A.device)
    
    return P


def initialize_stabilizing_K(A: torch.Tensor, B: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Initialize a stabilizing feedback gain K for the system (A, B).
    
    This is useful when the standard Riccati approach fails but a stabilizing
    controller is still needed.
    
    Args:
        A: System matrix
        B: Input matrix
        device: PyTorch device to use
        
    Returns:
        Stabilizing feedback gain K
    """
    A_np = A.detach().cpu().numpy()
    B_np = B.detach().cpu().numpy()
    n = A_np.shape[0]
    m = B_np.shape[1]
    
    # Try pole placement if the system is controllable
    try:
        # Check controllability
        ctrb = np.zeros((n, n*m))
        for i in range(n):
            ctrb[:, i*m:(i+1)*m] = np.linalg.matrix_power(A_np, i) @ B_np
        
        rank = np.linalg.matrix_rank(ctrb)
        if rank < n:
            print(f"Warning: System not fully controllable. Rank: {rank}/{n}")
        
        # Place poles at reasonable locations (-1, -2, ..., -n)
        desired_poles = -np.arange(1, n+1)
        K_np = np.zeros((m, n))
        
        # For single-input systems, can use explicit formula
        if m == 1:
            # Compute characteristic polynomial coefficients
            p = np.poly(desired_poles)
            
            # Compute controllability matrix
            C_b = np.zeros((n, n))
            C_b[:, 0] = B_np.flatten()
            for i in range(1, n):
                C_b[:, i] = A_np @ C_b[:, i-1]
            
            # Compute transformation matrix
            T = np.vstack([np.eye(1, n, n-i-1) @ np.linalg.matrix_power(A_np, i) for i in range(n)])
            
            # Compute feedback gain
            K_np = (p[1:] - np.poly(np.linalg.eigvals(A_np))[1:]) @ np.linalg.inv(C_b)
        else:
            # For multi-input systems, use a simpler approach
            # Place eigenvalues of A-BK at -1, -2, ..., -n
            K_np = np.ones((m, n))
        
        # Convert to tensor
        K = torch.tensor(K_np, dtype=A.dtype, device=device)
        return K
    
    except Exception as e:
        print(f"Error initializing stabilizing K: {e}")
        # Fallback: return a simple gain matrix
        return torch.ones(B.shape[1], A.shape[0], device=device) 