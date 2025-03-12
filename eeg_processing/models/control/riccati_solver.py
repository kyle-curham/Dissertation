"""
Riccati Equation Solvers

This module provides functions for solving continuous algebraic Riccati equations (CARE)
and associated control problems. It includes implementations of numerical methods
for solving matrix Riccati equations, such as the Smith iteration and implicit Euler methods.
"""

import numpy as np
import torch
from scipy import linalg
from typing import Tuple, Optional, Dict, List, Union
import logging

# Add a counter at the module level
_warning_counter = 0


def _log_warning(message: str) -> None:
    """
    Log a warning message using the Python logging module.
    
    Args:
        message: The warning message to log
    """
    logging.warning(message)


class RiccatiSolver:
    """
    Class for solving continuous-time algebraic Riccati equations using the Cayley transform
    method for numerical integration.
    """
    
    def __init__(self, device: torch.device = None):
        """
        Initialize the Riccati equation solver.
        
        Args:
            device: PyTorch device to use for computations
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Cache for matrix exponentials
        self.exp_cache = {}
    
    def _log_warning(self, message: str) -> None:
        """
        Log a warning message using the module-level _log_warning function.
        
        Args:
            message: The warning message to log
        """
        _log_warning(message)
    
    def _get_matrix_exp(self, matrix: torch.Tensor, h: float) -> torch.Tensor:
        matrix_norm = torch.norm(matrix, p=1).item()
        if matrix_norm * h > 10:
            scaling_factor = 2 ** (int(np.ceil(np.log2(matrix_norm * h / 10))))
            scaled_h = h / scaling_factor
            exp_scaled = torch.matrix_exp(scaled_h * matrix)
            result = exp_scaled
            for _ in range(int(np.log2(scaling_factor))):
                result = torch.matmul(result, result)
            return result
        else:
            return torch.matrix_exp(h * matrix)
    
    
    def build_augmented_system(self, A: torch.Tensor, B: torch.Tensor, Q: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
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
        # Reset the matrix exponential cache when system parameters change
        self.exp_cache = {}
        
        # Compute R inverse with added stability
        eps = 1e-6 * torch.eye(R.shape[0], device=R.device)
        R_with_eps = R + eps
        R_inv = torch.inverse(R_with_eps)
        
        # Create the Hamiltonian matrix
        BR_inv_BT = B @ R_inv @ B.T
        
        # Ensure Q and BR_inv_BT are symmetric as they should be mathematically
        Q_sym = 0.5 * (Q + Q.T)
        BR_inv_BT_sym = 0.5 * (BR_inv_BT + BR_inv_BT.T)
        
        A_top = torch.cat([A, -BR_inv_BT_sym], dim=1)
        A_bottom = torch.cat([-Q_sym, -A.T], dim=1)
        augmented_A = torch.cat([A_top, A_bottom], dim=0)
        
        return augmented_A
    
    def step(self, S_current, h):
        """
        One symplectic step to update S and compute X_new.
        
        Args:
            S_current (2n × n): Current state [Z1; Z2]
            h: Time step size
            
        Returns:
            X_new (n × n): Updated Riccati solution
            S_new (2n × n): Updated state
        """
        n = self.n
        
        # Ensure consistent tensor type
        dtype = self.dtype if hasattr(self, 'dtype') else torch.float
        device = self.device if hasattr(self, 'device') else (S_current.device if hasattr(S_current, 'device') else None)
        
        # Convert S_current to the expected type if needed
        if S_current.dtype != dtype:
            S_current = S_current.to(dtype=dtype)
            
        # Create identity matrix with correct type
        I_2n = torch.eye(2 * n, device=device, dtype=dtype)
        
        # Make sure H has the correct type
        if hasattr(self, 'H') and self.H.dtype != dtype:
            self.H = self.H.to(dtype=dtype)
    
        # Implicit midpoint update
        A = I_2n + (h / 2) * self.H
        B = I_2n - (h / 2) * self.H
        
        # Solve the linear system
        try:
            S_new = torch.linalg.solve(A, B @ S_current)
        except RuntimeError as e:
            # Fallback to a more numerically stable approach
            self._log_warning(f"Standard linear solve failed: {str(e)}. Using pseudoinverse.")
            A_pinv = torch.linalg.pinv(A)
            S_new = A_pinv @ (B @ S_current)
    
        # Extract Z1 and Z2
        Z1_new = S_new[:n, :]
        Z2_new = S_new[n:, :]
    
        # Compute X_new with error handling
        try:
            X_new = Z2_new @ torch.inverse(Z1_new)
        except torch.linalg.LinAlgError:
            try:
                # Try with regularization
                Z1_reg = Z1_new + torch.eye(n, device=device, dtype=dtype) * 1e-6
                X_new = Z2_new @ torch.inverse(Z1_reg)
            except torch.linalg.LinAlgError:
                # Fallback to pseudoinverse
                X_new = Z2_new @ torch.pinverse(Z1_new)
    
        # Symmetrize
        X_new = 0.5 * (X_new + X_new.T)
        
        # # Ensure positive definiteness
        # eigvals, eigvecs = torch.linalg.eigh(X_new)
        # if torch.min(eigvals) < 0:
        #     eigvals = torch.clamp(eigvals, min=1e-6)
        #     X_new = eigvecs @ torch.diag(eigvals) @ eigvecs.T
    
        return X_new, S_new

    def integrate(self, X0, h, steps):
        """
        Integrate to get X_new over multiple steps.
        
        Args:
            X0 (n × n): Initial X
            h: Time step size
            steps: Number of steps
            
        Returns:
            X_new (n × n): Final Riccati solution
        """
        n = self.n
        
        # Ensure consistent tensor type
        dtype = self.dtype if hasattr(self, 'dtype') else torch.float
        device = self.device if hasattr(self, 'device') else (X0.device if hasattr(X0, 'device') else None)
        
        # Convert X0 to the expected type if needed
        if X0.dtype != dtype:
            X0 = X0.to(dtype=dtype)
            
        # Create identity matrix with correct type
        I_n = torch.eye(n, device=device, dtype=dtype)
        
        # Initialize S with proper type
        S = torch.cat([I_n, X0], dim=0)
        
        # Limit steps for safety
        max_safe_steps = 50000  # Adjust as needed
        steps = min(steps, max_safe_steps)
        
        # Use adaptive stepping for large step counts
        if steps > 1000:
            # Group into batches for efficiency
            batch_size = 100
            num_batches = steps // batch_size
            remaining = steps % batch_size
            
            for _ in range(num_batches):
                for _ in range(batch_size):
                    X_new, S = self.step(S, h)
                    # Update S for next iteration
                    S = torch.cat([S[:n, :], X_new @ S[:n, :]], dim=0)
            
            # Handle remaining steps
            for _ in range(remaining):
                X_new, S = self.step(S, h)
                S = torch.cat([S[:n, :], X_new @ S[:n, :]], dim=0)
        else:
            # Standard approach for reasonable step counts
            for _ in range(steps):
                X_new, S = self.step(S, h)
                S = torch.cat([S[:n, :], X_new @ S[:n, :]], dim=0)
        
        return X_new

    