"""
Numerical Stability Utilities

This module provides functions to ensure numerical stability in
matrix operations, particularly for covariance matrices and
Cholesky decompositions used in state-space models.
"""

import torch
import numpy as np
from typing import List, Tuple, Union, Optional


def enforce_cholesky_structure(*matrices: torch.Tensor) -> None:
    """
    Enforce lower triangular structure for Cholesky decomposition matrices
    without breaking the computation graph.
    """
    with torch.no_grad():
        for mat in matrices:
            # Create new tensor with lower triangular structure
            new_mat = torch.tril(mat.data)
            # Copy values without breaking gradient tracking
            mat.data = new_mat


def ensure_positive_definite(matrix: torch.Tensor, 
                             min_eigenvalue: float = 1e-6) -> torch.Tensor:
    """
    Ensure a matrix is positive definite by adding a small value to the diagonal if needed.
    
    Args:
        matrix: Input matrix to ensure is positive definite
        min_eigenvalue: Minimum eigenvalue to enforce (default: 1e-6)
        
    Returns:
        Positive definite matrix
    """
    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvalsh(matrix)
    min_eig = torch.min(eigenvalues)
    
    # If minimum eigenvalue is less than threshold, add to diagonal
    if min_eig < min_eigenvalue:
        eps = min_eigenvalue - min_eig
        matrix = matrix + eps * torch.eye(matrix.shape[0], 
                                          device=matrix.device)
    
    return matrix


def stabilize_covariance(cov: torch.Tensor, 
                         min_eigenvalue: float = 1e-6,
                         max_condition: float = 1e8) -> torch.Tensor:
    """
    Stabilize a covariance matrix by ensuring it's positive definite
    and has a reasonable condition number.
    
    Args:
        cov: Covariance matrix to stabilize
        min_eigenvalue: Minimum eigenvalue to enforce (default: 1e-6)
        max_condition: Maximum condition number to enforce (default: 1e8)
        
    Returns:
        Stabilized covariance matrix
    """
    # Make symmetric by averaging with transpose
    cov = 0.5 * (cov + cov.transpose(-1, -2))
    
    # Compute eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    
    # Enforce minimum eigenvalue
    eigenvalues = torch.clamp(eigenvalues, min=min_eigenvalue)
    
    # Enforce maximum condition number
    max_eig = torch.max(eigenvalues)
    min_allowed = max_eig / max_condition
    eigenvalues = torch.maximum(eigenvalues, 
                               torch.tensor(min_allowed, device=cov.device))
    
    # Reconstruct matrix
    stabilized_cov = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.transpose(-1, -2)
    
    return stabilized_cov


def safe_cholesky(matrix: torch.Tensor, 
                 min_eigenvalue: float = 1e-6) -> torch.Tensor:
    """
    Safely compute Cholesky decomposition by ensuring the matrix is positive definite.
    
    Args:
        matrix: Matrix to decompose
        min_eigenvalue: Minimum eigenvalue to enforce (default: 1e-6)
        
    Returns:
        Lower triangular Cholesky factor
    """
    # Ensure matrix is positive definite
    matrix_pd = ensure_positive_definite(matrix, min_eigenvalue)
    
    # Compute Cholesky decomposition
    try:
        L = torch.linalg.cholesky(matrix_pd)
        return L
    except RuntimeError:
        # If Cholesky still fails, use more aggressive stabilization
        matrix_pd = stabilize_covariance(matrix_pd, 
                                        min_eigenvalue=min_eigenvalue*10)
        return torch.linalg.cholesky(matrix_pd) 