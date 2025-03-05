"""
Numerical Differentiation Utilities

This module provides functions for numerical differentiation of time series data,
which is useful for estimating derivatives in state-space models and for 
converting between discrete and continuous-time representations.
"""

import numpy as np
import torch
from typing import Tuple, Union, Optional, List, Callable


def central_difference(x: Union[np.ndarray, torch.Tensor], 
                       dt: float = 1.0,
                       axis: int = 0) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute first derivatives using central differences.
    
    Args:
        x: Input data array of shape (..., time_steps, ...)
        dt: Time step size (default: 1.0)
        axis: Axis along which to compute derivatives (default: 0)
        
    Returns:
        Derivatives of x with respect to time with same shape as x
    """
    if isinstance(x, np.ndarray):
        # Create output array
        dx = np.zeros_like(x)
        
        # Get slices for indexing
        slices = tuple(slice(None) if i != axis else slice(2, None) for i in range(x.ndim))
        slices_prev = tuple(slice(None) if i != axis else slice(0, -2) for i in range(x.ndim))
        
        # Central difference for interior points
        dx[slices] = (x[tuple(slice(None) if i != axis else slice(2, None) for i in range(x.ndim))] - 
                      x[tuple(slice(None) if i != axis else slice(0, -2) for i in range(x.ndim))]) / (2 * dt)
        
        # Forward difference for first point
        first_idx = tuple(slice(None) if i != axis else 0 for i in range(x.ndim))
        second_idx = tuple(slice(None) if i != axis else 1 for i in range(x.ndim))
        dx[first_idx] = (x[second_idx] - x[first_idx]) / dt
        
        # Backward difference for last point
        last_idx = tuple(slice(None) if i != axis else -1 for i in range(x.ndim))
        second_last_idx = tuple(slice(None) if i != axis else -2 for i in range(x.ndim))
        dx[last_idx] = (x[last_idx] - x[second_last_idx]) / dt
        
    else:  # torch.Tensor
        # Create output tensor
        dx = torch.zeros_like(x)
        
        # Get tensor shape
        shape = x.shape
        time_steps = shape[axis]
        
        # Create indexing tensors
        idx_prev = [slice(None)] * x.dim()
        idx_next = [slice(None)] * x.dim()
        idx_curr = [slice(None)] * x.dim()
        
        # Central difference for interior points
        for t in range(1, time_steps - 1):
            idx_prev[axis] = t - 1
            idx_next[axis] = t + 1
            idx_curr[axis] = t
            dx[tuple(idx_curr)] = (x[tuple(idx_next)] - x[tuple(idx_prev)]) / (2 * dt)
        
        # Forward difference for first point
        idx_curr[axis] = 0
        idx_next[axis] = 1
        dx[tuple(idx_curr)] = (x[tuple(idx_next)] - x[tuple(idx_curr)]) / dt
        
        # Backward difference for last point
        idx_curr[axis] = time_steps - 1
        idx_prev[axis] = time_steps - 2
        dx[tuple(idx_curr)] = (x[tuple(idx_curr)] - x[tuple(idx_prev)]) / dt
    
    return dx


def savitzky_golay_derivative(x: Union[np.ndarray, torch.Tensor],
                              window_length: int = 5,
                              polyorder: int = 2,
                              deriv: int = 1,
                              dt: float = 1.0,
                              axis: int = 0) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute derivatives using Savitzky-Golay filter for smoother results.
    
    Args:
        x: Input data array of shape (..., time_steps, ...)
        window_length: Window length for the filter (default: 5)
        polyorder: Polynomial order for the filter (default: 2)
        deriv: Derivative order (default: 1)
        dt: Time step size (default: 1.0)
        axis: Axis along which to compute derivatives (default: 0)
        
    Returns:
        Derivatives of x with respect to time with same shape as x
    """
    import scipy.signal as signal
    
    # Check if input is PyTorch tensor
    is_torch = isinstance(x, torch.Tensor)
    device = x.device if is_torch else None
    
    # Convert to numpy if needed
    if is_torch:
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    
    # Apply Savitzky-Golay filter
    dx_np = signal.savgol_filter(
        x_np, 
        window_length=window_length, 
        polyorder=polyorder, 
        deriv=deriv, 
        delta=dt,
        axis=axis
    )
    
    # Convert back to PyTorch if needed
    if is_torch:
        dx = torch.tensor(dx_np, dtype=x.dtype, device=device)
    else:
        dx = dx_np
    
    return dx


def discrete_to_continuous(A_discrete: Union[np.ndarray, torch.Tensor],
                           dt: float) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert a discrete-time state transition matrix to its continuous-time equivalent.
    
    Uses the matrix logarithm: A_continuous = log(A_discrete) / dt
    
    Args:
        A_discrete: Discrete-time state transition matrix
        dt: Time step size
        
    Returns:
        Continuous-time state transition matrix
    """
    if isinstance(A_discrete, np.ndarray):
        from scipy.linalg import logm
        return logm(A_discrete) / dt
    else:  # torch.Tensor
        # Eigendecomposition approach for numerical stability
        eigvals, eigvecs = torch.linalg.eig(A_discrete)
        
        # Take log of eigenvalues
        log_eigvals = torch.log(eigvals)
        
        # Reconstruct matrix using log eigenvalues
        A_continuous = torch.real(
            eigvecs @ torch.diag(log_eigvals) @ torch.inverse(eigvecs)
        ) / dt
        
        return A_continuous


def continuous_to_discrete(A_continuous: Union[np.ndarray, torch.Tensor],
                           dt: float) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert a continuous-time state transition matrix to its discrete-time equivalent.
    
    Uses the matrix exponential: A_discrete = exp(A_continuous * dt)
    
    Args:
        A_continuous: Continuous-time state transition matrix
        dt: Time step size
        
    Returns:
        Discrete-time state transition matrix
    """
    if isinstance(A_continuous, np.ndarray):
        from scipy.linalg import expm
        return expm(A_continuous * dt)
    else:  # torch.Tensor
        # Eigendecomposition approach for numerical stability
        eigvals, eigvecs = torch.linalg.eig(A_continuous)
        
        # Multiply eigenvalues by dt and exponentiate
        exp_eigvals = torch.exp(eigvals * dt)
        
        # Reconstruct matrix using exponentiated eigenvalues
        A_discrete = torch.real(
            eigvecs @ torch.diag(exp_eigvals) @ torch.inverse(eigvecs)
        )
        
        return A_discrete


def estimate_derivatives(x: Union[np.ndarray, torch.Tensor],
                         dt: float = 1.0,
                         method: str = "savitzky_golay",
                         **kwargs) -> Union[np.ndarray, torch.Tensor]:
    """
    Estimate derivatives of time series data.
    
    Args:
        x: Input data array of shape (..., time_steps, ...)
        dt: Time step size (default: 1.0)
        method: Method to use for differentiation (default: "savitzky_golay")
               Options: "central_difference", "savitzky_golay"
        **kwargs: Additional arguments to pass to the differentiation method
        
    Returns:
        Derivatives of x with respect to time with same shape as x
    """
    if method == "central_difference":
        return central_difference(x, dt=dt, **kwargs)
    elif method == "savitzky_golay":
        return savitzky_golay_derivative(x, dt=dt, **kwargs)
    else:
        raise ValueError(f"Unknown differentiation method: {method}") 