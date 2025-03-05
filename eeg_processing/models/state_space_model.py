"""
State Space Model Implementation

This module provides implementations of state space models for EEG data analysis.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Optional, Union, Any
from ..utils.numerical_stability import (
    enforce_cholesky_structure,
    ensure_positive_definite,
    stabilize_covariance,
    safe_cholesky
)
from abc import ABC, abstractmethod


class BaseStateSpaceModel(nn.Module, ABC):
    """
    Base class for state-space models.
    
    This abstract base class defines the interface and common functionality
    for all state-space model implementations.
    """
    
    def __init__(self, 
                 x_dim: int, 
                 y_dim: int, 
                 u_dim: Optional[int] = None,
                 dt: float = 1.0,
                 device: Optional[torch.device] = None) -> None:
        """
        Initialize the base state-space model.
        
        Args:
            x_dim: Dimension of the state vector
            y_dim: Dimension of the observation vector
            u_dim: Dimension of the input/control vector (default: None)
            dt: Time step (default: 1.0)
            device: Torch device to use (default: None, uses CPU)
        """
        super().__init__()
        
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.u_dim = u_dim if u_dim is not None else 0
        self.dt = dt
        self.device = device if device is not None else torch.device('cpu')
        
        # Initialize model parameters
        self.initialize_parameters()
    
    @abstractmethod
    def initialize_parameters(self) -> None:
        """
        Initialize model parameters.
        
        This method should be implemented by subclasses to initialize
        the model parameters (A, B, C, W, V).
        """
        pass
    
    @abstractmethod
    def forward(self, 
                y: torch.Tensor, 
                u: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Forward pass of the model.
        
        Args:
            y: Observation tensor of shape (batch_size, sequence_length, y_dim)
            u: Input tensor of shape (batch_size, sequence_length, u_dim) (optional)
            
        Returns:
            Dictionary containing model outputs
        """
        pass
    
    @abstractmethod
    def predict(self, 
                x0: torch.Tensor, 
                u: Optional[torch.Tensor] = None, 
                steps: int = 1) -> torch.Tensor:
        """
        Predict future states given initial state and inputs.
        
        Args:
            x0: Initial state tensor of shape (batch_size, x_dim)
            u: Input tensor of shape (batch_size, steps, u_dim) (optional)
            steps: Number of steps to predict (default: 1)
            
        Returns:
            Predicted states of shape (batch_size, steps, x_dim)
        """
        pass
    
    def predict_observations(self, 
                           x0: torch.Tensor, 
                           u: Optional[torch.Tensor] = None, 
                           steps: int = 1) -> torch.Tensor:
        """
        Predict future observations given initial state and inputs.
        
        Args:
            x0: Initial state tensor of shape (batch_size, x_dim)
            u: Input tensor of shape (batch_size, steps, u_dim) (optional)
            steps: Number of steps to predict (default: 1)
            
        Returns:
            Predicted observations of shape (batch_size, steps, y_dim)
        """
        # Predict states
        states = self.predict(x0, u, steps)
        
        # Get observation matrix
        C = self.get_observation_matrix()
        
        # Compute observations
        return torch.matmul(states, C.transpose(-2, -1))
    
    @abstractmethod
    def get_state_transition_matrix(self) -> torch.Tensor:
        """
        Get the state transition matrix A.
        
        Returns:
            State transition matrix of shape (x_dim, x_dim)
        """
        pass
    
    @abstractmethod
    def get_input_matrix(self) -> torch.Tensor:
        """
        Get the input matrix B.
        
        Returns:
            Input matrix of shape (x_dim, u_dim)
        """
        pass
    
    @abstractmethod
    def get_observation_matrix(self) -> torch.Tensor:
        """
        Get the observation matrix C.
        
        Returns:
            Observation matrix of shape (y_dim, x_dim)
        """
        pass
    
    @abstractmethod
    def get_process_noise_covariance(self) -> torch.Tensor:
        """
        Get the process noise covariance matrix W.
        
        Returns:
            Process noise covariance matrix of shape (x_dim, x_dim)
        """
        pass
    
    @abstractmethod
    def get_observation_noise_covariance(self) -> torch.Tensor:
        """
        Get the observation noise covariance matrix V.
        
        Returns:
            Observation noise covariance matrix of shape (y_dim, y_dim)
        """
        pass
    
    def save(self, file_path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            file_path: Path to save the model
        """
        torch.save({
            'x_dim': self.x_dim,
            'y_dim': self.y_dim,
            'u_dim': self.u_dim,
            'dt': self.dt,
            'state_dict': self.state_dict(),
        }, file_path)
    
    @classmethod
    def load(cls, file_path: str, device: Optional[torch.device] = None) -> 'BaseStateSpaceModel':
        """
        Load the model from a file.
        
        Args:
            file_path: Path to load the model from
            device: Torch device to use (default: None, uses CPU)
            
        Returns:
            Loaded model
        """
        checkpoint = torch.load(file_path, map_location=device)
        model = cls(
            x_dim=checkpoint['x_dim'],
            y_dim=checkpoint['y_dim'],
            u_dim=checkpoint['u_dim'],
            dt=checkpoint['dt'],
            device=device
        )
        model.load_state_dict(checkpoint['state_dict'])
        return model


class LinearStateSpaceModel(BaseStateSpaceModel):
    """
    Linear State Space Model for EEG data.
    
    This model implements a linear state space model of the form:
    
    x_t = A x_{t-1} + w_t, w_t ~ N(0, Q)
    y_t = C x_t + v_t, v_t ~ N(0, R)
    
    where:
    - x_t is the latent state at time t
    - y_t is the observed data at time t
    - A is the state transition matrix
    - C is the observation matrix
    - Q is the process noise covariance
    - R is the observation noise covariance
    """
    
    def __init__(self, 
                 n_states: int, 
                 n_channels: int,
                 device: Optional[torch.device] = None):
        """
        Initialize the Linear State Space Model.
        
        Args:
            n_states: Number of latent states
            n_channels: Number of observed channels
            device: PyTorch device to use (default: None, use CPU)
        """
        super().__init__(x_dim=n_states, y_dim=n_channels, device=device)
        self.n_states = n_states
        self.n_channels = n_channels
    
    def initialize_parameters(self) -> None:
        """Initialize model parameters with reasonable defaults."""
        # State transition matrix (A)
        self.A = torch.eye(self.x_dim, device=self.device)
        
        # Observation matrix (C)
        self.C = torch.randn(self.y_dim, self.x_dim, device=self.device) * 0.1
        
        # Process noise covariance (Q)
        self.Q_chol = torch.eye(self.x_dim, device=self.device) * 0.1
        enforce_cholesky_structure(self.Q_chol)
        
        # Observation noise covariance (R)
        self.R_chol = torch.eye(self.y_dim, device=self.device) * 0.1
        enforce_cholesky_structure(self.R_chol)
        
        # Initial state mean and covariance
        self.x0_mean = torch.zeros(self.x_dim, device=self.device)
        self.x0_chol = torch.eye(self.x_dim, device=self.device) * 0.1
        enforce_cholesky_structure(self.x0_chol)
    
    @property
    def Q(self) -> torch.Tensor:
        """Process noise covariance matrix."""
        return self.Q_chol @ self.Q_chol.T
    
    @property
    def R(self) -> torch.Tensor:
        """Observation noise covariance matrix."""
        return self.R_chol @ self.R_chol.T
    
    @property
    def x0_cov(self) -> torch.Tensor:
        """Initial state covariance matrix."""
        return self.x0_chol @ self.x0_chol.T
    
    def forward(self, 
               y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run forward pass (filtering) through the model.
        
        Args:
            y: Observed data tensor of shape (batch_size, time_steps, n_channels)
            
        Returns:
            Tuple containing:
            - Filtered state means of shape (batch_size, time_steps, n_states)
            - Filtered state covariances of shape (batch_size, time_steps, n_states, n_states)
            - Log likelihood of the data
        """
        batch_size, time_steps, _ = y.shape
        
        # Initialize state estimates
        x_filt = torch.zeros(batch_size, time_steps, self.x_dim, device=self.device)
        P_filt = torch.zeros(batch_size, time_steps, self.x_dim, self.x_dim, device=self.device)
        
        # Initialize log likelihood
        log_likelihood = torch.zeros(batch_size, device=self.device)
        
        # Initial state
        x_pred = self.x0_mean.expand(batch_size, -1)
        P_pred = self.x0_cov.expand(batch_size, -1, -1)
        
        # Run Kalman filter
        for t in range(time_steps):
            # Get observation at time t
            y_t = y[:, t, :]
            
            # Kalman gain
            S = self.C @ P_pred @ self.C.T + self.R
            S = stabilize_covariance(S)
            K = P_pred @ self.C.T @ torch.inverse(S)
            
            # Update step
            x_filt[:, t, :] = x_pred + K @ (y_t - self.C @ x_pred).unsqueeze(-1)
            P_filt[:, t, :, :] = P_pred - K @ self.C @ P_pred
            
            # Compute log likelihood
            log_det_S = torch.logdet(S)
            y_pred_error = y_t - self.C @ x_pred
            weighted_error = torch.sum(y_pred_error @ torch.inverse(S) * y_pred_error, dim=1)
            log_likelihood -= 0.5 * (log_det_S + weighted_error + self.y_dim * np.log(2 * np.pi))
            
            # Predict next state
            if t < time_steps - 1:
                x_pred = self.A @ x_filt[:, t, :].unsqueeze(-1)
                P_pred = self.A @ P_filt[:, t, :, :] @ self.A.T + self.Q
        
        return x_filt, P_filt, log_likelihood
    
    def smooth(self, 
              x_filt: torch.Tensor, 
              P_filt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run backward pass (smoothing) through the model.
        
        Args:
            x_filt: Filtered state means from forward pass
            P_filt: Filtered state covariances from forward pass
            
        Returns:
            Tuple containing:
            - Smoothed state means
            - Smoothed state covariances
        """
        batch_size, time_steps, _ = x_filt.shape
        
        # Initialize smoothed estimates
        x_smooth = x_filt.clone()
        P_smooth = P_filt.clone()
        
        # Run Rauch-Tung-Striebel smoother
        for t in range(time_steps - 2, -1, -1):
            # Predict next state
            x_pred = self.A @ x_filt[:, t, :].unsqueeze(-1)
            P_pred = self.A @ P_filt[:, t, :, :] @ self.A.T + self.Q
            
            # Smoother gain
            J = P_filt[:, t, :, :] @ self.A.T @ torch.inverse(P_pred)
            
            # Update smoothed estimates
            x_smooth[:, t, :] = x_filt[:, t, :] + J @ (x_smooth[:, t+1, :] - x_pred).unsqueeze(-1)
            P_smooth[:, t, :, :] = P_filt[:, t, :, :] + J @ (P_smooth[:, t+1, :, :] - P_pred) @ J.transpose(-1, -2)
        
        return x_smooth, P_smooth
    
    def fit(self, 
           y: torch.Tensor, 
           n_iterations: int = 100,
           learning_rate: float = 0.01,
           verbose: bool = True) -> Dict[str, List[float]]:
        """
        Fit the model parameters to the data using expectation-maximization.
        
        Args:
            y: Observed data tensor of shape (batch_size, time_steps, n_channels)
            n_iterations: Number of EM iterations (default: 100)
            learning_rate: Learning rate for parameter updates (default: 0.01)
            verbose: Whether to print progress (default: True)
            
        Returns:
            Dictionary containing training history
        """
        history = {
            'log_likelihood': []
        }
        
        for i in range(n_iterations):
            # E-step: Run forward and backward passes
            x_filt, P_filt, log_likelihood = self.forward(y)
            x_smooth, P_smooth = self.smooth(x_filt, P_filt)
            
            # Store log likelihood
            avg_log_likelihood = log_likelihood.mean().item()
            history['log_likelihood'].append(avg_log_likelihood)
            
            if verbose and (i % 10 == 0 or i == n_iterations - 1):
                print(f"Iteration {i+1}/{n_iterations}, Log Likelihood: {avg_log_likelihood:.4f}")
            
            # M-step: Update parameters
            self._update_parameters(y, x_smooth, P_smooth, learning_rate)
        
        return history
    
    def _update_parameters(self, 
                          y: torch.Tensor, 
                          x_smooth: torch.Tensor, 
                          P_smooth: torch.Tensor,
                          learning_rate: float) -> None:
        """
        Update model parameters based on smoothed state estimates.
        
        Args:
            y: Observed data
            x_smooth: Smoothed state means
            P_smooth: Smoothed state covariances
            learning_rate: Learning rate for parameter updates
        """
        batch_size, time_steps, _ = y.shape
        
        # Compute sufficient statistics
        x_t = x_smooth[:, 1:, :]
        x_tm1 = x_smooth[:, :-1, :]
        
        # Update A
        A_num = torch.zeros_like(self.A)
        A_den = torch.zeros((self.x_dim, self.x_dim), device=self.device)
        
        for t in range(time_steps - 1):
            A_num += torch.sum(x_t[:, t, :].unsqueeze(-1) @ x_tm1[:, t, :].unsqueeze(-2), dim=0)
            A_den += torch.sum(x_tm1[:, t, :].unsqueeze(-1) @ x_tm1[:, t, :].unsqueeze(-2) + P_smooth[:, t, :, :], dim=0)
        
        A_new = A_num @ torch.inverse(A_den)
        self.A = (1 - learning_rate) * self.A + learning_rate * A_new
        
        # Update C
        C_num = torch.zeros_like(self.C)
        C_den = torch.zeros((self.y_dim, self.x_dim), device=self.device)
        
        for t in range(time_steps):
            C_num += torch.sum(y[:, t, :].unsqueeze(-1) @ x_smooth[:, t, :].unsqueeze(-2), dim=0)
            C_den += torch.sum(x_smooth[:, t, :].unsqueeze(-1) @ x_smooth[:, t, :].unsqueeze(-2) + P_smooth[:, t, :, :], dim=0)
        
        C_new = C_num @ torch.inverse(C_den)
        self.C = (1 - learning_rate) * self.C + learning_rate * C_new
        
        # Update Q
        Q_new = torch.zeros_like(self.Q)
        
        for t in range(time_steps - 1):
            pred_err = x_t[:, t, :] - self.A @ x_tm1[:, t, :]
            Q_new += torch.sum(pred_err.unsqueeze(-1) @ pred_err.unsqueeze(-2), dim=0) / batch_size
        
        Q_new /= (time_steps - 1)
        Q_new = stabilize_covariance(Q_new)
        Q_chol_new = safe_cholesky(Q_new)
        self.Q_chol = (1 - learning_rate) * self.Q_chol + learning_rate * Q_chol_new
        enforce_cholesky_structure(self.Q_chol)
        
        # Update R
        R_new = torch.zeros_like(self.R)
        
        for t in range(time_steps):
            obs_err = y[:, t, :] - self.C @ x_smooth[:, t, :]
            R_new += torch.sum(obs_err.unsqueeze(-1) @ obs_err.unsqueeze(-2), dim=0) / batch_size
        
        R_new /= time_steps
        R_new = stabilize_covariance(R_new)
        R_chol_new = safe_cholesky(R_new)
        self.R_chol = (1 - learning_rate) * self.R_chol + learning_rate * R_chol_new
        enforce_cholesky_structure(self.R_chol)
        
        # Update initial state
        x0_mean_new = x_smooth[:, 0, :].mean(dim=0)
        x0_cov_new = torch.zeros_like(self.x0_cov)
        
        for i in range(batch_size):
            x0_diff = x_smooth[i, 0, :] - x0_mean_new
            x0_cov_new += x0_diff.unsqueeze(-1) @ x0_diff.unsqueeze(-2) + P_smooth[i, 0, :, :]
        
        x0_cov_new /= batch_size
        x0_cov_new = stabilize_covariance(x0_cov_new)
        
        self.x0_mean = (1 - learning_rate) * self.x0_mean + learning_rate * x0_mean_new
        x0_chol_new = safe_cholesky(x0_cov_new)
        self.x0_chol = (1 - learning_rate) * self.x0_chol + learning_rate * x0_chol_new
        enforce_cholesky_structure(self.x0_chol)
    
    def save(self, file_path: str) -> None:
        """
        Save model parameters to a file.
        
        Args:
            file_path: Path to save the model to
        """
        torch.save({
            'n_states': self.n_states,
            'n_channels': self.n_channels,
            'A': self.A,
            'C': self.C,
            'Q_chol': self.Q_chol,
            'R_chol': self.R_chol,
            'x0_mean': self.x0_mean,
            'x0_chol': self.x0_chol
        }, file_path)
        print(f"Model saved to {file_path}")
    
    @classmethod
    def load(cls, file_path: str, device: Optional[torch.device] = None) -> 'LinearStateSpaceModel':
        """
        Load model parameters from a file.
        
        Args:
            file_path: Path to load the model from
            device: PyTorch device to use (default: None, use CPU)
            
        Returns:
            Loaded LinearStateSpaceModel
        """
        checkpoint = torch.load(file_path, map_location=device)
        model = cls(
            n_states=checkpoint['n_states'],
            n_channels=checkpoint['n_channels'],
            device=device
        )
        
        model.A = checkpoint['A']
        model.C = checkpoint['C']
        model.Q_chol = checkpoint['Q_chol']
        model.R_chol = checkpoint['R_chol']
        model.x0_mean = checkpoint['x0_mean']
        model.x0_chol = checkpoint['x0_chol']
        
        print(f"Model loaded from {file_path}")
        return model 