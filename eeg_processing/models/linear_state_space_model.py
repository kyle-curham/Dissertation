"""
Linear State Space Model

This module implements a linear state-space model for EEG data analysis.
The model is defined as:

x_{t+1} = A x_t + B u_t + w_t,    w_t ~ N(0, W)
y_t = C x_t + v_t,                v_t ~ N(0, V)

This implementation includes Kalman filtering and learning algorithms.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any

from .state_space_model import BaseStateSpaceModel
from ..utils.numerical_stability import ensure_positive_definite, enforce_cholesky_structure


class LinearStateSpaceModel(BaseStateSpaceModel):
    """
    Linear State Space Model with Kalman filtering capabilities.
    
    This model implements a standard linear state-space model with
    Kalman filtering for state estimation and EM for parameter learning.
    """
    
    def __init__(self, 
                 x_dim: int, 
                 y_dim: int, 
                 u_dim: Optional[int] = None,
                 dt: float = 1.0,
                 use_control: bool = False,
                 device: Optional[torch.device] = None) -> None:
        """
        Initialize the linear state-space model.
        
        Args:
            x_dim: Dimension of the state vector
            y_dim: Dimension of the observation vector
            u_dim: Dimension of the input/control vector (default: None)
            dt: Time step (default: 1.0)
            use_control: Whether to use control inputs (default: False)
            device: Torch device to use (default: None, uses CPU)
        """
        self.use_control = use_control
        super().__init__(x_dim, y_dim, u_dim, dt, device)
    
    def initialize_parameters(self) -> None:
        """
        Initialize model parameters.
        
        This method initializes the model parameters (A, B, C, W, V)
        with reasonable default values.
        """
        # State transition matrix (A)
        self.register_parameter(
            'A',
            nn.Parameter(torch.eye(self.x_dim, device=self.device))
        )
        
        # Input matrix (B), only if control inputs are used
        if self.use_control and self.u_dim > 0:
            self.register_parameter(
                'B',
                nn.Parameter(torch.zeros(self.x_dim, self.u_dim, device=self.device))
            )
        
        # Observation matrix (C)
        self.register_parameter(
            'C',
            nn.Parameter(torch.randn(self.y_dim, self.x_dim, device=self.device) * 0.1)
        )
        
        # Process noise covariance (W) - parameterized as Cholesky factor for positive definiteness
        self.register_parameter(
            'W_chol',
            nn.Parameter(torch.eye(self.x_dim, device=self.device) * 0.1)
        )
        
        # Observation noise covariance (V) - parameterized as Cholesky factor
        self.register_parameter(
            'V_chol',
            nn.Parameter(torch.eye(self.y_dim, device=self.device) * 0.1)
        )
        
        # Initial state mean and covariance
        self.register_buffer('x0_mean', torch.zeros(self.x_dim, device=self.device))
        self.register_parameter(
            'x0_cov_chol', 
            nn.Parameter(torch.eye(self.x_dim, device=self.device) * 0.1)
        )
        
        # Enforce Cholesky structure of covariance factors
        with torch.no_grad():
            enforce_cholesky_structure(self.W_chol, self.V_chol, self.x0_cov_chol)
    
    def forward(self, 
                y: torch.Tensor, 
                u: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Forward pass of the model.
        
        Performs Kalman filtering to estimate states from observations.
        
        Args:
            y: Observation tensor of shape (batch_size, sequence_length, y_dim)
            u: Input tensor of shape (batch_size, sequence_length, u_dim) (optional)
            
        Returns:
            Dictionary containing:
                - x_filtered: Filtered state estimates
                - P_filtered: Filtered state covariances
                - x_smoothed: Smoothed state estimates (if smooth=True)
                - P_smoothed: Smoothed state covariances (if smooth=True)
                - log_likelihood: Log likelihood of the observations
        """
        # Check input dimensionality
        if y.dim() != 3 or y.size(2) != self.y_dim:
            raise ValueError(
                f"Expected y to have shape (batch_size, seq_len, {self.y_dim}), "
                f"but got {y.shape}"
            )
        
        if self.use_control and self.u_dim > 0:
            if u is None:
                raise ValueError("Control inputs (u) are required when use_control=True")
            if u.dim() != 3 or u.size(2) != self.u_dim:
                raise ValueError(
                    f"Expected u to have shape (batch_size, seq_len, {self.u_dim}), "
                    f"but got {u.shape if u is not None else None}"
                )
        
        # Get batch size and sequence length
        batch_size, seq_len, _ = y.shape
        
        # Run Kalman filter
        filter_results = self.kalman_filter(y, u)
        
        # Run Kalman smoother if requested
        smoothed_results = self.kalman_smoother(filter_results)
        
        return {**filter_results, **smoothed_results}
    
    def kalman_filter(self, 
                     y: torch.Tensor, 
                     u: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Perform Kalman filtering on the observations.
        
        Args:
            y: Observation tensor of shape (batch_size, sequence_length, y_dim)
            u: Input tensor of shape (batch_size, sequence_length, u_dim) (optional)
            
        Returns:
            Dictionary containing:
                - x_filtered: Filtered state estimates
                - P_filtered: Filtered state covariances
                - log_likelihood: Log likelihood of the observations
        """
        with torch.no_grad():
            enforce_cholesky_structure(self.W_chol, self.V_chol, self.x0_cov_chol)
        
        # Get covariance matrices from Cholesky factors
        W = self.W_chol @ self.W_chol.transpose(-2, -1)
        V = self.V_chol @ self.V_chol.transpose(-2, -1)
        x0_cov = self.x0_cov_chol @ self.x0_cov_chol.transpose(-2, -1)
        
        # Get dimensions
        batch_size, seq_len, _ = y.shape
        
        # Initialize state estimates and covariances
        x_filt = torch.zeros(batch_size, seq_len, self.x_dim, device=self.device)
        P_filt = torch.zeros(batch_size, seq_len, self.x_dim, self.x_dim, device=self.device)
        
        # Initialize log likelihood
        log_likelihood = torch.zeros(batch_size, device=self.device)
        
        # Initialize with prior
        x_pred = self.x0_mean.expand(batch_size, -1)
        P_pred = x0_cov.expand(batch_size, -1, -1)
        
        # Kalman filter loop
        for t in range(seq_len):
            # Get observation at time t
            y_t = y[:, t, :]
            
            # Compute Kalman gain
            K_t = self._compute_kalman_gain(P_pred, V)
            
            # Update state estimate
            x_filt[:, t, :] = self._update_state(x_pred, y_t, K_t)
            
            # Update state covariance
            P_filt[:, t, :, :] = self._update_covariance(P_pred, K_t)
            
            # Compute log likelihood
            log_likelihood += self._compute_log_likelihood(y_t, x_pred, P_pred, V)
            
            # Predict next state
            if t < seq_len - 1:
                # Get control input if provided
                u_t = None
                if self.use_control and self.u_dim > 0 and u is not None:
                    u_t = u[:, t, :]
                
                # Predict next state
                x_pred, P_pred = self._predict_state(x_filt[:, t, :], P_filt[:, t, :, :], u_t, W)
        
        return {
            'x_filtered': x_filt,
            'P_filtered': P_filt,
            'log_likelihood': log_likelihood
        }
    
    def kalman_smoother(self, filter_results: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform Kalman smoothing using the results from Kalman filtering.
        
        Args:
            filter_results: Results from Kalman filtering
            
        Returns:
            Dictionary containing:
                - x_smoothed: Smoothed state estimates
                - P_smoothed: Smoothed state covariances
        """
        # Extract filter results
        x_filt = filter_results['x_filtered']
        P_filt = filter_results['P_filtered']
        
        # Get dimensions
        batch_size, seq_len, _ = x_filt.shape
        
        # Initialize smoothed state estimates and covariances
        x_smooth = torch.zeros_like(x_filt)
        P_smooth = torch.zeros_like(P_filt)
        
        # Last filtered state becomes last smoothed state
        x_smooth[:, -1, :] = x_filt[:, -1, :]
        P_smooth[:, -1, :, :] = P_filt[:, -1, :, :]
        
        # Backward smoothing loop
        for t in range(seq_len - 2, -1, -1):
            # Get filtered state and covariance at time t
            x_t = x_filt[:, t, :]
            P_t = P_filt[:, t, :, :]
            
            # Predict next state and covariance
            u_t = None  # No control input for smoothing
            x_pred, P_pred = self._predict_state(x_t, P_t, u_t, 
                                               self.W_chol @ self.W_chol.transpose(-2, -1))
            
            # Compute smoothing gain
            G_t = torch.bmm(P_t, torch.bmm(self.A.expand(batch_size, -1, -1).transpose(1, 2),
                                         torch.inverse(P_pred)))
            
            # Update smoothed state
            x_smooth[:, t, :] = x_t + torch.bmm(G_t, (x_smooth[:, t+1, :] - x_pred).unsqueeze(-1)).squeeze(-1)
            
            # Update smoothed covariance
            P_diff = P_smooth[:, t+1, :, :] - P_pred
            P_smooth[:, t, :, :] = P_t + torch.bmm(G_t, torch.bmm(P_diff, G_t.transpose(1, 2)))
        
        return {
            'x_smoothed': x_smooth,
            'P_smoothed': P_smooth
        }
    
    def _compute_kalman_gain(self, 
                           P_pred: torch.Tensor, 
                           V: torch.Tensor) -> torch.Tensor:
        """
        Compute the Kalman gain.
        
        Args:
            P_pred: Predicted state covariance [batch, x_dim, x_dim]
            V: Observation noise covariance [y_dim, y_dim]
            
        Returns:
            Kalman gain [batch, x_dim, y_dim]
        """
        # Innovation covariance
        S = self._compute_innovation_covariance(P_pred, V)  # [batch, y_dim, y_dim]
        
        # Expand C matrix for batch operations
        batch_size = P_pred.size(0)
        C_expanded = self.C.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, y_dim, x_dim]
        
        # Kalman gain: K = P_pred C^T S^{-1}
        PCt = torch.bmm(P_pred, C_expanded.transpose(1, 2))  # [batch, x_dim, y_dim]
        S_inv = torch.inverse(S)  # [batch, y_dim, y_dim]
        
        return torch.bmm(PCt, S_inv)  # [batch, x_dim, y_dim]
    
    def _compute_innovation_covariance(self, P_pred: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Compute innovation covariance S = C P_pred C^T + V"""
        # Expand C matrix for batch operations
        C_expanded = self.C.unsqueeze(0).expand(P_pred.size(0), -1, -1)  # [batch, y_dim, x_dim]
        
        # Compute CPC^T term
        CPC = torch.bmm(torch.bmm(C_expanded, P_pred), C_expanded.transpose(1, 2))
        
        # Add observation noise covariance with proper broadcasting
        S = CPC + V.unsqueeze(0).expand(CPC.size(0), -1, -1)  # [batch, y_dim, y_dim]
        
        return S
    
    def _update_state(self, 
                    x_pred: torch.Tensor, 
                    y: torch.Tensor, 
                    K: torch.Tensor) -> torch.Tensor:
        """
        Update the state estimate using the Kalman filter update step.
        
        Args:
            x_pred: Predicted state
            y: Observation
            K: Kalman gain
            
        Returns:
            Updated state estimate
        """
        # Innovation
        innovation = y - torch.mm(self.C, x_pred.transpose(0, 1)).transpose(0, 1)
        
        # Updated state
        return x_pred + torch.bmm(K, innovation.unsqueeze(-1)).squeeze(-1)
    
    def _update_covariance(self, 
                         P_pred: torch.Tensor, 
                         K: torch.Tensor) -> torch.Tensor:
        """
        Update the state covariance using the Kalman filter update step.
        
        Args:
            P_pred: Predicted state covariance
            K: Kalman gain
            
        Returns:
            Updated state covariance
        """
        # Identity matrix
        I = torch.eye(self.x_dim, device=self.device).expand_as(P_pred)
        
        # Updated covariance
        KC = torch.bmm(K, self.C.expand(P_pred.size(0), -1, -1))
        return torch.bmm(I - KC, P_pred)
    
    def _predict_state(self, 
                     x: torch.Tensor, 
                     P: torch.Tensor, 
                     u: Optional[torch.Tensor], 
                     W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the next state using the Kalman filter prediction step.
        
        Args:
            x: Current state estimate
            P: Current state covariance
            u: Control input (optional)
            W: Process noise covariance
            
        Returns:
            Tuple of predicted state and covariance
        """
        # Expand matrices for batch operations
        batch_size = x.size(0)
        A_expanded = self.A.expand(batch_size, -1, -1)
        
        # Predict state
        x_pred = torch.bmm(A_expanded, x.unsqueeze(-1)).squeeze(-1)
        
        # Add control input effect if provided
        if self.use_control and self.u_dim > 0 and u is not None:
            B_expanded = self.B.expand(batch_size, -1, -1)
            x_pred = x_pred + torch.bmm(B_expanded, u.unsqueeze(-1)).squeeze(-1)
        
        # Predict covariance
        P_pred = torch.bmm(torch.bmm(A_expanded, P), A_expanded.transpose(1, 2)) + W.expand_as(P)
        
        return x_pred, P_pred
    
    def _compute_log_likelihood(self, 
                              y: torch.Tensor, 
                              x_pred: torch.Tensor, 
                              P_pred: torch.Tensor, 
                              V: torch.Tensor) -> torch.Tensor:
        """
        Compute the log likelihood of the observations.
        
        Args:
            y: Observation
            x_pred: Predicted state
            P_pred: Predicted state covariance
            V: Observation noise covariance
            
        Returns:
            Log likelihood
        """
        # Innovation covariance
        S = self._compute_innovation_covariance(P_pred, V)
        
        # Innovation
        v = y - torch.mm(self.C, x_pred.transpose(0, 1)).transpose(0, 1)
        
        # Cast to float32 for logdet calculation
        S_fp32 = S.to(torch.float32)
        log_det_S = torch.logdet(S_fp32).to(S.dtype)
        
        # Weighted innovation
        weighted_v = torch.bmm(v.unsqueeze(1), torch.inverse(S)).bmm(v.unsqueeze(-1)).squeeze()
        
        # Log likelihood
        return -0.5 * (log_det_S + weighted_v + self.y_dim * np.log(2 * np.pi))
    
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
        batch_size = x0.size(0)
        
        # Initialize predictions
        x_pred = torch.zeros(batch_size, steps, self.x_dim, device=self.device)
        
        # Initial state
        x_t = x0
        
        # Prediction loop
        for t in range(steps):
            # Predict next state
            x_next = torch.mm(self.A, x_t.transpose(0, 1)).transpose(0, 1)
            
            # Add control input effect if provided
            if self.use_control and self.u_dim > 0 and u is not None:
                u_t = u[:, t, :]
                x_next = x_next + torch.mm(self.B, u_t.transpose(0, 1)).transpose(0, 1)
            
            # Store prediction
            x_pred[:, t, :] = x_next
            
            # Update current state for next iteration
            x_t = x_next
        
        return x_pred
    
    def get_state_transition_matrix(self) -> torch.Tensor:
        """
        Get the state transition matrix A.
        
        Returns:
            State transition matrix of shape (x_dim, x_dim)
        """
        return self.A
    
    def get_input_matrix(self) -> torch.Tensor:
        """
        Get the input matrix B.
        
        Returns:
            Input matrix of shape (x_dim, u_dim)
        """
        if self.use_control and self.u_dim > 0:
            return self.B
        else:
            return torch.zeros(self.x_dim, 0, device=self.device)
    
    def get_observation_matrix(self) -> torch.Tensor:
        """
        Get the observation matrix C.
        
        Returns:
            Observation matrix of shape (y_dim, x_dim)
        """
        return self.C
    
    def get_process_noise_covariance(self) -> torch.Tensor:
        """
        Get the process noise covariance matrix W.
        
        Returns:
            Process noise covariance matrix of shape (x_dim, x_dim)
        """
        return self.W_chol @ self.W_chol.transpose(-2, -1)
    
    def get_observation_noise_covariance(self) -> torch.Tensor:
        """
        Get the observation noise covariance matrix V.
        
        Returns:
            Observation noise covariance matrix of shape (y_dim, y_dim)
        """
        return self.V_chol @ self.V_chol.transpose(-2, -1)
    
    def fit(self, 
            y: torch.Tensor, 
            u: Optional[torch.Tensor] = None,
            n_iterations: int = 10, 
            lr: float = 1e-3,
            optimizer: str = 'adam') -> Dict[str, List[float]]:
        """
        Fit the model to data using maximum likelihood estimation.
        
        Args:
            y: Observation tensor of shape (batch_size, sequence_length, y_dim)
            u: Input tensor of shape (batch_size, sequence_length, u_dim) (optional)
            n_iterations: Number of iterations (default: 10)
            lr: Learning rate (default: 1e-3)
            optimizer: Optimizer to use ('adam' or 'sgd', default: 'adam')
            
        Returns:
            Dictionary containing training metrics (loss history)
        """
        # Choose optimizer
        if optimizer.lower() == 'adam':
            opt = torch.optim.Adam(self.parameters(), lr=lr)
        elif optimizer.lower() == 'sgd':
            opt = torch.optim.SGD(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        # Training history
        history = {
            'loss': [],
            'log_likelihood': []
        }
        
        # Training loop
        for iteration in range(n_iterations):
            # Zero gradients
            opt.zero_grad()
            
            # Forward pass with mixed precision
            with torch.autocast(device_type='cuda' if 'cuda' in str(self.device) else 'cpu',
                               dtype=torch.float32):  # Force float32 for stability
                results = self.forward(y, u)
                log_likelihood = results['log_likelihood']
                loss = -log_likelihood.mean()
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            opt.step()
            
            # Enforce constraints WITHOUT breaking computation graph
            with torch.no_grad():
                # Create new lower triangular parameters
                new_W = torch.tril(self.W_chol.data)
                new_V = torch.tril(self.V_chol.data)
                new_x0 = torch.tril(self.x0_cov_chol.data)
                
                # Copy values back to parameters
                self.W_chol.data = new_W
                self.V_chol.data = new_V
                self.x0_cov_chol.data = new_x0
            
            # Record metrics
            history['loss'].append(loss.item())
            history['log_likelihood'].append(log_likelihood.mean().item())
            
            # Print progress
            if (iteration + 1) % (n_iterations // 10 or 1) == 0:
                print(f"Iteration {iteration + 1}/{n_iterations}, Loss: {loss.item():.4f}")
        
        return history 