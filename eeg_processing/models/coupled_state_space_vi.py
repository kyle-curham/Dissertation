"""
Coupled State Space Model with Variational Inference

This module implements a coupled state-space model with variational inference,
using a 2-level augmented system structure to simplify dynamics.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
import scipy.linalg as la

class CoupledStateSpaceVI(nn.Module):
    """
    Coupled State-Space Model with Variational Inference.
    
    This model uses a 2-level augmented system with structure:
    x_dot = Ax + Bu  (state dynamics)
    y = Cx + noise   (observations)
    """
    
    def __init__(self, 
                 x_dim: int, 
                 y_dim: int, 
                 C: np.ndarray, 
                 u_dim: Optional[int] = None,
                 beta: float = 0.1, 
                 prior_std: float = 1.0, 
                 dt: float = 1/1024,
                 eps: float = 1e-4) -> None:
        """
        Initialize the model.
        
        Args:
            x_dim: State dimension
            y_dim: Observation dimension
            C: Observation matrix
            u_dim: Input dimension (default: x_dim)
            beta: KL divergence weight
            prior_std: Prior standard deviation
            dt: Time step
            eps: Small constant for stability
        """
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.u_dim = u_dim if u_dim is not None else x_dim
        self.beta = beta
        self.prior_std = prior_std
        self.dt = dt
        self.eps = eps
        
        # Device setup 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model parameters
        self.C = nn.Parameter(torch.tensor(C, dtype=torch.float32, device=self.device))
        self._A = nn.Parameter(torch.randn(x_dim, x_dim, device=self.device) * 0.1)
        self._B = nn.Parameter(torch.eye(x_dim, self.u_dim, device=self.device))
        self._Q = nn.Parameter(torch.eye(x_dim, device=self.device))
        self._R = nn.Parameter(torch.eye(self.u_dim, device=self.device))
        
        # Initialize P for Riccati equation
        self.P = torch.eye(2*x_dim, device=self.device)
    
    # Property accessors
    @property
    def A(self): return self._A
    
    @property
    def B(self): return self._B
    
    @property
    def Q(self): return self._Q
    
    @property
    def R(self): return self._R
    
    def init_variational_params(self, train_length: int, val_length: Optional[int] = None) -> None:
        """Initialize variational parameters for training and validation."""
        # For the 2-level model, the augmented state has dimension 2*x_dim
        aug_dim = 2 * self.x_dim
        self.q_mu_train = nn.Parameter(torch.randn(train_length, aug_dim, device=self.device) * 0.01)
        self.q_logvar_train = nn.Parameter(torch.zeros(train_length, aug_dim, device=self.device))
        
        if val_length is not None:
            self.q_mu_val = nn.Parameter(torch.randn(val_length, aug_dim, device=self.device) * 0.01)
            self.q_logvar_val = nn.Parameter(torch.zeros(val_length, aug_dim, device=self.device))
    
    def sample_augmented_state(self, is_validation: bool = False) -> torch.Tensor:
        """Sample from the variational distribution of the augmented state."""
        if is_validation:
            std = torch.exp(0.5 * self.q_logvar_val)
            eps = torch.randn_like(std)
            return self.q_mu_val + eps * std
        else:
            std = torch.exp(0.5 * self.q_logvar_train)
            eps = torch.randn_like(std)
            return self.q_mu_train + eps * std
    
    def build_augmented_system(self, A: torch.Tensor, B: torch.Tensor, Q: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """
        Build the augmented system matrix for the 2-level system with structure:
        [  A   -BR^(-1)B^T ]
        [ -Q      -A^T     ]
        """
        # Compute BR^(-1)B^T
        R_inv = torch.inverse(R)
        BR_inv_BT = B @ R_inv @ B.T
        
        # Build the augmented system matrix
        A_top = torch.cat([A, -BR_inv_BT], dim=1)
        A_bottom = torch.cat([-Q, -A.T], dim=1)
        return torch.cat([A_top, A_bottom], dim=0)
    
    def implicit_euler_step(self, A: torch.Tensor, Q: torch.Tensor, X: torch.Tensor, h: float) -> torch.Tensor:
        """Perform implicit Euler step for the Riccati equation."""
        # For augmented system, expand Q to full size if needed
        if Q.shape[0] != X.shape[0]:
            aug_dim = X.shape[0]
            Q_aug = torch.zeros(aug_dim, aug_dim, device=self.device)
            Q_aug[:self.x_dim, :self.x_dim] = Q
        else:
            Q_aug = Q
            
        # Simple implementation with fewer iterations
        RHS = X + h * Q_aug
        X_new = X.clone()
        for _ in range(3):  # Reduced iterations for efficiency
            X_new = RHS + h * (A.T @ X_new + X_new @ A)
        return 0.5 * (X_new + X_new.T)  # Ensure symmetry
    
    def elbo(self, y: torch.Tensor, batch_start: int, batch_size: int, is_validation: bool = False) -> Tuple[Dict, Dict, torch.Tensor]:
        """Compute the Evidence Lower BOund (ELBO)."""
        batch_end = min(batch_start + batch_size, y.shape[0])
        actual_batch_size = batch_end - batch_start
        
        if is_validation:
            q_mu = self.q_mu_val[batch_start:batch_end]
            q_logvar = self.q_logvar_val[batch_start:batch_end]
        else:
            q_mu = self.q_mu_train[batch_start:batch_end]
            q_logvar = self.q_logvar_train[batch_start:batch_end]
        
        # KL divergence term
        prior_var = torch.tensor(self.prior_std ** 2, device=self.device, dtype=torch.float32)
        kl_loss = -0.5 * torch.sum(1 + q_logvar - torch.log(prior_var) - (q_mu.pow(2) + q_logvar.exp()) / prior_var)
        weighted_kl_loss = self.beta * kl_loss

        # Sample from variational distribution
        std = torch.exp(0.5 * q_logvar)
        z = q_mu + torch.randn_like(std) * std

        # Reconstruction loss
        y_batch = y[batch_start:batch_end]
        x_batch = z[:, :self.x_dim]  # Extract state part
        y_pred_batch = torch.matmul(x_batch, self.C.T)
        recon_error_batch = y_batch - y_pred_batch
        recon_loss = -0.5 * (torch.sum(recon_error_batch * recon_error_batch) +
                             actual_batch_size * self.y_dim * torch.log(torch.tensor(2 * np.pi, device=self.device)))

        # Dynamics loss
        dt = self.dt
        P_current = self.P.clone()
        dynamics_loss = torch.tensor(0.0, device=self.device)
        
        # Build augmented system once outside the loop
        A_aug = self.build_augmented_system(self.A, self.B, self.Q, self.R)
        

        
        # Compute dynamics loss for each timestep
        for t in range(actual_batch_size - 1):
            # Update dynamics using augmented system
            z_next_pred = z[t] + dt * (A_aug @ z[t])
            
            # Update covariance 
            P_current = self.implicit_euler_step(A_aug, self.Q, P_current, dt)
            
            # Compute dynamics error
            dynamics_error = z[t+1] - z_next_pred
            
            # Dynamics loss term
            dynamics_term = -0.5 * (
                torch.sum(dynamics_error * dynamics_error) + 
                2 * self.x_dim * torch.log(torch.tensor(2 * np.pi, device=self.device))
            )
            
            if torch.isfinite(dynamics_term):
                dynamics_loss = dynamics_loss + dynamics_term
        
        # Store updated P
        self.P = P_current.clone()

        # Combine ELBO terms
        elbo_terms = {
            'recon_loss': recon_loss,
            'dynamics_loss': dynamics_loss,
            'kl_loss': weighted_kl_loss
        }
        
        # Basic diagnostics
        diagnostics = {
            'elbo_terms': {
                'recon_loss': recon_loss.item(),
                'dynamics_loss': dynamics_loss.item(),
                'kl_loss': kl_loss.item(),
                'weighted_kl_loss': weighted_kl_loss.item()
            }
        }
        
        return elbo_terms, diagnostics, P_current
    
    def forward(self, y: torch.Tensor, is_validation: bool = False) -> Dict[str, Any]:
        """
        Forward pass through the model.
        
        Args:
            y: Observations tensor of shape (time_steps, y_dim)
            is_validation: Whether to use validation parameters
            
        Returns:
            Dictionary of results
        """
        batch_size = min(128, y.shape[0])
        batch_start = 0
        
        elbo_terms, diagnostics, _ = self.elbo(y, batch_start, batch_size, is_validation)
        
        # Sample augmented states
        z = self.sample_augmented_state(is_validation)
        
        # Extract latent states and adjoint states
        x = z[:, :self.x_dim]
        p = z[:, self.x_dim:]
        
        # Compute reconstruction
        y_pred = torch.matmul(x, self.C.T)
        
        # Compute optimal control inputs
        try:
            # Convert to numpy for scipy's ARE solver
            A_np = self.A.detach().cpu().numpy()
            B_np = self.B.detach().cpu().numpy()
            Q_np = self.Q.detach().cpu().numpy()
            R_np = self.R.detach().cpu().numpy()
            
            # Solve continuous algebraic Riccati equation
            P_np = la.solve_continuous_are(A_np, B_np, Q_np, R_np)
            
            # Compute gain matrix K
            R_inv_np = la.inv(R_np)
            K_np = np.dot(R_inv_np, np.dot(B_np.T, P_np))
            
            # Convert back to torch and compute control
            K = torch.tensor(K_np, dtype=torch.float32, device=self.device)
            u = -torch.matmul(K, x.T).T
        except:
            # Simple fallback
            R_inv = torch.inverse(self.R)
            u = -torch.matmul(R_inv, torch.matmul(self.B.T, p.T)).T
        
        # Compute total ELBO
        total_elbo = elbo_terms['recon_loss'] + elbo_terms['dynamics_loss'] - elbo_terms['kl_loss']
        
        # Return simplified results
        return {
            'elbo': total_elbo.item(),
            'recon_loss': elbo_terms['recon_loss'].item(),
            'dynamics_loss': elbo_terms['dynamics_loss'].item(),
            'kl_loss': elbo_terms['kl_loss'].item(),
            'latent_states': x.detach(),
            'adjoint_states': p.detach(),
            'control_inputs': u.detach(),
            'predicted_observations': y_pred.detach(),
            'diagnostics': diagnostics
        }
    
    def fit(self, 
            y: torch.Tensor, 
            n_epochs: int = 100, 
            batch_size: int = 64, 
            learning_rate: float = 1e-3,
            validation_data: Optional[torch.Tensor] = None,
            verbose: bool = True) -> Dict[str, List[float]]:
        """
        Fit the model to data.
        
        Args:
            y: Observations tensor
            n_epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            validation_data: Optional validation data
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        # Initialize variational parameters
        self.init_variational_params(
            train_length=y.shape[0],
            val_length=validation_data.shape[0] if validation_data is not None else None
        )
        
        # Setup optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Training history
        history = {
            'elbo': [],
            'val_elbo': [] if validation_data is not None else None
        }
        
        # Training loop
        for epoch in range(n_epochs):
            # Shuffle data
            perm = torch.randperm(y.shape[0])
            
            # Process data in batches
            epoch_elbo = 0.0
            n_batches = 0
            
            for i in range(0, y.shape[0], batch_size):
                batch_indices = perm[i:min(i+batch_size, y.shape[0])]
                batch_y = y[batch_indices]
                
                # Forward pass
                optimizer.zero_grad()
                elbo_terms, _, _ = self.elbo(y, batch_indices[0], len(batch_indices))
                
                # Compute ELBO
                elbo = elbo_terms['recon_loss'] + elbo_terms['dynamics_loss'] - elbo_terms['kl_loss']
                
                # Compute loss (negative ELBO for minimization)
                loss = -elbo
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                # Track ELBO
                epoch_elbo += elbo.item()
                n_batches += 1
            
            # Compute average ELBO
            epoch_elbo /= n_batches
            history['elbo'].append(epoch_elbo)
            
            # Validation if provided
            if validation_data is not None:
                with torch.no_grad():
                    val_elbo_terms, _, _ = self.elbo(validation_data, 0, validation_data.shape[0], is_validation=True)
                    val_elbo = val_elbo_terms['recon_loss'] + val_elbo_terms['dynamics_loss'] - val_elbo_terms['kl_loss']
                    
                    # Normalize by number of samples to make comparable to training
                    val_elbo = val_elbo / validation_data.shape[0]
                    history['val_elbo'].append(val_elbo.item())
            
            # Print progress
            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                if validation_data is not None:
                    print(f"Epoch {epoch+1}/{n_epochs}: ELBO = {epoch_elbo:.4f}, Val ELBO = {history['val_elbo'][-1]:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{n_epochs}: ELBO = {epoch_elbo:.4f}")
        
        return history
    
    def save(self, file_path: str) -> None:
        """Save the model to a file."""
        state_dict = {
            'model_state': self.state_dict(),
            'model_config': {
                'x_dim': self.x_dim,
                'y_dim': self.y_dim,
                'u_dim': self.u_dim,
                'beta': self.beta,
                'prior_std': self.prior_std,
                'dt': self.dt,
                'eps': self.eps
            }
        }
        torch.save(state_dict, file_path)
    
    @classmethod
    def load(cls, file_path: str, device: Optional[torch.device] = None) -> 'CoupledStateSpaceVI':
        """Load a model from a file."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        checkpoint = torch.load(file_path, map_location=device)
        
        # Get config and create model
        config = checkpoint['model_config']
        model = cls(
            x_dim=config['x_dim'],
            y_dim=config['y_dim'],
            u_dim=config['u_dim'],
            beta=config['beta'],
            prior_std=config['prior_std'],
            dt=config['dt'],
            eps=config['eps'],
            C=np.eye(config['y_dim'], config['x_dim'])  # Temporary C matrix
        )
        
        # Load state dictionary
        model.load_state_dict(checkpoint['model_state'])
        return model 