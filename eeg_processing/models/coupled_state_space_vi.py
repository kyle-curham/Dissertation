"""
Coupled State Space Model with Variational Inference and Optimal Control
This module implements a state-space model with deterministic dynamics,
optimal control, and variational inference for sequential data.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from scipy.sparse.linalg import svds

# Placeholder for RiccatiSolver (assumed to be part of your project)
class RiccatiSolver:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class StateSpaceEncoder(nn.Module):
    """
    Encoder network for amortized inference in state space models.
    Implements q(x_t | y_1:t) for filtering.
    """
    def __init__(self, y_dim: int, x_dim: int, h_dim: int = 128, n_layers: int = 2, bidirectional: bool = False):
        super().__init__()
        self.y_dim = y_dim
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(
            input_size=y_dim,
            hidden_size=h_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        rnn_output_size = h_dim * 2 if bidirectional else h_dim
        self.state_mean = nn.Linear(rnn_output_size, x_dim)
        self.state_logvar = nn.Linear(rnn_output_size, x_dim)

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rnn_output, _ = self.rnn(y)
        x_mean = self.state_mean(rnn_output)
        x_logvar = torch.clamp(self.state_logvar(rnn_output), min=-10, max=10)
        return x_mean, x_logvar

class CoupledStateSpaceVI(nn.Module):
    """
    Coupled State Space Model with Variational Inference and Optimal Control.
    Dynamics: x_{t+1} = x_t + dt * (A x_t + B u_t)
    Observation: y_t = C x_t + v_t, v_t ~ N(0, V)
    Control cost: J = sum_t (x_t^T Q x_t + u_t^T R u_t)
    """
    def __init__(self,
                 x_dim: int,
                 y_dim: int,
                 C: np.ndarray,
                 u_dim: Optional[int] = None,
                 beta: float = 0.1,
                 prior_std: float = 1.0,
                 dt: float = 1/1024,
                 eps: float = 1e-4,
                 gamma: float = 0.001,  # Control cost weight
                 lambda_dyn: float = 100.0,  # Dynamics penalty weight
                 encoder_h_dim: int = 128,
                 device: Optional[torch.device] = None) -> None:
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.u_dim = u_dim if u_dim is not None else x_dim
        self.beta = beta
        self.prior_std = prior_std
        self.dt = dt
        self.eps = eps
        self.gamma = gamma
        self.lambda_dyn = lambda_dyn

        self._riccati_solver = RiccatiSolver(device=device)
        self._device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # State transition matrix A (stable)
        A_init = torch.zeros(x_dim, x_dim)
        for i in range(x_dim):
            A_init[i, i] = -0.01 * (i + 1)
        self.A = nn.Parameter(A_init)

        # Input matrix B
        self.B = nn.Parameter(torch.eye(x_dim, self.u_dim) * 0.1)

        # Observation matrix C (fixed)
        self.register_buffer('C', torch.tensor(C, dtype=torch.float32, device=self._device))

        # State cost matrix Q (reduced scale)
        self.Q = nn.Parameter(torch.eye(x_dim, device=self._device) * 0.1)

        # Control cost matrix R (reduced scale)
        self.R = nn.Parameter(torch.eye(self.u_dim, device=self._device) * 0.1)

        # Observation noise covariance V
        self.Vn = nn.Parameter(torch.eye(y_dim, device=self._device) * 0.1)

        # Prior mean
        self.register_buffer('prior_mean', torch.zeros(x_dim, device=self._device))

        self.encoder = StateSpaceEncoder(y_dim=y_dim, x_dim=x_dim, h_dim=encoder_h_dim, bidirectional=False)

    @property
    def device(self) -> torch.device:
        return self.A.device

    @property
    def Q_chol(self) -> torch.Tensor:
        return self._ensure_psd(self.Q)

    @property
    def R_chol(self) -> torch.Tensor:
        return self._ensure_psd(self.R)

    @property
    def V_chol(self) -> torch.Tensor:
        return self._ensure_psd(self.Vn)

    def _ensure_psd(self, mat: torch.Tensor) -> torch.Tensor:
        return torch.linalg.cholesky(mat @ mat.T + torch.eye(mat.shape[0], device=self.device) * self.eps)

    def sample_from_gaussian(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps

    def kl_divergence(self, mean: torch.Tensor, logvar: torch.Tensor, prior_mean: torch.Tensor = None) -> torch.Tensor:
        if prior_mean is None:
            prior_mean = self.prior_mean.expand_as(mean)
        prior_var = torch.tensor(self.prior_std ** 2, device=self.device)
        kl = 0.5 * torch.sum(
            torch.log(prior_var) - logvar +
            (torch.exp(logvar) + (mean - prior_mean).pow(2)) / prior_var - 1.0
        )
        return kl / mean.numel()

    def elbo(self, y: torch.Tensor, x_seq: torch.Tensor, p_seq: torch.Tensor) -> Dict:
        batch_size, seq_len, _ = y.shape
        x_mean, x_logvar = self.encoder(y)

        # KL for initial state
        kl_x0 = self.kl_divergence(x_mean[:, 0, :], x_logvar[:, 0, :])

        # KL for dynamics
        kl_dynamics = 0.0
        u_seq = self.compute_control_inputs(p_seq)
        for t in range(1, seq_len):
            q_mean_t = x_mean[:, t, :]
            q_logvar_t = x_logvar[:, t, :]
            x_prev = x_seq[:, t-1, :]
            u_prev = u_seq[:, t-1, :]
            Ax_prev = torch.matmul(self.A, x_prev.unsqueeze(-1)).squeeze(-1)
            Bu_prev = torch.matmul(self.B, u_prev.unsqueeze(-1)).squeeze(-1)
            p_mean_t = x_prev + self.dt * (Ax_prev + Bu_prev)
            p_logvar_t = torch.full_like(q_logvar_t, math.log(self.eps))
            kl_t = self.kl_divergence(q_mean_t, q_logvar_t, p_mean_t)
            kl_dynamics += kl_t
        kl_dynamics /= (seq_len - 1)

        # Reconstruction loss
        y_pred = torch.bmm(x_seq, self.C.t().unsqueeze(0).expand(batch_size, -1, -1))
        error = y - y_pred
        V_chol = self.V_chol
        Vn = V_chol @ V_chol.t()
        V_inv = torch.inverse(Vn + self.eps * torch.eye(self.y_dim, device=self.device))
        log_det_V = 2 * torch.sum(torch.log(torch.diagonal(V_chol) + self.eps))
        error_flat = error.reshape(-1, self.y_dim)
        quad_term = torch.sum(torch.matmul(error_flat, V_inv) * error_flat, dim=1).reshape(batch_size, seq_len)
        recon_loss = torch.mean(-0.5 * quad_term - 0.5 * log_det_V)

        # Dynamics penalty
        dynamics_penalty = 0.0
        for t in range(1, seq_len):
            x_t = x_seq[:, t, :]
            x_prev = x_seq[:, t-1, :]
            u_prev = u_seq[:, t-1, :]
            Ax_prev = torch.matmul(self.A, x_prev.unsqueeze(-1)).squeeze(-1)
            Bu_prev = torch.matmul(self.B, u_prev.unsqueeze(-1)).squeeze(-1)
            x_expected = x_prev + self.dt * (Ax_prev + Bu_prev)
            dynamics_error = x_t - x_expected
            dynamics_penalty += torch.mean(dynamics_error ** 2)
        dynamics_penalty = -self.lambda_dyn * dynamics_penalty / (seq_len - 1)

        # Control cost
        x_cost = torch.sum(torch.bmm(x_seq, self.Q.unsqueeze(0).expand(batch_size, -1, -1)) * x_seq, dim=(1, 2))
        u_cost = torch.sum(torch.bmm(u_seq, self.R.unsqueeze(0).expand(batch_size, -1, -1)) * u_seq, dim=(1, 2))
        control_cost = -self.gamma * torch.mean(x_cost + u_cost)

        elbo_terms = {
            'recon_loss': recon_loss,
            'kl_x0': self.beta * kl_x0,
            'kl_dynamics': self.beta * kl_dynamics,
            'dynamics_penalty': dynamics_penalty,
            'control_cost': control_cost
        }

        self.current_loss_components = {
            'recon_loss_raw': recon_loss.item(),
            'kl_x0_raw': kl_x0.item(),
            'kl_dynamics_raw': kl_dynamics.item(),
            'dynamics_penalty_raw': dynamics_penalty.item(),
            'control_cost_raw': control_cost.item()
        }

        return elbo_terms

    def forward(self, y: torch.Tensor) -> Dict[str, Any]:
        # Normalize input
        y_mean = y.mean(dim=(0, 1), keepdim=True)
        y_std = y.std(dim=(0, 1), keepdim=True)
        y_normalized = (y - y_mean) / (y_std + self.eps)

        batch_size, seq_len, _ = y_normalized.shape
        x_mean, x_logvar = self.encoder(y_normalized)
        x0 = self.sample_from_gaussian(x_mean[:, 0, :], x_logvar[:, 0, :])
        x_seq = self.propagate_state_sequence(x0, seq_len, y_observations=y_normalized)
        p_seq = self.derive_adjoint_states(x_seq, y_normalized)
        u_seq = self.compute_control_inputs(p_seq)
        y_pred = torch.bmm(x_seq, self.C.t().unsqueeze(0).expand(batch_size, -1, -1))

        elbo_terms = self.elbo(y_normalized, x_seq, p_seq)
        total_elbo = (elbo_terms['recon_loss'] - elbo_terms['kl_x0'] - elbo_terms['kl_dynamics'] +
                      elbo_terms['dynamics_penalty'] + elbo_terms['control_cost'])

        if batch_size == 1:
            x_seq = x_seq.squeeze(0)
            p_seq = p_seq.squeeze(0)
            u_seq = u_seq.squeeze(0)
            y_pred = y_pred.squeeze(0)

        grad_norms = self._compute_gradient_norms() if self.training else {}

        result = {
            'elbo': total_elbo.item(),
            'recon_loss': elbo_terms['recon_loss'].item(),
            'kl_x0': elbo_terms['kl_x0'].item(),
            'kl_dynamics': elbo_terms['kl_dynamics'].item(),
            'dynamics_penalty': elbo_terms['dynamics_penalty'].item(),
            'control_cost': elbo_terms['control_cost'].item(),
            'latent_states': x_seq.detach(),
            'adjoint_states': p_seq.detach(),
            'control_inputs': u_seq.detach(),
            'y_pred': y_pred.detach(),
            'grad_norms': grad_norms
        }

        if hasattr(self, 'current_loss_components'):
            result.update(self.current_loss_components)

        return result

    def _compute_gradient_norms(self) -> Dict[str, float]:
        grad_norms = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norms[name.split('.')[0]] = torch.norm(param.grad).item()
            else:
                grad_norms[name.split('.')[0]] = 0.0
        return grad_norms

    def fit(self,
            y: torch.Tensor,
            n_epochs: int = 100,
            learning_rate: float = 1e-4,
            validation_data: Optional[torch.Tensor] = None,
            verbose: bool = True) -> Dict[str, List[float]]:
        if y.dim() == 2:
            y = y.unsqueeze(0)
        if validation_data is not None and validation_data.dim() == 2:
            validation_data = validation_data.unsqueeze(0)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        history = {
            'elbo': [], 'recon_loss': [], 'kl_x0': [], 'kl_dynamics': [], 'dynamics_penalty': [], 'control_cost': [],
            'val_elbo': [] if validation_data is not None else None,
            'val_recon_loss': [], 'val_kl_x0': [], 'val_kl_dynamics': [], 'val_dynamics_penalty': [], 'val_control_cost': []
        } if validation_data is not None else {
            'elbo': [], 'recon_loss': [], 'kl_x0': [], 'kl_dynamics': [], 'dynamics_penalty': [], 'control_cost': []
        }

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            output = self.forward(y)
            total_elbo = torch.tensor(output['elbo'], requires_grad=True)
            (-total_elbo).backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
            optimizer.step()

            for key in ['elbo', 'recon_loss', 'kl_x0', 'kl_dynamics', 'dynamics_penalty', 'control_cost']:
                history[key].append(output[key])

            if validation_data is not None:
                with torch.no_grad():
                    val_output = self.forward(validation_data)
                    for key in ['elbo', 'recon_loss', 'kl_x0', 'kl_dynamics', 'dynamics_penalty', 'control_cost']:
                        history[f'val_{key}'].append(val_output[key])

            if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
                log_msg = f"Epoch {epoch+1}/{n_epochs}, ELBO: {output['elbo']:.4f}"
                log_msg += f"\n  Recon: {output['recon_loss']:.4f}, KL_x0: {output['kl_x0']:.4f}, KL_dyn: {output['kl_dynamics']:.4f}, Dyn_Penalty: {output['dynamics_penalty']:.4f}, Control: {output['control_cost']:.4f}"
                if validation_data is not None:
                    log_msg += f"\n  Val ELBO: {val_output['elbo']:.4f}"
                print(log_msg)

        return history

    def propagate_state_sequence(self, x0: torch.Tensor, seq_len: int,
                                y_observations: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x0.shape[0]
        x_list = [x0]
        x_t = x0.clone()

        # Preliminary propagation
        for _ in range(seq_len - 1):
            Ax = torch.matmul(self.A, x_t.unsqueeze(-1)).squeeze(-1)
            x_next = x_t + self.dt * Ax
            x_list.append(x_next)
            x_t = x_next

        x_seq_prelim = torch.stack(x_list, dim=1)
        y_target = y_observations if y_observations is not None else torch.zeros(batch_size, seq_len, self.y_dim, device=self.device)
        p_seq = self.derive_adjoint_states(x_seq_prelim, y_target)
        u_seq = self.compute_control_inputs(p_seq)

        # Refined propagation
        x_list = [x0]
        x_t = x0.clone()
        for t in range(seq_len - 1):
            u_t = u_seq[:, t, :]
            Ax = torch.matmul(self.A, x_t.unsqueeze(-1)).squeeze(-1)
            Bu = torch.matmul(self.B, u_t.unsqueeze(-1)).squeeze(-1)
            x_next = x_t + self.dt * (Ax + Bu)
            x_list.append(x_next)
            x_t = x_next

        return torch.stack(x_list, dim=1)

    def derive_adjoint_states(self, x_seq: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x_seq.shape
        y_pred = torch.bmm(x_seq, self.C.t().unsqueeze(0).expand(batch_size, -1, -1))
        obs_error = y - y_pred

        p_list = [torch.zeros(batch_size, self.x_dim, device=self.device) for _ in range(seq_len)]
        C_expanded = self.C.unsqueeze(0).expand(batch_size, -1, -1)

        p_list[-1] = (torch.bmm(obs_error[:, -1, :].unsqueeze(1), C_expanded).squeeze(1) +
                      torch.matmul(self.Q, x_seq[:, -1, :].unsqueeze(-1)).squeeze(-1))

        A_expanded = self.A.unsqueeze(0).expand(batch_size, -1, -1)
        Q_expanded = self.Q.unsqueeze(0).expand(batch_size, -1, -1)
        for t in range(seq_len - 2, -1, -1):
            p_next = p_list[t+1]
            A_term = torch.bmm(p_next.unsqueeze(1), A_expanded).squeeze(1)
            C_term = torch.bmm(obs_error[:, t, :].unsqueeze(1), C_expanded).squeeze(1)
            Q_term = torch.bmm(x_seq[:, t, :].unsqueeze(1), Q_expanded).squeeze(1)
            p_list[t] = p_next - self.dt * (A_term + C_term + Q_term)

        return torch.stack(p_list, dim=1)

    def compute_control_inputs(self, p_seq: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = p_seq.shape
        R_inv = torch.inverse(self.R + self.eps * torch.eye(self.u_dim, device=self.device))
        u_seq = -torch.bmm(
            p_seq.reshape(-1, 1, self.x_dim),
            torch.matmul(self.B, R_inv).t().unsqueeze(0).expand(batch_size * seq_len, -1, -1)
        ).reshape(batch_size, seq_len, self.u_dim)
        return u_seq

    def save(self, file_path: str) -> None:
        torch.save({
            'model_state': self.state_dict(),
            'model_config': {
                'x_dim': self.x_dim, 'y_dim': self.y_dim, 'u_dim': self.u_dim,
                'beta': self.beta, 'prior_std': self.prior_std, 'dt': self.dt,
                'eps': self.eps, 'encoder_h_dim': self.encoder.h_dim, 'gamma': self.gamma, 'lambda_dyn': self.lambda_dyn
            }
        }, file_path)

    @classmethod
    def load(cls, file_path: str, device: Optional[torch.device] = None) -> 'CoupledStateSpaceVI':
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(file_path, map_location=device)
        config = state_dict['model_config']

        file_dir = Path(file_path).parent.parent.parent
        leadfield_path = file_dir / "leadfield" / "sub-001_ses-t1_leadfield.npy"

        if leadfield_path.exists():
            leadfield = np.load(leadfield_path)
            leadfield = np.nan_to_num(leadfield, nan=0.0, posinf=1e6, neginf=-1e6)
            n_components = min(config['x_dim'], min(leadfield.shape) - 1)
            U, s, Vh = svds(leadfield, k=n_components)
            idx = np.argsort(s)[::-1]
            C = U[:, idx] @ np.diag(s[idx])
            V_tensor = torch.tensor(Vh[idx, :].T, dtype=torch.float32)
        else:
            print(f"Warning: Leadfield not found at {leadfield_path}. Using identity matrix.")
            C = np.eye(config['y_dim'], config['x_dim'])
            V_tensor = torch.eye(config['x_dim'], config['y_dim'])

        model = cls(
            x_dim=config['x_dim'], y_dim=config['y_dim'], C=C, u_dim=config['u_dim'],
            beta=config['beta'], prior_std=config['prior_std'], dt=config['dt'],
            eps=config['eps'], gamma=config.get('gamma', 0.001), lambda_dyn=config.get('lambda_dyn', 100.0),
            encoder_h_dim=config.get('encoder_h_dim', 128), device=device
        )
        model.register_buffer('V', V_tensor)
        model.load_state_dict(state_dict['model_state'])
        return model

    def predict(self, y: torch.Tensor, project_to_sources: bool = False) -> Dict[str, torch.Tensor]:
        if y.dim() == 2:
            y = y.unsqueeze(0)
        with torch.no_grad():
            output = self.forward(y)
            results = {
                'y_pred': output['y_pred'],
                'x': output['latent_states'],
                'p': output['adjoint_states'],
                'u': output['control_inputs']
            }
            if project_to_sources and hasattr(self, 'V'):
                x_flat = output['latent_states'].reshape(-1, self.x_dim)
                source_activity = torch.matmul(x_flat, self.V.T).reshape(y.shape[0], -1, self.V.shape[1])
                results['source_activity'] = source_activity
            return results

# Example usage
if __name__ == "__main__":
    # Dummy data
    x_dim, y_dim, seq_len = 10, 20, 100
    C = np.random.randn(y_dim, x_dim)
    y = torch.randn(1, seq_len, y_dim)

    # Initialize model
    model = CoupledStateSpaceVI(x_dim=x_dim, y_dim=y_dim, C=C, gamma=0.001, lambda_dyn=100.0)

    # Train
    history = model.fit(y, n_epochs=50, learning_rate=1e-4, verbose=True)

    # Predict
    preds = model.predict(y)
    print("Predictions:", preds['y_pred'].shape)