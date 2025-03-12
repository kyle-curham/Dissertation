"""
State Space Model with Optimal Control and System Identification
Learns A, B, Q, R from observations using continuous dynamics and LQR with your forward symplectic Riccati solver.
No variational inference—just Kalman filtering and likelihood maximization.

This script uses real EEG data for training and evaluation.
"""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import mne
from mne.io import read_raw_edf
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from scipy.sparse.linalg import svds
import math

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eeg_processing.models.control.riccati_solver import RiccatiSolver
from eeg_processing.utils.plotting import (
    plot_training_results, plot_matrix_analysis, 
    plot_training_history, plot_prediction_comparison, plot_latent_dynamics
)

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class CoupledStateSpace(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, C: np.ndarray, u_dim: Optional[int] = None, dt: float = 1/1024, eps: float = 1e-4, device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.u_dim = u_dim if u_dim is not None else x_dim
        self.dt = dt
        self.eps = eps
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        A_init = torch.zeros(x_dim, x_dim)
        for i in range(x_dim):
            A_init[i, i] = -0.01 * (i + 1)
        self.A = nn.Parameter(A_init)
        self.B = nn.Parameter(torch.eye(x_dim, self.u_dim) * 0.1)
        self.register_buffer('C', torch.tensor(C, dtype=torch.float32, device=self.device))
        self.L_Q = nn.Parameter(torch.eye(x_dim, device=self.device) * 0.316)
        self.L_R = nn.Parameter(torch.eye(self.u_dim, device=self.device) * 0.316)
        self.L_V = nn.Parameter(torch.eye(y_dim, device=self.device))
        self.L_Q_proc = nn.Parameter(torch.eye(x_dim, device=self.device) * 0.1)
        self.register_buffer('x0_mean', torch.zeros(x_dim, device=self.device))
        self.register_buffer('P0', torch.eye(x_dim, device=self.device) * 0.1)
        self.riccati_solver = RiccatiSolver(device=self.device)

    @property
    def Q(self) -> torch.Tensor:
        return self.L_Q @ self.L_Q.t()

    @property
    def R(self) -> torch.Tensor:
        return self.L_R @ self.L_R.t()

    @property
    def V(self) -> torch.Tensor:
        return self.L_V @ self.L_V.t()

    @property
    def Q_proc(self) -> torch.Tensor:
        return self.L_Q_proc @ self.L_Q_proc.t()

    def compute_riccati_solution(self, seq_len: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        P0 = self.Q.clone()
        Q_psd = self.Q
        R_psd = self.R
        K_seq, P_seq = [], []
        P_t = P0
        S_t = torch.cat([torch.eye(self.x_dim, device=self.device), P0], dim=0)
        self.riccati_solver.n = self.x_dim
        self.riccati_solver.H = self.riccati_solver.build_augmented_system(self.A, self.B, Q_psd, R_psd)
        for t in range(seq_len):
            X_new, S_t = self.riccati_solver.step(S_t, self.dt)
            P_t = X_new
            P_t = 0.5 * (P_t + P_t.t())
            BtP = torch.matmul(self.B.t(), P_t)
            R_reg = R_psd + self.eps * torch.eye(self.u_dim, device=self.device)
            R_inv = torch.inverse(R_reg)
            K_t = torch.matmul(R_inv, BtP)
            K_seq.append(K_t)
            P_seq.append(P_t)
        return P_seq, K_seq

    def kalman_filter(self, y: torch.Tensor, K_seq: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = y.shape
        x_filt = torch.zeros(batch_size, seq_len, self.x_dim, device=self.device)
        P_filt = torch.zeros(batch_size, seq_len, self.x_dim, self.x_dim, device=self.device)
        x_t = self.x0_mean.expand(batch_size, -1)
        P_t = self.P0
        log_likelihood = 0.0
        Q_proc = self.Q_proc
        V_inv = torch.inverse(self.V + self.eps * torch.eye(self.y_dim, device=self.device))
        L_V = torch.linalg.cholesky(self.V + self.eps * torch.eye(self.y_dim, device=self.device))
        log_det_V = 2 * torch.sum(torch.log(torch.diagonal(L_V)))
        
        # Symplectic setup: Initialize S_t once at the start
        self.riccati_solver.n = self.x_dim
        S_t = torch.cat([torch.eye(self.x_dim, device=self.device), P_t], dim=0)
        
        for t in range(seq_len):
            K_t = K_seq[t]
            A_eff = self.A - torch.matmul(self.B, K_t)
            x_dot = torch.matmul(A_eff, x_t.unsqueeze(-1)).squeeze(-1)
            x_pred = x_t + self.dt * x_dot
            
            # Build augmented system for Lyapunov: H = [A_eff, Q_proc; 0, -A_eff^T]
            H = torch.zeros(2 * self.x_dim, 2 * self.x_dim, device=self.device)
            H[:self.x_dim, :self.x_dim] = A_eff
            H[:self.x_dim, self.x_dim:] = Q_proc
            H[self.x_dim:, self.x_dim:] = -A_eff.t()
            self.riccati_solver.H = H
            
            # Evolve S_t with symplectic step
            P_new, S_t = self.riccati_solver.step(S_t, self.dt)
            P_pred = P_new
            P_pred = 0.5 * (P_pred + P_pred.t())  # Ensure symmetry
            
            y_t = y[:, t, :]
            y_pred = torch.matmul(self.C, x_pred.unsqueeze(-1)).squeeze(-1)
            residual = y_t - y_pred
            CtP = torch.matmul(self.C, P_pred)
            S = torch.matmul(CtP, self.C.t()) + self.V
            S_inv = torch.inverse(S + self.eps * torch.eye(self.y_dim, device=self.device))
            K = torch.matmul(CtP.transpose(-1, -2), S_inv)
            x_t = x_pred + torch.matmul(K, residual.unsqueeze(-1)).squeeze(-1)
            P_t = P_pred - torch.matmul(K, torch.matmul(self.C, P_pred))
            x_filt[:, t, :] = x_t
            P_filt[:, t, :, :] = P_t
            quad_term = torch.sum(torch.matmul(residual.unsqueeze(-1).transpose(-1, -2), torch.matmul(S_inv, residual.unsqueeze(-1))))
            log_likelihood += -0.5 * (quad_term + log_det_V + self.y_dim * math.log(2 * math.pi))
        log_likelihood /= seq_len

        return x_filt, log_likelihood

    def forward(self, y: torch.Tensor) -> Dict[str, Any]:
        y_mean = y.mean(dim=(0, 1), keepdim=True)
        y_std = y.std(dim=(0, 1), keepdim=True)
        y_normalized = (y - y_mean) / (y_std + self.eps)
        batch_size, seq_len, _ = y_normalized.shape
        P_seq, K_seq = self.compute_riccati_solution(seq_len)
        x_filt, log_likelihood = self.kalman_filter(y_normalized, K_seq)
        u_seq = torch.zeros(batch_size, seq_len, self.u_dim, device=self.device)
        for t in range(seq_len):
            u_seq[:, t, :] = -torch.matmul(K_seq[t], x_filt[:, t, :].unsqueeze(-1)).squeeze(-1)
        y_pred = torch.bmm(x_filt, self.C.t().unsqueeze(0).expand(batch_size, -1, -1))
        x_cost = torch.sum(torch.bmm(x_filt, self.Q.unsqueeze(0).expand(batch_size, -1, -1)) * x_filt, dim=(1, 2))
        u_cost = torch.sum(torch.bmm(u_seq, self.R.unsqueeze(0).expand(batch_size, -1, -1)) * u_seq, dim=(1, 2))
        control_cost = torch.mean(x_cost + u_cost)
        result = {
            'log_likelihood': log_likelihood,
            'control_cost': control_cost,
            'log_likelihood_value': log_likelihood.item(),
            'control_cost_value': control_cost.item() if not torch.isnan(control_cost) else 0.0,
            'states': x_filt.detach(),
            'controls': u_seq.detach(),
            'y_pred': y_pred.detach(),
            'y_normalized': y_normalized.detach()
        }
        return result

    def fit(self, y: torch.Tensor, n_epochs: int = 100, learning_rate: float = 1e-3, validation_data: Optional[torch.Tensor] = None, verbose: bool = True) -> Dict[str, List[float]]:
        if y.dim() == 2:
            y = y.unsqueeze(0)
        if validation_data is not None and validation_data.dim() == 2:
            validation_data = validation_data.unsqueeze(0)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        history = {
            'log_likelihood': [], 'control_cost': [],
            'val_log_likelihood': [] if validation_data is not None else None,
            'val_control_cost': [] if validation_data is not None else None
        }
        plot_epochs = [1, n_epochs // 2, n_epochs]
        predictions = {}
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            output = self.forward(y)
            loss = -output['log_likelihood']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            history['log_likelihood'].append(output['log_likelihood_value'])
            history['control_cost'].append(output['control_cost_value'])
            if validation_data is not None:
                with torch.no_grad():
                    val_output = self.forward(validation_data)
                    history['val_log_likelihood'].append(val_output['log_likelihood_value'])
                    history['val_control_cost'].append(val_output['control_cost_value'])
            if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
                log_msg = f"Epoch {epoch+1}/{n_epochs}, Log-Likelihood: {output['log_likelihood_value']:.4f}, Control Cost: {output['control_cost_value']:.4f}"
                if validation_data is not None:
                    log_msg += f"\n  Val Log-Likelihood: {val_output['log_likelihood_value']:.4f}, Val Control Cost: {val_output['control_cost_value']:.4f}"
                print(log_msg)
            if epoch + 1 in plot_epochs:
                predictions[epoch + 1] = {
                    'train_pred': output['y_pred'].cpu().numpy(),
                    'train_true': output['y_normalized'].cpu().numpy()
                }
                if validation_data is not None:
                    with torch.no_grad():
                        val_pred_output = self.forward(validation_data)
                        predictions[epoch + 1]['val_pred'] = val_pred_output['y_pred'].cpu().numpy()
                        predictions[epoch + 1]['val_true'] = val_pred_output['y_normalized'].cpu().numpy()
        fig = plot_training_results(
            output=output,
            history=history,
            predictions=predictions,
            n_epochs=n_epochs,
            plot_epochs=plot_epochs,
            y=y,
            validation_data=validation_data
        )
        fig.savefig('training_results.png')
        plt.show()
        return history

    def save(self, file_path: str) -> None:
        torch.save({
            'model_state': self.state_dict(),
            'model_config': {'x_dim': self.x_dim, 'y_dim': self.y_dim, 'u_dim': self.u_dim, 'dt': self.dt, 'eps': self.eps}
        }, file_path)

    def predict(self, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        if y.dim() == 2:
            y = y.unsqueeze(0)
        with torch.no_grad():
            output = self.forward(y)
            return {'y_pred': output['y_pred'], 'states': output['states'], 'controls': output['controls']}

def parse_args():
    parser = argparse.ArgumentParser(description='Run Coupled State Space model for EEG data')
    parser.add_argument('--subject', type=str, default='sub-001', help='Subject ID (default: sub-001)')
    parser.add_argument('--session', type=str, default='ses-t1', help='Session ID (default: ses-t1)')
    parser.add_argument('--task', type=str, default='resteyesc', help='Task ID (default: resteyesc)')
    parser.add_argument('--x_dim', type=int, default=8, help='Latent state dimension (default: 8)')
    parser.add_argument('--timepoints', type=int, default=5000, help='Number of timepoints to use (default: 2000)')
    parser.add_argument('--training_epochs', type=int, default=100, help='Number of training epochs (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    return parser.parse_args()

def main():
    args = parse_args()
    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    sub_id = args.subject
    ses_id = args.session
    task_id = args.task
    data_root = project_root / "data"
    leadfield_dir = project_root / "leadfield"
    output_dir = project_root / "eeg_processing" / "results"
    os.makedirs(output_dir, exist_ok=True)
    edf_file = data_root / sub_id / ses_id / "eeg" / f"{sub_id}_{ses_id}_task-{task_id}_eeg.edf"
    print(f"Loading continuous EEG data from: {edf_file}")
    raw = read_raw_edf(edf_file, preload=True)
    print(f"Raw data loaded: {len(raw.ch_names)} channels, {raw.n_times} time points")
    raw_data = raw.get_data()
    raw_data = raw_data * 1e3
    print(f"Converting EEG data from volts to microvolts (μV)")
    eeg_data = raw_data.T
    timepoints = min(args.timepoints, eeg_data.shape[0])
    eeg_data = eeg_data[:timepoints, :]
    print(f"Using {timepoints} timepoints of EEG data")
    eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32, device=device)
    print(f"EEG data shape: {eeg_tensor.shape}")
    leadfield_file = leadfield_dir / f"{sub_id}_{ses_id}_leadfield.npy"
    leadfield = np.load(leadfield_file)
    print(f"Leadfield matrix shape: {leadfield.shape}")
    leadfield = leadfield * 1e-3
    print(f"Converting leadfield to μV/(nAm) for unit consistency")
    y_dim = eeg_tensor.shape[1]
    x_dim = args.x_dim
    print(f"Using latent state dimension: {x_dim}")
    print(f"Observation dimension: {y_dim}")
    if np.any(np.isnan(leadfield)):
        print("Warning: Lead field matrix contains NaN values")
        leadfield = np.nan_to_num(leadfield, nan=0.0)
    if np.any(np.isinf(leadfield)):
        print("Warning: Lead field matrix contains infinite values")
        leadfield = np.nan_to_num(leadfield, posinf=1e6, neginf=-1e6)
    n_components = min(x_dim, min(leadfield.shape) - 1)
    U, s, Vh = svds(leadfield, k=n_components)
    idx = np.argsort(s)[::-1]
    s = s[idx]
    U = U[:, idx]
    Vh = Vh[idx, :]
    C_downsampled = U @ np.diag(s)
    V = Vh.T
    explained_variance = np.sum(s**2) / np.sum(leadfield.flatten()**2) * 100
    print(f"\nDimensionality reduction:")
    print(f"Original leadfield dimensions: {leadfield.shape}")
    print(f"Reduced leadfield dimensions: {C_downsampled.shape}")
    print(f"Number of components kept: {n_components}")
    print(f"Variance explained: {explained_variance:.2f}%")
    total_timepoints = eeg_tensor.shape[0]
    val_size = int(0.2 * total_timepoints)
    train_size = total_timepoints - val_size
    train_data = eeg_tensor[:train_size]
    val_data = eeg_tensor[train_size:]
    print(f"Training data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    model = CoupledStateSpace(
        x_dim=x_dim,
        y_dim=y_dim,
        C=C_downsampled,
        u_dim=x_dim,
        dt=1/raw.info['sfreq'],
        device=device
    )
    print(f"\nTraining model for {args.training_epochs} epochs...")
    history = model.fit(
        y=train_data,
        n_epochs=args.training_epochs,
        learning_rate=args.learning_rate,
        validation_data=val_data,
        verbose=True
    )
    print("\nRunning forward pass on validation data...")
    pred_output = model.predict(val_data)
    predicted_obs = pred_output['y_pred']
    latent_states = pred_output['states']
    controls = pred_output['controls']
    print("\nPrediction results:")
    print(f"Predicted observations shape: {predicted_obs.shape}")
    print(f"Latent states shape: {latent_states.shape}")
    print(f"Control inputs shape: {controls.shape}")
    A = model.A.detach().cpu().numpy()
    B = model.B.detach().cpu().numpy()
    Q = model.Q.detach().cpu().numpy()
    R = model.R.detach().cpu().numpy()
    print("\nLearned model parameters:")
    print(f"A shape: {A.shape}")
    print(f"B shape: {B.shape}")
    print(f"Q shape: {Q.shape}")
    print(f"R shape: {R.shape}")
    eigenvalues = np.linalg.eigvals(A)
    print("\nEigenvalues of A:")
    print(f"Min real part: {np.min(eigenvalues.real):.4f}")
    print(f"Max real part: {np.max(eigenvalues.real):.4f}")
    print(f"Number of unstable modes: {np.sum(eigenvalues.real > 0)}")
    train_pred = model.predict(train_data)
    val_pred = model.predict(val_data)
    matrix_fig = plot_matrix_analysis(
        A=A,
        B=B,
        train_data=train_data.cpu().numpy(),
        train_pred=train_pred['y_pred'].cpu().numpy(),
        val_data=val_data.cpu().numpy(),
        val_pred=val_pred['y_pred'].cpu().numpy(),
        channel_idx=0
    )
    matrix_fig.savefig(output_dir / f"{sub_id}_matrix_analysis.png")
    plt.show()
    model_save_path = output_dir / f"{sub_id}_coupled_state_space_model.pt"
    model.save(str(model_save_path))
    print(f"Model saved to {model_save_path}")
    matrices_save_path = output_dir / f"{sub_id}_learned_matrices.npz"
    np.savez(
        matrices_save_path,
        A=A,
        B=B,
        C=model.C.detach().cpu().numpy(),
        Q=Q,
        R=R,
        V=V,
        Q_proc=model.Q_proc.detach().cpu().numpy()
    )
    print(f"Learned matrices saved to {matrices_save_path}")
    return model

if __name__ == "__main__":
    model = main()