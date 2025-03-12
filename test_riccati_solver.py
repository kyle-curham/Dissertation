import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Adjust import path as needed for your project structure
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eeg_processing.models.control.riccati_solver import RiccatiSolver

def create_test_matrices(n, device):
    """
    Create synthetic test matrices for testing the Riccati solver.
    
    Args:
        n (int): Dimension of state space
        device (torch.device): Device to create tensors on
        
    Returns:
        tuple: A, B, Q, R, X_current matrices for testing
    """
    # Create a stable system matrix A (with eigenvalues having negative real parts)
    A_np = np.random.randn(n, n)
    A_np = (A_np - A_np.T) - 0.5 * np.eye(n)  # Make it stable and slightly asymmetric
    A = torch.tensor(A_np, dtype=torch.float32, device=device)
    
    # Create input matrix B
    B_np = np.random.randn(n, n//2)
    B = torch.tensor(B_np, dtype=torch.float32, device=device)
    
    # Create positive definite Q matrix (state cost)
    Q_np = np.random.randn(n, n)
    Q_np = Q_np @ Q_np.T + np.eye(n)  # Ensure positive definiteness
    Q = torch.tensor(Q_np, dtype=torch.float32, device=device)
    
    # Create positive definite R matrix (control cost)
    R_np = np.random.randn(n//2, n//2)
    R_np = R_np @ R_np.T + 2 * np.eye(n//2)  # Ensure positive definiteness
    R = torch.tensor(R_np, dtype=torch.float32, device=device)
    
    # Create positive definite initial X
    X_np = np.random.randn(n, n)
    X_np = X_np @ X_np.T + np.eye(n)  # Ensure positive definiteness
    X_current = torch.tensor(X_np, dtype=torch.float32, device=device)
    
    return A, B, Q, R, X_current

def create_hamiltonian_matrix(A, B, Q, R, device):
    """
    Create a Hamiltonian matrix for Riccati equation directly.
    
    For the Riccati equation, the correct Hamiltonian is:
    
    H = [  A           -B·R⁻¹·B^T  ]
        [ -Q           -A^T        ]
    
    Args:
        A (torch.Tensor): System dynamics matrix
        B (torch.Tensor): Input matrix
        Q (torch.Tensor): State cost matrix (must be symmetric)
        R (torch.Tensor): Control cost matrix (must be symmetric positive definite)
        device (torch.device): Device to create tensors on
        
    Returns:
        torch.Tensor: Hamiltonian matrix (2n × 2n)
    """
    n = A.shape[0]
    
    # Ensure Q and R are symmetric
    Q = 0.5 * (Q + Q.transpose(-1, -2))
    R = 0.5 * (R + R.transpose(-1, -2))
    
    # Compute B·R⁻¹·B^T
    try:
        R_inv = torch.linalg.inv(R)
        BR_inv_BT = torch.matmul(B, torch.matmul(R_inv, B.transpose(-1, -2)))
    except Exception as e:
        print(f"Error computing R_inv or BR_inv_BT: {str(e)}")
        # If R is ill-conditioned, use pseudoinverse
        print("Using pseudoinverse for R")
        R_inv = torch.linalg.pinv(R)
        BR_inv_BT = torch.matmul(B, torch.matmul(R_inv, B.transpose(-1, -2)))
    
    # Top row of the Hamiltonian
    top_row = torch.cat([A, -BR_inv_BT], dim=1)
    
    # Bottom row of the Hamiltonian
    bottom_row = torch.cat([-Q, -A.transpose(-1, -2)], dim=1)
    
    # Full Hamiltonian
    H = torch.cat([top_row, bottom_row], dim=0)
    
    # Verify it's Hamiltonian (J·H should be symmetric)
    J = torch.zeros((2*n, 2*n), device=device)
    J[:n, n:] = torch.eye(n, device=device)
    J[n:, :n] = -torch.eye(n, device=device)
    
    JH = torch.matmul(J, H)
    is_hamiltonian = torch.allclose(JH, JH.transpose(-1, -2), rtol=1e-5, atol=1e-5)
    
    if not is_hamiltonian:
        print("Warning: Constructed matrix is not precisely Hamiltonian")
        diff = torch.norm(JH - JH.transpose(-1, -2)).item()
        print(f"Asymmetry in J·H: {diff:.6e}")
    else:
        print("Verified: Matrix is correctly Hamiltonian")
    
    return H

def test_symplectic_integrator_step():
    """
    Test the symplectic_integrator_step method directly using a constructed
    Hamiltonian matrix for the Riccati equation.
    """
    print("\n===== Testing symplectic_integrator_step method =====")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize the solver
    solver = RiccatiSolver(device=device)
    
    # Create test matrices
    n = 4  # State dimension
    A, B, Q, R, X_current = create_test_matrices(n, device)
    
    # Create proper Hamiltonian matrix for Riccati equation
    H = create_hamiltonian_matrix(A, B, Q, R, device)
    
    # Time step
    h = 0.01
    
    print(f"Hamiltonian matrix shape: {H.shape}")
    print(f"X_current shape: {X_current.shape}")
    
    # Test single step integration
    try:
        # Create the augmented state vector Z = [Z1; Z2] where Z1 = I and Z2 = X_current
        n = X_current.shape[0]
        Z1 = torch.eye(n, device=device)
        Z2 = X_current
        Z_current = torch.cat([Z1, Z2], dim=0)
        
        print(f"Z_current shape: {Z_current.shape}")
        
        # Call symplectic_integrator_step which now returns both X_new and Z_new
        X_new, Z_new = solver.symplectic_integrator_step(H, Z_current, h)
        
        print(f"X_new shape: {X_new.shape}")
        print(f"Z_new shape: {Z_new.shape}")
        
        # Check properties of X_new
        sym_diff = torch.norm(X_new - X_new.T).item()
        print(f"Symmetry error: {sym_diff:.6e}")
        
        # Check positive definiteness
        try:
            eigvals = torch.linalg.eigvals(X_new)
            min_eigval = torch.min(eigvals.real).item()
            print(f"Minimum eigenvalue: {min_eigval:.6e}")
            if min_eigval < 0:
                print("Warning: Result has negative eigenvalues")

        except Exception as e:
            print(f"Could not compute eigenvalues: {str(e)}")
        
        # Verify that Z_new has the expected structure
        Z1_new = Z_new[:n, :]
        Z2_new = Z_new[n:, :]
        reconstructed_X = torch.matmul(Z2_new, torch.inverse(Z1_new))
        reconstr_error = torch.norm(X_new - reconstructed_X).item()
        print(f"Z verification error: {reconstr_error:.6e}")
            
        print("Step integration successful!")
        
    except Exception as e:
        print(f"Error during integration step: {str(e)}")
    
    return H, X_current, Z_current, X_new if 'X_new' in locals() else None, Z_new if 'Z_new' in locals() else None

def test_symplectic_integrator():
    """
    Test the symplectic integrator by directly using the step method.
    This avoids calling the high-level symplectic_integrator method which might be incorrect.
    """
    print("\n===== Testing Riccati Integration with Step Method =====")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize the solver
    solver = RiccatiSolver(device=device)
    
    # Create synthetic test data
    n = 4  # State dimension
    A, B, Q, R, X_current = create_test_matrices(n, device)
    
    # Time step and number of steps
    h = 0.001
    num_steps = 200
    
    print(f"System dimensions: {n}x{n}")
    print(f"Time step h: {h}")
    print(f"Number of integration steps: {num_steps}")
    
    # Store results for visualization
    X_history = [X_current.detach().cpu().numpy()]
    
    # Build the Hamiltonian matrix - this is the key correction
    try:
        # Option 1: Create using our helper function
        H_external = create_hamiltonian_matrix(A, B, Q, R, device)
        print(f"External Hamiltonian shape: {H_external.shape}")
        
        # Option 2: Use solver's built-in method (if available)
        try:
            H_solver = solver.build_augmented_system(A, B, Q, R)
            print(f"Solver Hamiltonian shape: {H_solver.shape}")
            
            # Compare the two matrices
            diff = torch.norm(H_external - H_solver).item()
            print(f"Difference between Hamiltonians: {diff:.6e}")
            
            # Use solver's Hamiltonian if it looks correct
            H = H_solver if diff < 1e-5 else H_external
        except Exception as e:
            print(f"Error building solver's Hamiltonian: {str(e)}")
            print("Using manually constructed Hamiltonian")
            H = H_external
    except Exception as e:
        print(f"Error constructing Hamiltonian: {str(e)}")
        return None
    
    # Create the initial augmented state Z = [Z1; Z2] where Z1 = I and Z2 = X_current
    Z1 = torch.eye(n, device=device)
    Z2 = X_current
    Z_current = torch.cat([Z1, Z2], dim=0)
    
    print(f"Initial Z_current shape: {Z_current.shape}")
    
    # Run the integration using symplectic_integrator_step
    print("\nRunning integration with symplectic_integrator_step:")
    for i in range(num_steps):
        try:
            # Ensure H and Z_current are of type float, not double
            H = H.to(dtype=torch.float)
            Z_current = Z_current.to(dtype=torch.float)
            
            # Use the step method directly with constructed Hamiltonian
            X_new, Z_new = solver.symplectic_integrator_step(H, Z_current, h)
            
            # Check for NaNs or infinities
            if torch.isnan(X_new).any() or torch.isinf(X_new).any():
                print(f"Step {i}: Error - NaN or Inf values detected")
                break
            
            # Check and enforce symmetry (important for stability)
            sym_diff = torch.norm(X_new - X_new.T).item()
            if sym_diff > 1e-5:
                print(f"Step {i}: Warning - Result not symmetric (diff: {sym_diff:.6f})")
            
            # Check and enforce positive definiteness
            try:
                eigvals = torch.linalg.eigvals(X_new)
                min_eigval = torch.min(eigvals.real).item()
                if min_eigval < -1e-6:
                    print(f"Step {i}: Warning - Result has negative eigenvalue: {min_eigval:.6e}")

            except Exception as e:
                print(f"Step {i}: Could not check eigenvalues - {str(e)}")
            
            # Update and store
            X_current = X_new
            Z_current = Z_new
            X_history.append(X_current.detach().cpu().numpy())
            
            if i % 5 == 0:
                print(f"Step {i}: Integration successful")
                
        except Exception as e:
            print(f"Step {i}: Error - {str(e)}")
            break
    
    # Compute and print metrics
    if len(X_history) > 1:
        # Determinant ratio (should be preserved by symplectic methods)
        initial_det = np.linalg.det(X_history[0])
        final_det = np.linalg.det(X_history[-1])
        det_ratio = final_det / initial_det
        
        # Trace evolution
        initial_trace = np.trace(X_history[0])
        final_trace = np.trace(X_history[-1])
        
        print("\nResults Summary:")
        print(f"Initial determinant: {initial_det:.6e}")
        print(f"Final determinant: {final_det:.6e}")
        print(f"Determinant ratio: {det_ratio:.6f} (ideally close to 1.0)")
        print(f"Initial trace: {initial_trace:.6f}")
        print(f"Final trace: {final_trace:.6f}")
        
        # Plot results
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Trace evolution
        plt.subplot(1, 3, 1)
        traces = [np.trace(X) for X in X_history]
        plt.plot(traces)
        plt.title('Trace Evolution')
        plt.xlabel('Step')
        plt.ylabel('Trace')
        
        # Plot 2: Determinant evolution
        plt.subplot(1, 3, 2)
        determinants = [np.linalg.det(X) for X in X_history]
        plt.plot([d/determinants[0] for d in determinants])  # Normalized
        plt.title('Normalized Determinant')
        plt.xlabel('Step')
        plt.ylabel('Det/Det₀')
        
        # Plot 3: Eigenvalue evolution
        plt.subplot(1, 3, 3)
        min_eigenvalues = []
        for X in X_history:
            eigvals = np.linalg.eigvals(X)
            min_eigenvalues.append(np.min(np.real(eigvals)))
        plt.plot(min_eigenvalues)
        plt.title('Minimum Eigenvalue')
        plt.xlabel('Step')
        plt.ylabel('λ_min')
        
        plt.tight_layout()
        plt.savefig('symplectic_integrator_test.png')
        print("Results plotted and saved to 'symplectic_integrator_test.png'")
        
    return X_history

def test_symplectic_integration_stability():
    """
    Test the stability of the symplectic integrator over many time steps,
    tracking the condition number and determinant of Z1 to identify when and
    why numerical instability occurs.
    """
    print("\n===== Testing Symplectic Integration Stability =====")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize the solver
    solver = RiccatiSolver(device=device)
    
    # Create synthetic test data
    n = 4  # State dimension
    A, B, Q, R, X_current = create_test_matrices(n, device)
    
    # Experiment with different time steps
    time_steps = [0.01, 0.005, 0.001]
    
    for h in time_steps:
        print(f"\nTesting with time step h = {h}")
        
        # Build the Hamiltonian matrix
        H = create_hamiltonian_matrix(A, B, Q, R, device)
        
        # Initial augmented state
        Z1 = torch.eye(n, device=device)
        Z2 = X_current.clone()
        Z_current = torch.cat([Z1, Z2], dim=0)
        
        # Store diagnostic information
        num_steps = 100
        det_Z1_history = []
        cond_Z1_history = []
        symplectic_error = []
        
        # Run the integration with detailed diagnostics
        for i in range(num_steps):
            try:
                # Ensure correct data type
                H = H.to(dtype=torch.float)
                Z_current = Z_current.to(dtype=torch.float)
                
                # Compute one step
                X_new, Z_new = solver.symplectic_integrator_step(H, Z_current, h)
                
                # Extract Z1
                Z1_new = Z_new[:n, :]
                Z2_new = Z_new[n:, :]
                
                # Check determinant of Z1
                det_Z1 = torch.linalg.det(Z1_new).item()
                det_Z1_history.append(det_Z1)
                
                # Check condition number of Z1
                try:
                    # Computing condition number can be expensive and unstable
                    # for nearly singular matrices
                    sing_vals = torch.linalg.svdvals(Z1_new)
                    cond_Z1 = (sing_vals[0] / sing_vals[-1]).item() if sing_vals[-1] > 1e-10 else float('inf')
                    cond_Z1_history.append(cond_Z1)
                except Exception as e:
                    print(f"Step {i}: SVD computation failed - {str(e)}")
                    cond_Z1_history.append(float('inf'))
                
                # Check symplectic property preservation
                # For a symplectic matrix, we should have M^T J M = J
                # where J is the symplectic form
                full_dim = 2*n
                J = torch.zeros((full_dim, full_dim), device=device)
                J[:n, n:] = torch.eye(n, device=device)
                J[n:, :n] = -torch.eye(n, device=device)
                
                # Reconstruct the evolution matrix from Z_current to Z_new
                # This is an approximation as we don't have the exact matrix
                # We're checking if Z_new maintains the symplectic structure
                sym_error = torch.norm(Z_new.T @ J @ Z_new - Z_current.T @ J @ Z_current).item()
                symplectic_error.append(sym_error)
                
                # Update for next iteration
                Z_current = Z_new
                
                # Print diagnostics periodically
                if i % 10 == 0 or i == num_steps - 1:
                    print(f"Step {i}: det(Z1) = {det_Z1:.6e}, cond(Z1) = {cond_Z1:.6e}, sym_error = {sym_error:.6e}")
                
            except Exception as e:
                print(f"Step {i}: Error - {str(e)}")
                break
        
        # Plot the diagnostics
        if len(det_Z1_history) > 0:
            plt.figure(figsize=(15, 12))
            
            plt.subplot(3, 1, 1)
            plt.semilogy([abs(d) for d in det_Z1_history])
            plt.title(f'|det(Z1)| vs Step (h = {h})')
            plt.xlabel('Step')
            plt.ylabel('|det(Z1)|')
            plt.grid(True)
            
            plt.subplot(3, 1, 2)
            plt.semilogy(cond_Z1_history)
            plt.title(f'Condition Number of Z1 vs Step (h = {h})')
            plt.xlabel('Step')
            plt.ylabel('cond(Z1)')
            plt.grid(True)
            
            plt.subplot(3, 1, 3)
            plt.semilogy(symplectic_error)
            plt.title(f'Symplectic Structure Error vs Step (h = {h})')
            plt.xlabel('Step')
            plt.ylabel('Error')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f'symplectic_stability_h{h}.png')
            print(f"Diagnostics saved to 'symplectic_stability_h{h}.png'")
            
    return True

def diagnose_instability_source():
    """
    Diagnose the exact source of instability in the Riccati integration by testing
    fundamental mathematical properties of the integration process.
    """
    print("\n===== DIAGNOSING ROOT CAUSE OF INSTABILITY =====")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize the solver
    solver = RiccatiSolver(device=device)
    
    # Create test matrices
    n = 4  # State dimension
    A, B, Q, R, X_current = create_test_matrices(n, device)
    
    # 1. TEST THE HAMILTONIAN CONSTRUCTION
    print("\n1. ANALYZING HAMILTONIAN CONSTRUCTION:")
    H = create_hamiltonian_matrix(A, B, Q, R, device)
    
    # Create the canonical symplectic form J
    I_n = torch.eye(n, device=device)
    J = torch.zeros((2*n, 2*n), device=device)
    J[:n, n:] = I_n
    J[n:, :n] = -I_n
    
    # Check if H is truly Hamiltonian: H^T J + J H = 0
    hamiltonian_error = torch.norm(H.T @ J + J @ H).item()
    print(f"   a) Hamiltonian property error: {hamiltonian_error:.6e}")
    if hamiltonian_error > 1e-6:
        print("      ERROR: H is not properly Hamiltonian!")
    
    # Check eigenvalues of H - Hamiltonian matrices have eigenvalues in pairs (λ, -λ)
    eigvals_H = torch.linalg.eigvals(H)
    real_parts = eigvals_H.real
    imag_parts = eigvals_H.imag
    print(f"   b) Eigenvalue spectrum of H: real min/max [{torch.min(real_parts).item():.2e}, {torch.max(real_parts).item():.2e}]")
    print(f"                                 imag min/max [{torch.min(imag_parts).item():.2e}, {torch.max(imag_parts).item():.2e}]")
    
    # Check eigenvalue pairing
    sorted_eigs = sorted(eigvals_H.detach().cpu().numpy(), key=lambda x: (x.real, x.imag))
    has_paired_eigs = True
    for i in range(0, len(sorted_eigs), 2):
        if i+1 < len(sorted_eigs):
            e1, e2 = sorted_eigs[i], sorted_eigs[i+1]
            if not (abs(e1 + e2) < 1e-5 or abs(abs(e1) - abs(e2)) < 1e-5):
                has_paired_eigs = False
                break
    print(f"   c) Eigenvalues properly paired: {has_paired_eigs}")
    
    # Check conditioning of H
    try:
        H_cond = torch.linalg.cond(H).item()
        print(f"   d) Condition number of H: {H_cond:.2e}")
        if H_cond > 1e8:
            print("      ERROR: H is extremely ill-conditioned!")
    except:
        print("   d) Failed to compute condition number of H")
    
    # 2. TEST THE MATRIX EXPONENTIAL
    print("\n2. ANALYZING MATRIX EXPONENTIAL:")
    
    # Generate matrix exponential
    h = 0.01
    exp_H = solver._get_cached_matrix_exp(H, h)
    
    # Check if exp(H) is symplectic: exp(H)^T J exp(H) = J
    symplectic_error = torch.norm(exp_H.T @ J @ exp_H - J).item()
    print(f"   a) Symplecticity of exp(h*H): {symplectic_error:.6e}")
    if symplectic_error > 1e-4:
        print("      ERROR: Matrix exponential is not properly symplectic!")
    
    # 3. TEST Z1 EVOLUTION
    print("\n3. ANALYZING Z1 EVOLUTION:")
    
    # Initial Z = [I; X]
    Z1 = torch.eye(n, device=device)
    Z2 = X_current.clone()
    Z_current = torch.cat([Z1, Z2], dim=0)
    
    # Track determinant and condition of Z1 over many steps
    num_steps = 50
    det_Z1 = []
    cond_Z1 = []
    symplectic_error_Z = []
    
    # Split the work over smaller step sizes to see progression of instability
    h_values = [0.01, 0.005, 0.001]
    
    for h in h_values:
        print(f"\n   Testing with h={h}:")
        
        # Reset for this h
        Z1 = torch.eye(n, device=device)
        Z2 = X_current.clone()
        Z_current = torch.cat([Z1, Z2], dim=0)
        
        det_Z1 = []
        cond_Z1 = []
        symplectic_error_Z = []
        
        for step in range(num_steps):
            # Step forward
            try:
                exp_H = solver._get_cached_matrix_exp(H, h)
                Z_new = exp_H @ Z_current
                
                # Extract Z1
                Z1_new = Z_new[:n, :]
                
                # Analyze Z1 properties
                det_val = torch.linalg.det(Z1_new).item()
                det_Z1.append(det_val)
                
                sing_vals = torch.linalg.svdvals(Z1_new)
                cond_val = (sing_vals[0] / sing_vals[-1]).item() if sing_vals[-1] > 1e-10 else float('inf')
                cond_Z1.append(cond_val)
                
                # Check symplectic property from Z_current to Z_new
                # For a proper symplectic transform from one step to another:
                # Z_new^T J Z_new should equal Z_current^T J Z_current 
                sym_error = torch.norm(Z_new.T @ J @ Z_new - Z_current.T @ J @ Z_current).item()
                symplectic_error_Z.append(sym_error)
                
                # Update
                Z_current = Z_new
                
                # Print diagnostics every 10 steps
                if step % 10 == 0 or step == num_steps-1 or cond_val > 1e8:
                    print(f"      Step {step}: det(Z1)={det_val:.6e}, cond(Z1)={cond_val:.6e}, sym_error={sym_error:.6e}")
                
                # Check for imminent failure
                if cond_val > 1e10:
                    print("      CRITICAL INSTABILITY DETECTED - TERMINATING EARLY")
                    break
                    
            except Exception as e:
                print(f"      Step {step}: FAILED - {str(e)}")
                break
        
        # Analyze trends
        if len(det_Z1) > 10:
            det_ratio = det_Z1[-1] / det_Z1[0] if det_Z1[0] != 0 else float('inf')
            print(f"   a) Z1 determinant trend: init={det_Z1[0]:.6e}, final={det_Z1[-1]:.6e}, ratio={det_ratio:.6e}")
            
            max_cond = max(cond_Z1)
            print(f"   b) Z1 condition number trend: init={cond_Z1[0]:.6e}, max={max_cond:.6e}")
            
            max_sym_error = max(symplectic_error_Z)
            print(f"   c) Maximum symplectic error: {max_sym_error:.6e}")
            
    # 4. CONCLUSIONS
    print("\n4. ROOT CAUSE ANALYSIS:")
    
    # Check if H is properly Hamiltonian
    if hamiltonian_error > 1e-6:
        print("   - ISSUE: The Hamiltonian matrix H is not properly structured.")
        print("     This breaks the symplectic properties of the integration.")
    
    # Check if exp(H) is properly symplectic
    if symplectic_error > 1e-4:
        print("   - ISSUE: The matrix exponential is not properly preserving the symplectic structure.")
        print("     This can lead to numerical drift and eventual instability.")
    
    # Check for stiff systems (widely separated eigenvalues)
    if H_cond > 1e6:
        print("   - ISSUE: The Hamiltonian has extremely high condition number.")
        print("     This indicates a stiff system that is inherently difficult to integrate.")
    
    # Check for Z1 singularity
    if any(c > 1e6 for c in cond_Z1):
        print("   - ISSUE: Z1 becomes increasingly ill-conditioned during propagation.")
        print("     This leads to numerical instability when computing X = Z2 * Z1^(-1).")
    
    # Check symplectic evolution
    if any(e > 1e-4 for e in symplectic_error_Z):
        print("   - ISSUE: The symplectic structure is not preserved during evolution.")
        print("     This can cause the solution to drift away from the manifold of valid solutions.")
    
    return "Analysis completed"

if __name__ == "__main__":
    # Run diagnostic to find the root cause
    diagnose_instability_source()
    
    # Test the direct integrator step
    H, X_start, Z_start, X_step, Z_step = test_symplectic_integrator_step()
    
    # Test the full integrator
    X_history = test_symplectic_integrator()
    
    # Test the stability of the integrator
    test_symplectic_integration_stability()
    
    print("\nAll tests completed.") 