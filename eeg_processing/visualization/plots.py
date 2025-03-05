"""
State-Space Visualization Functions

This module contains visualization tools for coupled state-space models.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_states_and_observations(states, observations, time_points):
    """
    Plot hidden states and observed signals together.
    
    Parameters:
    states (np.ndarray): Matrix of state variables (states x time)
    observations (np.ndarray): Observed EEG signals (channels x time)
    time_points (np.ndarray): Time vector for x-axis
    """
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    
    # State variables plot
    ax[0].plot(time_points, states.T)
    ax[0].set_title("Hidden State Variables")
    ax[0].set_xlabel("Time (s)")
    
    # Observations plot
    ax[1].plot(time_points, observations.T)
    ax[1].set_title("Observed EEG Signals")
    ax[1].set_xlabel("Time (s)")
    
    plt.tight_layout()
    return fig

def plot_state_couplings(coupling_matrix, state_labels=None):
    """
    Visualize coupling matrix between hidden states.
    
    Parameters:
    coupling_matrix (np.ndarray): State coupling matrix (n_states x n_states)
    state_labels (list): Optional list of state labels
    """
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(coupling_matrix, cmap='viridis')
    plt.colorbar(label='Coupling Strength')
    plt.title("State-Space Coupling Matrix")
    
    if state_labels:
        plt.xticks(np.arange(len(state_labels)), state_labels)
        plt.yticks(np.arange(len(state_labels)), state_labels)
    
    return fig

def plot_state_evolution(state_trajectories, time_window):
    """
    Plot the evolution of state variables over a specific time window.
    
    Parameters:
    state_trajectories (np.ndarray): 3D array (trials x states x time)
    time_window (tuple): Start and end time for plotting
    """
    # Implementation logic here
    pass 