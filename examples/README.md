# Example Scripts

This directory contains example scripts for using the EEG processing models and utilities.

## Coupled State Space VI Model

The `run_coupled_state_space_vi.py` script demonstrates how to load EEG data and a leadfield matrix for a subject, then train a Coupled State Space Model with Variational Inference.

### Prerequisites

Ensure you have all required dependencies installed:

```bash
pip install -r ../requirements.txt
```

You'll need the following Python packages:
- numpy
- torch
- matplotlib
- mne
- pathlib

### Running the Script

To run the Coupled State Space VI model with default settings (subject 001):

```bash
python run_coupled_state_space_vi.py
```

#### Command-line Options

The script supports the following command-line arguments:

```
usage: run_coupled_state_space_vi.py [-h] [--subject SUBJECT] [--session SESSION] [--task TASK] [--x_dim X_DIM]
                                      [--n_epochs_to_use N_EPOCHS_TO_USE] [--training_epochs TRAINING_EPOCHS]
                                      [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE]
                                      [--beta BETA] [--gpu]

Run Coupled State Space VI model for EEG data

options:
  -h, --help            show this help message and exit
  --subject SUBJECT     Subject ID (default: sub-001)
  --session SESSION     Session ID (default: ses-t1)
  --task TASK           Task ID (default: resteyesc)
  --x_dim X_DIM         Latent state dimension (default: 8)
  --n_epochs_to_use N_EPOCHS_TO_USE
                        Number of EEG epochs to use for training (default: 10)
  --training_epochs TRAINING_EPOCHS
                        Number of training epochs (default: 100)
  --batch_size BATCH_SIZE
                        Batch size (default: 64)
  --learning_rate LEARNING_RATE
                        Learning rate (default: 0.001)
  --beta BETA           KL divergence weight (default: 0.1)
  --gpu                 Use GPU if available
```

#### Examples

Train with 20 EEG epochs and 12 latent states:
```bash
python run_coupled_state_space_vi.py --n_epochs_to_use 20 --x_dim 12
```

Train for a different subject with GPU acceleration:
```bash
python run_coupled_state_space_vi.py --subject sub-002 --gpu
```

### What the Script Does

1. **Data Loading**:
   - Loads the leadfield matrix for the specified subject from the leadfield directory
   - Loads pre-processed EEG data from the derivatives/cleaned_epochs directory

2. **Data Preparation**:
   - Reshapes the EEG data for model input
   - Downsamples the leadfield matrix to match the chosen latent state dimension
   - Splits data into training and validation sets

3. **Model Training**:
   - Initializes the CoupledStateSpaceVI model with the leadfield as the observation matrix C
   - Trains the model to learn parameters A, B, Q, R, and P
   - The model learns:
     - A: State transition matrix
     - B: Control input matrix
     - Q: Process noise covariance
     - R: Control cost matrix
     - P: Solution to the Riccati equation

4. **Visualization and Analysis**:
   - Plots training history (ELBO values)
   - Visualizes latent states
   - Compares observed vs. predicted EEG signals

5. **Results Saving**:
   - Saves the trained model
   - Saves the learned matrices (A, B, C, Q, R, P)
   - Outputs visualization plots

### Output

The script creates results in the `eeg_processing/results` directory:
- `{subject}_coupled_state_space_model.pt`: The trained PyTorch model
- `{subject}_learned_matrices.npz`: The learned matrices in NumPy format
- `{subject}_training_history.png`: Plot of training progress
- `{subject}_latent_states.png`: Visualization of latent states
- `{subject}_predictions.png`: Comparison of observed vs. predicted EEG signals

### Note on State Dimension

The script uses a default latent state dimension of 8, which is a simplification. 
In practice, you might want to experiment with different dimensions or use more sophisticated 
methods to determine the optimal number of latent states.

The leadfield matrix is downsampled to match this state dimension using a simple approach.
In a more sophisticated analysis, you might use:
- ROI-based averaging
- Principal Component Analysis (PCA)
- Anatomically-informed source selection 