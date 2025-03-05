# EEG Processing

A Python library for advanced EEG data processing, analysis, and state-space modeling.

## Overview

This library provides tools for processing, analyzing, and modeling EEG data using state-space models and variational inference techniques. It includes functionality for:

- Data loading and preprocessing
- Filtering and artifact removal
- Feature extraction
- State-space modeling
- Variational inference
- Visualization

## Project Structure

```
eeg_processing/
├── data/                  # Data loading and management
├── models/                # State-space models and inference
│   ├── control/           # Control-theoretic components
│   ├── linear_state_space_model.py
│   ├── coupled_state_space_vi.py
│   └── ...
├── preprocessing/         # Signal preprocessing tools
├── features/              # Feature extraction utilities
├── visualization/         # Plotting and visualization tools
└── utils/                 # Helper functions and utilities

examples/                  # Example scripts for different use cases
├── run_coupled_state_space_vi.py  # Example for subject-specific state-space modeling
└── leadfield_example.py  # Example for leadfield processing
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/eeg_processing.git
cd eeg_processing
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Available Models

### LinearStateSpaceModel

A basic linear state-space model for EEG data analysis. This model uses the standard state-space formulation:

```
x_{t+1} = Ax_t + Bu_t + w_t
y_t = Cx_t + v_t
```

Where:
- x_t is the latent state
- u_t is the input
- y_t is the observation (EEG data)
- w_t and v_t are process and measurement noise

### CoupledStateSpaceVI

An advanced state-space model using variational inference to estimate latent states and inputs. This model couples state and input estimation through a recurrent structure and uses variational inference for parameter learning.

Features:
- Optimal control-based state estimation
- Riccati equation solvers for optimal gains
- Variational inference for parameter learning
- Evidence Lower Bound (ELBO) optimization

## Usage Examples

### Example 1: Basic Linear State-Space Model

```python
from src.eeg_processing.models import LinearStateSpaceModel
import numpy as np

# Create some simulated EEG data
time_steps = 1000
channels = 64
data = np.random.randn(time_steps, channels)

# Create and fit the model
model = LinearStateSpaceModel(n_states=10, n_observations=channels)
model.fit(data)

# Get the estimated states
states = model.estimate_states(data)
```

### Example 2: Coupled State-Space VI Model

```python
from src.eeg_processing.models import CoupledStateSpaceVI
import torch

# Prepare your EEG data as a torch tensor
# data_tensor shape: [time_steps, channels]
data_tensor = torch.tensor(your_eeg_data, dtype=torch.float32)

# Create the model
model = CoupledStateSpaceVI(
    n_states=8,
    n_obs=data_tensor.shape[1],
    n_inputs=4,
    n_samples=20,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# Train the model
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(50):
    optimizer.zero_grad()
    loss = -model.elbo(data_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Sample states
with torch.no_grad():
    x_samples, u_samples = model.sample_augmented_state(data_tensor)
```

### Example 3: Run Coupled State-Space VI with Leadfield

We provide a complete example script that demonstrates how to:
- Load a subject's leadfield matrix (C)
- Load and preprocess cleaned EEG data
- Initialize and train the CoupledStateSpaceVI model
- Learn the A, B, Q, R, and P matrices
- Visualize and save results

Run the example:

```bash
# Basic usage with default parameters
python examples/run_coupled_state_space_vi.py

# Specify custom parameters
python examples/run_coupled_state_space_vi.py --subject sub-001 --x_dim 12 --n_epochs_to_use 20
```

The script supports the following command-line options:

```
--subject SUBJECT       Subject ID (default: sub-001)
--session SESSION       Session ID (default: ses-t1)
--task TASK             Task ID (default: resteyesc)
--x_dim X_DIM           Latent state dimension (default: 8)
--n_epochs_to_use N     Number of EEG epochs to use (default: 10)
--training_epochs N     Number of training epochs (default: 100)
--batch_size SIZE       Batch size (default: 64)
--learning_rate LR      Learning rate (default: 0.001)
--beta BETA             KL divergence weight (default: 0.1)
--gpu                   Use GPU if available
```

For more detailed examples, see the `examples/` directory.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributors

- Your Name <your.email@example.com>

## Acknowledgments

- Mention any resources, libraries, or research that inspired this work 