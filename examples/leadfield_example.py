"""
Simple example for using the build_leadfield pipeline
"""
import os
import importlib.util

# Load build_leadfield.py directly from the filesystem
module_path = os.path.join(os.path.dirname(__file__), '..', 'eeg_processing', 'leadfield', 'build_leadfield.py')
spec = importlib.util.spec_from_file_location('build_leadfield', module_path)
build_leadfield = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_leadfield)
main = build_leadfield.main

# Example BIDS dataset structure:
# /path/to/bids_root/
# ├─ sub-01/
# │  └─ ses-eeg/
# │     └─ eeg/
# │        ├─ sub-01_ses-eeg_eeg.edf
# │        ├─ sub-01_ses-eeg_eeg.json
# │        └─ sub-01_ses-eeg_channels.tsv

if __name__ == "__main__":
    # Run with test parameters
    leadfield, raw, src, fwd = main(
        bids_root=r"C:\Users\Kyle\Dissertation\data",  # Raw string for Windows paths
        subject="sub-001",                        # Example subject ID
        session="ses-t1",                        # Matches directory name
        sphere_radius=0.09,                 # 90mm sphere
        n_sources=5000,                     # Reduced for faster computation
        visualize=True
    )
    
    print("\nLeadfield matrix shape:", leadfield.shape)
    print("Channel names:", raw.info['ch_names'][:5])  # Show first 5 channels 