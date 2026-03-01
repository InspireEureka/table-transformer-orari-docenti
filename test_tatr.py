# test_tatr.py

# Import necessary libraries
import torch
import os

# Function to verify GPU availability
def verify_gpu():
    if torch.cuda.is_available():
        print("GPU is available.")
    else:
        print("GPU is not available.")

# Function to verify TATR setup
def verify_tatr_setup():
    # Assuming TATR is a package; check if it's installed
    try:
        import tatrlib  # Example import, change according to actual library
        print("TATR is properly set up.")
    except ImportError:
        print("TATR is not installed.")

if __name__ == '__main__':
    verify_gpu()
    verify_tatr_setup()
