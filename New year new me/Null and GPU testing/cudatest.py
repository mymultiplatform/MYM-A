# Method 1: Using torch
import torch

print(f"CUDA is available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")

# Method 2: Using nvidia-smi through subprocess
import subprocess
import os

def get_cuda_version():
    try:
        output = subprocess.check_output(["nvidia-smi"], universal_newlines=True)
        print("NVIDIA System Management Interface output:")
        print(output)
    except FileNotFoundError:
        print("nvidia-smi not found. Make sure NVIDIA drivers are installed.")
    except subprocess.CalledProcessError:
        print("Error executing nvidia-smi command.")

get_cuda_version()



import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")