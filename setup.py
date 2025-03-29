import torch
import retro

# Check if CUDA is available
print(f"CUDA is available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # Get CUDA version
    print(f"CUDA version: {torch.version.cuda}")
    
    # Get number of CUDA devices
    device_count = torch.cuda.device_count()
    print(f"Number of CUDA devices: {device_count}")
    
    # Get current device information
    current_device = torch.cuda.current_device()
    print(f"Current CUDA device: {current_device}")
    print(f"Device name: {torch.cuda.get_device_name(current_device)}")
    
    # Test with a simple tensor operation
    print("Testing CUDA with tensor operations:")
    x = torch.rand(3, 3)
    print(f"CPU tensor: {x}")
    x_cuda = x.cuda()
    print(f"GPU tensor: {x_cuda}")
    print(f"GPU tensor device: {x_cuda.device}")
else:
    print("CUDA is not available. Check your PyTorch installation or GPU drivers.")

