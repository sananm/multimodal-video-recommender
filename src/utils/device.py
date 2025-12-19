"""
Device Utilities for Apple Silicon, CUDA, and CPU

Automatically selects the best available device:
1. CUDA (NVIDIA GPU) - if available
2. MPS (Apple Silicon GPU) - if available
3. CPU - fallback
"""

import torch


def get_device(prefer_mps: bool = True) -> torch.device:
    """
    Get the best available device.

    Args:
        prefer_mps: Prefer MPS over CPU on Apple Silicon

    Returns:
        torch.device: Best available device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")

    elif prefer_mps and torch.backends.mps.is_available():
        # Check if MPS is actually usable (not just available)
        if torch.backends.mps.is_built():
            device = torch.device("mps")
            print("Using MPS (Apple Silicon GPU)")
        else:
            device = torch.device("cpu")
            print("MPS available but not built, using CPU")

    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device


def get_device_info() -> dict:
    """
    Get information about available devices.

    Returns:
        Dictionary with device availability info
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built() if hasattr(torch.backends.mps, 'is_built') else False,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if info["cuda_available"]:
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9

    return info


def move_to_device(data: dict, device: torch.device) -> dict:
    """
    Move all tensors in a dictionary to the specified device.

    Args:
        data: Dictionary potentially containing tensors
        device: Target device

    Returns:
        Dictionary with tensors moved to device
    """
    result = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device)
        else:
            result[key] = value
    return result


# Test
if __name__ == "__main__":
    print("Device Information:")
    print("-" * 40)

    info = get_device_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("-" * 40)
    device = get_device()
    print(f"\nSelected device: {device}")

    # Quick test
    print("\nTesting tensor operations...")
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    z = x @ y  # Matrix multiplication
    print(f"Matrix multiplication on {device}: OK")
    print(f"Result shape: {z.shape}")
