"""
Multi-GPU Utilities for GPU-Tigramite
======================================

Helper functions for detecting GPUs and initializing multi-GPU environments.

Author: Your Name (2025)
License: GPL-3.0
"""

import warnings
from typing import Optional

def get_gpu_count() -> int:
    """
    Get the number of available CUDA GPUs.
    
    Returns
    -------
    num_gpus : int
        Number of available GPUs (0 if CUDA not available)
    
    Examples
    --------
    >>> num_gpus = get_gpu_count()
    >>> print(f"Found {num_gpus} GPUs")
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        return 0
    except ImportError:
        warnings.warn(
            "PyTorch not found. Cannot detect GPUs. Install with: pip install torch",
            RuntimeWarning
        )
        return 0


def initialize_multi_gpu(num_gpus: Optional[int] = None, num_cpus: Optional[int] = None):
    """
    Initialize Ray for multi-GPU processing.
    
    Parameters
    ----------
    num_gpus : int or None
        Number of GPUs to use. If None, uses all available GPUs.
    num_cpus : int or None
        Number of CPUs to use. If None, uses all available CPUs.
    
    Returns
    -------
    success : bool
        True if Ray initialized successfully
    
    Examples
    --------
    >>> # Use all available GPUs
    >>> initialize_multi_gpu()
    
    >>> # Use specific number of GPUs
    >>> initialize_multi_gpu(num_gpus=4)
    """
    try:
        import ray
        import os
        
        # Detect available resources
        if num_gpus is None:
            num_gpus = get_gpu_count()
        
        if num_cpus is None:
            num_cpus = os.cpu_count()
        
        # Initialize Ray
        if num_gpus > 0:
            ray.init(
                ignore_reinit_error=True,
                num_cpus=num_cpus,
                num_gpus=num_gpus
            )
            print(f"✓ Ray initialized with {num_cpus} CPUs and {num_gpus} GPUs")
            return True
        else:
            warnings.warn(
                "No GPUs detected. GPU-Tigramite will fall back to CPU mode.",
                RuntimeWarning
            )
            return False
            
    except ImportError:
        warnings.warn(
            "Ray not found. Multi-GPU support requires Ray. Install with: pip install ray",
            RuntimeWarning
        )
        return False


def get_gpu_info() -> dict:
    """
    Get detailed information about available GPUs.
    
    Returns
    -------
    gpu_info : dict
        Dictionary with GPU information
    
    Examples
    --------
    >>> info = get_gpu_info()
    >>> print(f"GPU 0: {info['names'][0]}, {info['memory_total'][0]} MB")
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return {
                'count': 0,
                'names': [],
                'memory_total': [],
                'memory_free': [],
                'cuda_version': None
            }
        
        num_gpus = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        memory_total = [torch.cuda.get_device_properties(i).total_memory // (1024**2) for i in range(num_gpus)]
        
        return {
            'count': num_gpus,
            'names': names,
            'memory_total': memory_total,
            'cuda_version': torch.version.cuda
        }
        
    except ImportError:
        return {
            'count': 0,
            'names': [],
            'memory_total': [],
            'cuda_version': None
        }


def print_gpu_info():
    """
    Print formatted GPU information.
    
    Examples
    --------
    >>> print_gpu_info()
    GPU Information:
      GPU 0: Tesla V100-SXM2-16GB (16384 MB)
      GPU 1: Tesla V100-SXM2-16GB (16384 MB)
      CUDA Version: 12.0
    """
    info = get_gpu_info()
    
    print("\n" + "="*60)
    print("GPU Information")
    print("="*60)
    
    if info['count'] == 0:
        print("No GPUs detected")
    else:
        for i in range(info['count']):
            print(f"  GPU {i}: {info['names'][i]} ({info['memory_total'][i]} MB)")
        if info['cuda_version']:
            print(f"  CUDA Version: {info['cuda_version']}")
    
    print("="*60 + "\n")