"""
GPU-Tigramite: GPU-Accelerated Causal Discovery
=================================================

GPU-accelerated Conditional Mutual Information estimation for Tigramite.
50-430x faster than CPU with automatic multi-GPU scaling.

Quick Start:
-----------
```python
from gpu_tigramite import GPUCMIknn
from tigramite.pcmci import PCMCI
from tigramite import data_processing as pp

# Create GPU-accelerated test
cond_ind_test = GPUCMIknn(knn=5, sig_samples=100)

# Use with PCMCI
pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
results = pcmci.run_pcmci(tau_max=2, pc_alpha=0.05)
```

Multi-GPU Support:
------------------
Automatically uses all available GPUs when Ray is initialized:
```python
import ray
ray.init(num_gpus='auto')

# Same API - now uses all GPUs!
cond_ind_test = GPUCMIknn(knn=5, sig_samples=100)
```
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__license__ = "GPL-3.0"

from .wrapper import GPUCMIknn, create_gpu_cmi_test, GPU_AVAILABLE
from .multi_gpu import get_gpu_count, initialize_multi_gpu
from .gpu_parcorr import GPUParCorr
from .gpu_preprocessing import GPUPreprocessor
from .gpu_batch_processor import GPUBatchProcessor

__all__ = [
    "GPUCMIknn",
    "GPUParCorr",
    "GPUPreprocessor",
    "GPUBatchProcessor",
    "GPU_AVAILABLE",
    "create_gpu_cmi_test",
    "get_gpu_count",
    "initialize_multi_gpu",
    "__version__",
]