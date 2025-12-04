"""
CUDA Extension Module for GPU-Tigramite
========================================

This module is built by CMake during installation and provides the core
CUDA-accelerated functions for Conditional Mutual Information estimation.

The gpucmiknn module is a compiled extension (.so file) that is built
from C++/CUDA source code during `pip install`.
"""

# The gpucmiknn module is a compiled extension built by CMake
# It will be available after successful installation
try:
    from . import gpucmiknn
    __all__ = ['gpucmiknn']
except ImportError as e:
    # Module not yet built - this is expected during development
    # Users will see a more helpful error from wrapper.py
    pass