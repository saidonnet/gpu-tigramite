# Changelog

All notable changes to gpu-tigramite will be documented in this file.

## [0.1.0] - 2024

### Added
- Initial release of GPU-accelerated Tigramite
- GPUCMIknn: 50-430x faster Conditional Mutual Information estimation
- GPUParCorr: 20-100x faster partial correlation tests
- GPUPreprocessor: 5-20x faster data standardization
- GPUBatchProcessor: Efficient batch processing for multiple tests
- Multi-GPU support via Ray
- Double precision (float64) CUDA kernels
- pybind11-based Python bindings
- CMake build system with CUDA 12.x support

### Fixed
- Numerical stability improvements with epsilon handling
- Zero-variance crash prevention
- Proper double precision throughout pipeline