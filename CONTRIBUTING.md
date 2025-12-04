# Contributing to GPU-Tigramite

Thank you for your interest in contributing to GPU-Tigramite!

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/gpu-tigramite/gpu-tigramite.git
   cd gpu-tigramite
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Build CUDA extensions:
   ```bash
   mkdir build && cd build
   cmake ..
   make -j$(nproc)
   ```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

- Follow PEP 8 for Python code
- Use type hints where possible
- Document all public functions

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## Reporting Issues

Please use GitHub Issues to report bugs or request features.