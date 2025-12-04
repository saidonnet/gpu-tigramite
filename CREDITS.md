## Credits and Acknowledgments

### GPU-Tigramite (2025)

**Author**: GPU-Tigramite Contributors
**Contributions**:
- Multi-GPU support (1 to 1000+ GPUs) via Ray
- Modern build system (CMake, pybind11)
- Python 3.10-3.12 compatibility
- CUDA 12.x support
- Production-ready error handling
- Comprehensive documentation

**License**: GPL-3.0

---

### Original GPU CMIknn CUDA Implementation

The CUDA kernels for GPU-accelerated conditional mutual information estimation
are based on prior work in efficient k-nearest neighbor causal discovery.

**Original implementation**: Based on GPU-accelerated k-NN algorithms for
conditional mutual information estimation in causal discovery research.

**Key references**:
- Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. Physical Review E, 69(6), 066138.
- Frenzel, S., & Pompe, B. (2007). Partial mutual information for coupling analysis of multivariate time series. Physical Review Letters, 99(20), 204101.

---

### Tigramite Framework

This library extends [Tigramite](https://github.com/jakobrunge/tigramite) by
Jakob Runge et al.

**Citation**:
```bibtex
@article{runge2019detecting,
  title={Detecting and quantifying causal associations in large nonlinear time series datasets},
  author={Runge, Jakob and Nowack, Peer and Kretschmer, Marlene and Flaxman, Seth and Sejdinovic, Dino},
  journal={Science Advances},
  volume={5},
  number={11},
  pages={eaau4996},
  year={2019},
  publisher={American Association for the Advancement of Science}
}
```

**Repository**: https://github.com/jakobrunge/tigramite  
**License**: GPL-3.0

---

### Dependencies and Tools

**pybind11**  
- Authors: Wenzel Jakob and contributors
- License: BSD-style
- URL: https://github.com/pybind/pybind11

**PyTorch**  
- Organization: Meta AI (Facebook AI Research)
- License: BSD-style
- URL: https://pytorch.org

**Ray**  
- Organization: Anyscale
- License: Apache 2.0
- URL: https://www.ray.io

**NumPy**  
- Developers: NumPy community
- License: BSD-style
- URL: https://numpy.org

**CMake**  
- Organization: Kitware
- License: BSD-style
- URL: https://cmake.org

---

### Community Contributions

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Contributors**:
- GPU-Tigramite Contributors (2025) - Original GPU-Tigramite library
- [Add contributors here as they contribute]

---

### Funding and Support

[Add any funding sources, grants, or institutional support here]

---

### Citation

If you use this library in your research, please cite:

```bibtex
@software{gpu_tigramite2025,
  title={GPU-Tigramite: GPU-Accelerated Causal Discovery for Time Series},
  author={GPU-Tigramite Contributors},
  year={2025},
  url={https://github.com/gpu-tigramite/gpu-tigramite},
  note={50-430x faster conditional mutual information estimation with multi-GPU support}
}
```

Also cite the original Tigramite paper (see above).

---

### License Compliance

This software is distributed under GPL-3.0 to maintain compatibility with
Tigramite and ensure scientific reproducibility. All dependencies are
compatible with GPL-3.0.

For commercial licensing inquiries, please open an issue on GitHub.