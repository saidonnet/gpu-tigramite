/**
 * GPU-Tigramite CUDA Python Bindings (pybind11)
 * ==============================================
 * 
 * Modern Python bindings for GPU-accelerated CMI estimation using pybind11.
 * Replaces the older Boost.Python implementation for better performance and
 * compatibility with Python 3.10-3.12.
 * 
 * Author: Your Name (2025) - Modernization with pybind11
 * Based on original GPU CMIknn CUDA implementation
 * License: GPL-3.0
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "gpucmiknn.h"

namespace py = pybind11;

/**
 * Unconditional independence test: X _|_ Y
 *
 * @param data NumPy array (features × samples)
 * @param k Number of nearest neighbors
 * @param permutations Number of permutation samples
 * @return tuple of (MI value, p-value)
 */
py::tuple pval_l0_wrapper(
    py::array_t<double, py::array::c_style> data,
    int k,
    size_t permutations
) {
    // Get buffer info
    py::buffer_info buf = data.request();
    
    if (buf.ndim != 2) {
        throw std::runtime_error("Input must be 2-dimensional");
    }
    
    size_t data_width = buf.shape[0];      // features
    size_t data_height = buf.shape[1];     // samples
    
    // Get data pointer
    double* data_ptr = static_cast<double*>(buf.ptr);
    
    // Allocate and copy data in column-major format
    double* data_c = new double[data_height * data_width];
    
    for (size_t i = 0; i < data_height; i++) {
        for (size_t j = 0; j < data_width; j++) {
            data_c[data_height * j + i] = data_ptr[j * data_height + i];
        }
    }
    
    // Call CUDA function with MI value output
    double mi_value = 0.0;
    double pval = pval_l0_cuda_shared(data_c, data_height, data_width, k, permutations, &mi_value);
    
    // Cleanup
    delete[] data_c;
    
    return py::make_tuple(mi_value, pval);
}

/**
 * Conditional independence test: X _|_ Y | Z
 *
 * @param data NumPy array with concatenated [X, Y, Z] (features × samples)
 * @param x_permutations Pre-computed X permutations (num_perms × samples*dim_x)
 * @param k Number of nearest neighbors
 * @param k_perm Number of neighbors for permutation test
 * @param permutations Number of permutation samples
 * @return tuple of (CMI value, p-value)
 */
py::tuple pval_ln_wrapper(
    py::array_t<double, py::array::c_style> data,
    py::array_t<double, py::array::c_style> x_permutations,
    int k,
    int k_perm,
    size_t permutations
) {
    // Get data buffer
    py::buffer_info data_buf = data.request();
    if (data_buf.ndim != 2) {
        throw std::runtime_error("Data must be 2-dimensional");
    }
    
    size_t data_width = data_buf.shape[0];
    size_t data_height = data_buf.shape[1];
    
    // Get permutations buffer
    py::buffer_info perm_buf = x_permutations.request();
    if (perm_buf.ndim != 2) {
        throw std::runtime_error("Permutations must be 2-dimensional");
    }
    
    size_t x_permutations_width = perm_buf.shape[0];
    size_t x_permutations_height = perm_buf.shape[1];
    
    // Allocate and copy data
    double* data_c = new double[data_height * data_width];
    double* data_ptr = static_cast<double*>(data_buf.ptr);
    
    for (size_t i = 0; i < data_height; i++) {
        for (size_t j = 0; j < data_width; j++) {
            data_c[data_height * j + i] = data_ptr[j * data_height + i];
        }
    }
    
    // Allocate and copy permutations
    double* x_permutations_c = new double[x_permutations_height * x_permutations_width];
    double* perm_ptr = static_cast<double*>(perm_buf.ptr);
    
    for (size_t i = 0; i < x_permutations_height; i++) {
        for (size_t j = 0; j < x_permutations_width; j++) {
            x_permutations_c[x_permutations_height * j + i] =
                perm_ptr[j * x_permutations_height + i];
        }
    }
    
    // Call CUDA function with CMI value output
    double cmi_value = 0.0;
    double pval = pval_ln_cuda(
        data_c,
        x_permutations_c,
        data_height,
        data_width,
        k,
        k_perm,
        permutations,
        &cmi_value
    );
    
    // Cleanup
    delete[] data_c;
    delete[] x_permutations_c;
    
    return py::make_tuple(cmi_value, pval);
}

/**
 * Generate restricted permutations conditioned on Z
 * 
 * @param z Conditioning variables (features × samples)
 * @param permutations Number of permutations to generate
 * @return NumPy array of permutation indices (permutations × samples)
 */
py::array_t<int> rperm_multi_wrapper(
    py::array_t<double, py::array::c_style> z,
    size_t permutations
) {
    // Get buffer info
    py::buffer_info buf = z.request();
    
    if (buf.ndim != 2) {
        throw std::runtime_error("Z must be 2-dimensional");
    }
    
    size_t data_width = buf.shape[0];      // features
    size_t data_height = buf.shape[1];     // samples
    
    // Allocate and copy data
    double* data_c = new double[data_height * data_width];
    double* data_ptr = static_cast<double*>(buf.ptr);
    
    for (size_t i = 0; i < data_height; i++) {
        for (size_t j = 0; j < data_width; j++) {
            data_c[data_height * j + i] = data_ptr[j * data_height + i];
        }
    }
    
    // Allocate output
    int* x_permutations = new int[permutations * data_height];
    
    // Call CUDA function
    perm_cuda_multi(data_c, data_height, data_width, permutations, x_permutations);
    
    // Create NumPy array (permutations × samples)
    py::array_t<int> result({permutations, data_height});
    auto result_buf = result.request();
    int* result_ptr = static_cast<int*>(result_buf.ptr);
    
    // Copy data
    std::copy(x_permutations, x_permutations + permutations * data_height, result_ptr);
    
    // Cleanup
    delete[] data_c;
    delete[] x_permutations;
    
    return result;
}

/**
 * Initialize GPU context
 * 
 * Must be called once per GPU device before using CUDA functions.
 * Thread-safe for multi-GPU setups.
 * 
 * @return 0 on success
 */
size_t init_gpu_wrapper() {
    return call_init_gpu_cuda();
}

/**
 * Python module definition using pybind11
 */
PYBIND11_MODULE(gpucmiknn, m) {
    m.doc() = R"pbdoc(
        GPU-Tigramite CUDA Module
        =========================
        
        GPU-accelerated Conditional Mutual Information estimation for causal discovery.
        
        Functions:
        ----------
        init_gpu() : Initialize GPU context (must be called once per GPU)
        pval_l0(data, k, permutations) : Unconditional independence test
        pval_ln(data, x_permutations, k, k_perm, permutations) : Conditional independence test
        rperm_multi(z, permutations) : Generate restricted permutations
        
        Performance:
        -----------
        50-430x faster than CPU CMIknn on large datasets.
        
        License: GPL-3.0
    )pbdoc";
    
    // Initialize GPU
    m.def("init_gpu", &init_gpu_wrapper, R"pbdoc(
        Initialize GPU context.
        
        Must be called once per GPU device before using other functions.
        Thread-safe for multi-GPU setups where each worker initializes its own GPU.
        
        Returns
        -------
        status : int
            0 on success, non-zero on failure
        
        Examples
        --------
        >>> import gpucmiknn
        >>> gpucmiknn.init_gpu()
        0
    )pbdoc");
    
    // Unconditional test
    m.def("pval_l0", &pval_l0_wrapper, R"pbdoc(
        Unconditional independence test: X _|_ Y
        
        Tests whether X and Y are independent using k-NN based
        conditional mutual information estimation.
        
        Parameters
        ----------
        data : numpy.ndarray (float64)
            Data array with shape (features, samples).
            Features should be concatenation of X and Y.
        k : int
            Number of nearest neighbors
        permutations : int
            Number of permutation samples for significance testing
        
        Returns
        -------
        mi_value : float
            Mutual information estimate
        p_value : float
            p-value from permutation test
        
        Examples
        --------
        >>> import numpy as np
        >>> import gpucmiknn
        >>> gpucmiknn.init_gpu()
        >>> data = np.random.randn(2, 100).astype(np.float64)  # 2 features, 100 samples
        >>> mi_val, p_val = gpucmiknn.pval_l0(data, k=5, permutations=100)
    )pbdoc",
        py::arg("data"),
        py::arg("k"),
        py::arg("permutations")
    );
    
    // Conditional test
    m.def("pval_ln", &pval_ln_wrapper, R"pbdoc(
        Conditional independence test: X _|_ Y | Z
        
        Tests whether X and Y are conditionally independent given Z.
        
        Parameters
        ----------
        data : numpy.ndarray (float64)
            Data array with shape (features, samples).
            Features should be concatenation of [X, Y, Z].
        x_permutations : numpy.ndarray (float64)
            Pre-computed permutations of X with shape (permutations, samples*dim_x)
        k : int
            Number of nearest neighbors
        k_perm : int
            Number of neighbors for permutation test
        permutations : int
            Number of permutation samples
        
        Returns
        -------
        cmi_value : float
            Conditional mutual information estimate
        p_value : float
            p-value from permutation test
    )pbdoc",
        py::arg("data"),
        py::arg("x_permutations"),
        py::arg("k"),
        py::arg("k_perm"),
        py::arg("permutations")
    );
    
    // Restricted permutations
    m.def("rperm_multi", &rperm_multi_wrapper, R"pbdoc(
        Generate restricted permutations conditioned on Z.
        
        Creates permutation indices that preserve the local structure
        in the conditioning set Z. Used for conditional independence testing.
        
        Parameters
        ----------
        z : numpy.ndarray (float64)
            Conditioning variables with shape (features, samples)
        permutations : int
            Number of permutations to generate
        
        Returns
        -------
        perm_indices : numpy.ndarray (int32)
            Permutation indices with shape (permutations, samples)
    )pbdoc",
        py::arg("z"),
        py::arg("permutations")
    );
    
    // Version information
    m.attr("__version__") = "1.0.0";
}