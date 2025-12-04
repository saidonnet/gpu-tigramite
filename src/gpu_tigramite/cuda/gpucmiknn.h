/**
 * GPU-Tigramite CUDA Header
 * ==========================
 * 
 * Function declarations for GPU-accelerated CMI estimation.
 * 
 * Author: Your Name (2025)
 * Based on original GPU CMIknn CUDA implementation
 * License: GPL-3.0
 */

#ifndef GPUCMIKNN_H
#define GPUCMIKNN_H

#include <cstddef>

/**
 * Initialize GPU context
 * 
 * Must be called once per GPU device before using other CUDA functions.
 * Thread-safe for multi-GPU setups.
 * 
 * @return 0 on success, non-zero on failure
 */
size_t call_init_gpu_cuda();

/**
 * Unconditional independence test: X _|_ Y
 *
 * @param data Data array (column-major: samples × features)
 * @param data_height Number of samples
 * @param data_width Number of features (dim_X + dim_Y)
 * @param k Number of nearest neighbors
 * @param permutations Number of permutation samples
 * @param mi_value Output parameter for MI value (can be NULL if not needed)
 * @return p-value from permutation test
 */
double pval_l0_cuda_shared(
    const double* data,
    size_t data_height,
    size_t data_width,
    int k,
    size_t permutations,
    double* mi_value = nullptr
);

/**
 * Conditional independence test: X _|_ Y | Z
 *
 * @param data Data array with [X, Y, Z] (column-major)
 * @param x_permutations Pre-computed X permutations
 * @param data_height Number of samples
 * @param data_width Number of features (dim_X + dim_Y + dim_Z)
 * @param k Number of nearest neighbors
 * @param k_perm Number of neighbors for permutation test
 * @param permutations Number of permutation samples
 * @param mi_value Output parameter for CMI value (can be NULL if not needed)
 * @return p-value from permutation test
 */
double pval_ln_cuda(
    const double* data,
    const double* x_permutations,
    size_t data_height,
    size_t data_width,
    int k,
    int k_perm,
    size_t permutations,
    double* mi_value = nullptr
);

/**
 * Generate restricted permutations conditioned on Z
 * 
 * @param data Conditioning variables Z (column-major)
 * @param data_height Number of samples
 * @param data_width Number of features in Z
 * @param permutations Number of permutations to generate
 * @param output Output array for permutation indices
 */
void perm_cuda_multi(
    const double* data,
    size_t data_height,
    size_t data_width,
    size_t permutations,
    int* output
);

/**
 * Batch unconditional test for multiple Y variables
 * 
 * @param data Full dataset (column-major)
 * @param obs_count Number of samples
 * @param k Number of nearest neighbors
 * @param permutations Number of permutation samples
 * @param x_id Index of X variable
 * @param vars Total number of variables
 * @param pvalOfX Output array for p-values
 * @param candidates Indices of Y variables to test
 * @param yDim Number of Y variables
 */
void pval_l0_row_cuda(
    const double* data,
    size_t obs_count,
    int k,
    size_t permutations,
    int x_id,
    size_t vars,
    double* pvalOfX,
    int* candidates,
    size_t yDim
);

/**
 * Batch conditional test for multiple conditioning sets
 * 
 * @param data Full dataset (column-major)
 * @param x_permutations Pre-computed permutations for all conditioning sets
 * @param obs_count Number of samples
 * @param k Number of nearest neighbors
 * @param k_perm Number of neighbors for permutation test
 * @param permutations Number of permutation samples
 * @param x_id Index of X variable
 * @param vars Total number of variables
 * @param lvl Level (size) of conditioning sets
 * @param sList List of conditioning sets
 * @param sEntries Number of conditioning sets
 * @param sOfX Output array for selected conditioning sets
 * @param pvalOfX Output array for p-values
 * @param alpha Significance threshold
 * @param candidates Indices of Y variables
 * @param yDim Pointer to number of Y variables (may be modified)
 * @param originalYDim Original number of Y variables
 * @param splitted Whether to use split conditioning sets
 */
void pval_ln_row_cuda(
    const double* data,
    const double* x_permutations,
    size_t obs_count,
    int k,
    int k_perm,
    size_t permutations,
    int x_id,
    size_t vars,
    int lvl,
    int* sList,
    size_t sEntries,
    int* sOfX,
    double* pvalOfX,
    double alpha,
    int* candidates,
    size_t* yDim,
    size_t originalYDim,
    bool splitted
);

/**
 * Generate restricted permutations for multiple conditioning sets
 * 
 * @param data Full dataset (column-major)
 * @param obs_count Number of samples
 * @param vars Total number of variables
 * @param permutations Number of permutations
 * @param x_permutations Output array for permutations
 * @param sList List of conditioning sets
 * @param dim Size of each conditioning set
 * @param sEntries Number of conditioning sets
 * @param x_id Index of X variable
 */
void perm_cuda_multi_all(
    const double* data,
    size_t obs_count,
    size_t vars,
    size_t permutations,
    double* x_permutations,
    int* sList,
    size_t dim,
    size_t sEntries,
    int x_id
);

#endif // GPUCMIKNN_H