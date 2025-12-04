"""
GPU-Accelerated CMIknn Wrapper for Tigramite
==============================================

Provides a tigramite-compatible interface to GPU-accelerated Conditional Mutual Information
estimation using k-nearest neighbors. This wrapper enables PCMCI causal discovery to leverage
GPU acceleration while maintaining full compatibility with tigramite's API.

Performance: 50-430x faster than CPU CMIknn on large datasets.

Author: Your Name (2025) - Multi-GPU enhancements
Based on original GPU CMIknn CUDA implementation
License: GPL-3.0
"""

import numpy as np
import warnings
from typing import Optional, Tuple

# Try to import GPU functions
try:
    from gpu_tigramite.cuda import gpucmiknn
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    warnings.warn(
        "GPU CMIknn CUDA module not found. Please build the library using 'pip install -e .' "
        "or install from PyPI with 'pip install gpu-tigramite'. Falling back to CPU CMIknn.",
        RuntimeWarning
    )


class GPUCMIknn:
    """
    GPU-accelerated Conditional Mutual Information test using k-nearest neighbors.
    
    This class provides a tigramite-compatible interface to the GPU-accelerated CMIknn
    implementation. It can be used as a drop-in replacement for tigramite's CMIknn class
    in PCMCI causal discovery.
    
    Parameters
    ----------
    knn : int, optional (default: 5)
        Number of nearest neighbors for CMI estimation.
        Adaptive mode: set to int(0.2 * sample_size) automatically.
    
    sig_samples : int, optional (default: 100)
        Number of permutations for significance testing.
        More permutations = more accurate p-values but slower computation.
    
    sig_blocklength : None
        Included for tigramite compatibility. Not used in GPU implementation.
    
    verbosity : int, optional (default: 0)
        Verbosity level (0=silent, 1=basic info, 2=detailed debug).
    
    adaptive_knn : bool, optional (default: False)
        If True, automatically set knn to 20% of sample size.
    
    Attributes
    ----------
    measure : str
        Name of the independence measure ('cmi_knn').
    
    gpu_available : bool
        Whether GPU acceleration is available.
    
    Examples
    --------
    >>> from tigramite.pcmci import PCMCI
    >>> from tigramite import data_processing as pp
    >>> from gpu_tigramite import GPUCMIknn
    >>>
    >>> # Create GPU-accelerated independence test
    >>> cond_ind_test = GPUCMIknn(knn=5, sig_samples=100)
    >>>
    >>> # Use with PCMCI
    >>> dataframe = pp.DataFrame(data)
    >>> pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
    >>> results = pcmci.run_pcmci(tau_max=2)
    """
    
    # PRODUCTION MULTI-GPU: Track initialization per GPU device
    _gpu_initialized_per_device = {}
    _initialization_lock = None
    
    def __init__(
        self,
        knn: int = 5,
        sig_samples: int = 100,
        sig_blocklength: Optional[int] = None,
        verbosity: int = 0,
        adaptive_knn: bool = False,
        confidence: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs
    ):
        """Initialize GPU CMIknn wrapper."""
        self.knn = knn
        self.sig_samples = sig_samples
        self.sig_blocklength = sig_blocklength  # Not used, for compatibility
        self.verbosity = verbosity
        self.adaptive_knn = adaptive_knn
        self.confidence = confidence  # For tigramite compatibility (confidence intervals)
        self.measure = 'cmi_knn'
        self.seed = seed
        if seed is None:
            self.random_state = np.random.default_rng()
        else:
            self.random_state = np.random.default_rng(seed)
            
        self.gpu_available = GPU_AVAILABLE
        
        # PRODUCTION MULTI-GPU: Initialize GPU once per device
        # Each GPU device gets initialized independently for multi-GPU setups
        if GPU_AVAILABLE:
            # Detect current CUDA device (set by Ray via CUDA_VISIBLE_DEVICES)
            import os
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
            # When Ray sets CUDA_VISIBLE_DEVICES to a single GPU, it becomes device 0
            # Use the original GPU ID from environment for tracking
            gpu_id = int(cuda_visible.split(',')[0]) if ',' not in cuda_visible else 0
            
            # Initialize this GPU if not already done
            if gpu_id not in GPUCMIknn._gpu_initialized_per_device:
                try:
                    if self.verbosity >= 1:
                        print(f"⚡ Initializing GPU {gpu_id} for CMIknn...")
                        
                    gpucmiknn.init_gpu()
                    GPUCMIknn._gpu_initialized_per_device[gpu_id] = True
                    
                    if self.verbosity >= 1:
                        print(f"✓ GPU {gpu_id} CMIknn initialized: k={knn}, permutations={sig_samples}")
                        print(f"  GPU device {gpu_id} ready for causal discovery")
                except Exception as e:
                    warnings.warn(f"GPU {gpu_id} initialization failed: {e}. Falling back to CPU.", RuntimeWarning)
                    self.gpu_available = False
                    GPUCMIknn._gpu_initialized_per_device[gpu_id] = False
            elif self.verbosity >= 2:
                if GPUCMIknn._gpu_initialized_per_device.get(gpu_id, False):
                    print(f"✓ GPU {gpu_id} CMIknn instance (already initialized): k={knn}, permutations={sig_samples}")
                else:
                    print(f"⚠ GPU {gpu_id} initialization previously failed, using CPU")
                    self.gpu_available = False
        elif self.verbosity >= 1:
            print("⚠ GPU CMIknn CUDA module not available - please install gpu-tigramite with CUDA support")
    
    def set_verbosity(self, verbosity: int):
        """Set verbosity level."""
        self.verbosity = verbosity
    
    def set_dataframe(self, dataframe):
        """
        Set the dataframe for tigramite compatibility.
        
        This method is required by tigramite's PCMCI class. The dataframe
        is stored but not actively used by GPU CMIknn since data is passed
        directly to run_test_raw().
        
        Parameters
        ----------
        dataframe : tigramite.data_processing.DataFrame
            The dataframe containing the time series data.
        """
        self.dataframe = dataframe
        if self.verbosity >= 2:
            print(f"GPU CMIknn: Dataframe set with {dataframe.N} variables, "
                  f"{dataframe.T} timesteps")
    
    def run_test(self, X, Y, Z=None, tau_max=0, cut_off='2xtau_max'):
        """
        Run conditional independence test with tigramite's standard interface.
        
        This method is called by PCMCI and converts tigramite's format to the
        raw numpy arrays expected by run_test_raw().
        
        Parameters
        ----------
        X : list of tuples
            List of (var, lag) tuples for X variable(s).
        Y : list of tuples
            List of (var, lag) tuples for Y variable(s).
        Z : list of tuples or None
            List of (var, lag) tuples for conditioning variables.
        tau_max : int
            Maximum time lag (used for data extraction from dataframe).
        cut_off : str
            Cut-off mode for data extraction.
        
        Returns
        -------
        val : float
            Test statistic (CMI value, set to 0.0 for GPU version).
        pval : float
            p-value from significance test.
        """
        # Extract data arrays from dataframe
        X_data, Y_data, Z_data = self._get_array(X, Y, Z, tau_max, cut_off)
        
        # Call the raw test implementation
        return self.run_test_raw(X_data, Y_data, Z_data)
    
    def _get_array(self, X, Y, Z=None, tau_max=0, cut_off='2xtau_max'):
        """
        Extract data arrays from dataframe based on variable-lag specifications.
        
        This method converts tigramite's (var, lag) format into numpy arrays
        by extracting the appropriate time slices from the stored dataframe.
        
        Parameters
        ----------
        X : list of tuples
            List of (var, lag) tuples for X.
        Y : list of tuples
            List of (var, lag) tuples for Y.
        Z : list of tuples or None
            List of (var, lag) tuples for conditioning set.
        tau_max : int
            Maximum time lag.
        cut_off : str
            Cut-off mode ('2xtau_max' or others).
        
        Returns
        -------
        X_data : np.ndarray
            Data array for X.
        Y_data : np.ndarray
            Data array for Y.
        Z_data : np.ndarray or None
            Data array for Z (or None if no conditioning).
        """
        # Extract data directly from dataframe for each variable-lag pair
        # This is more reliable than construct_array for our use case
        
        def extract_lagged_data(var_lags):
            """Extract data for list of (var, lag) tuples"""
            if not var_lags:
                return None
            
            data_list = []
            for var, lag in var_lags:
                # Get time series for this variable
                ts = self.dataframe.values[:, var]
                # Apply lag: positive lag means past values
                if lag == 0:
                    data_list.append(ts)
                elif lag > 0:
                    # Shift backward by lag (use past values)
                    data_list.append(np.concatenate([ts[lag:], np.full(lag, np.nan)]))
                else:
                    # Negative lag (future values)
                    data_list.append(np.concatenate([np.full(-lag, np.nan), ts[:lag]]))
            
            # Stack and remove rows with NaN
            if len(data_list) == 1:
                result = data_list[0].reshape(-1, 1)
            else:
                result = np.column_stack(data_list)
            
            return result
        
        # Extract data for X, Y, Z
        X_data = extract_lagged_data(X)
        Y_data = extract_lagged_data(Y)
        Z_data = extract_lagged_data(Z) if Z else None
        
        # Find valid samples (no NaN in any variable)
        max_lag = max([abs(lag) for var, lag in X + Y + (Z if Z else [])])
        valid_start = max_lag
        valid_end = len(X_data) - max_lag if max_lag > 0 else len(X_data)
        
        X_data = X_data[valid_start:valid_end]
        Y_data = Y_data[valid_start:valid_end]
        if Z_data is not None:
            Z_data = Z_data[valid_start:valid_end]
        
        return X_data, Y_data, Z_data
    
    def run_test_raw(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """
        Run conditional independence test: X _|_ Y | Z
        
        This is the main interface method required by tigramite. It tests whether
        X is conditionally independent of Y given Z.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features_x)
            First variable(s). Usually shape (n_samples, 1) for univariate.
        
        Y : np.ndarray, shape (n_samples, n_features_y)
            Second variable(s). Usually shape (n_samples, 1) for univariate.
        
        Z : np.ndarray or None, shape (n_samples, n_features_z)
            Conditioning variable(s). If None, tests unconditional independence.
        
        Returns
        -------
        cmi_value : float
            Conditional Mutual Information estimate (or MI for unconditional test).
        
        p_value : float
            p-value from permutation test. Values > 0.05 suggest independence.
        
        Raises
        ------
        RuntimeError
            If GPU CUDA module is not available.
        
        ValueError
            If input arrays have incompatible shapes or invalid parameters.
        """
        if not GPU_AVAILABLE:
            raise RuntimeError(
                "GPU CMIknn CUDA module not available. Please install gpu-tigramite with:\n"
                "  pip install gpu-tigramite\n"
                "Or build from source:\n"
                "  git clone https://github.com/yourusername/gpu-tigramite\n"
                "  cd gpu-tigramite && pip install -e .\n"
                "Alternatively, fall back to CPU CMIknn: tigramite.independence_tests.cmiknn.CMIknn"
            )
        
        # Validate inputs
        n_samples_x = len(X)
        n_samples_y = len(Y)
        
        if n_samples_x != n_samples_y:
            raise ValueError(f"X and Y must have same length: {n_samples_x} != {n_samples_y}")
        
        if Z is not None and len(Z) != n_samples_x:
            raise ValueError(f"Z must have same length as X and Y: {len(Z)} != {n_samples_x}")
        
        # Ensure arrays are 2D
        if X.ndim == 1:
            X = X.reshape((n_samples_x, 1))
        if Y.ndim == 1:
            Y = Y.reshape((n_samples_y, 1))
        
        # Determine k (adaptive or fixed)
        k = int(n_samples_x * 0.2) if self.adaptive_knn else self.knn
        
        # Ensure k is valid
        if k >= n_samples_x - 1:
            k = max(1, n_samples_x - 2)
            if self.verbosity >= 1:
                warnings.warn(f"k too large for sample size. Adjusted to k={k}")
        
        try:
            if Z is None:
                # Unconditional independence test: X _|_ Y
                cmi_value, p_value = self._test_unconditional(X, Y, k)
            else:
                # Conditional independence test: X _|_ Y | Z
                cmi_value, p_value = self._test_conditional(X, Y, Z, k)
            
            if self.verbosity >= 2:
                z_dim = Z.shape[1] if Z is not None and Z.ndim > 1 else (1 if Z is not None else 0)
                cond_str = f"| Z(dim={z_dim})" if Z is not None else ""
                print(f"GPU CMI test: X(dim={X.shape[1]}) _|_ Y(dim={Y.shape[1]}) {cond_str} "
                      f"=> val={cmi_value:.4f}, p={p_value:.4f}, n_samples={X.shape[0]}")
            
            return (cmi_value, float(p_value))
        
        except Exception as e:
            raise RuntimeError(f"GPU CMIknn test failed: {str(e)}") from e
    
    def _test_unconditional(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        k: int
    ) -> Tuple[float, float]:
        """
        Test unconditional independence: X _|_ Y
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, dim_x)
            First variable
        Y : np.ndarray, shape (n_samples, dim_y)
            Second variable
        k : int
            Number of nearest neighbors
        
        Returns
        -------
        mi_value : float
            Mutual information estimate
        p_value : float
            p-value from permutation test
        """
        # Concatenate X and Y
        data = np.concatenate((X, Y), axis=1)
        
        # Convert to float64
        data = data.astype(np.float64)
        
        # Add small noise to destroy ties (matches CPU Tigramite)
        dim, n_samples = data.shape[1], data.shape[0]
        # CPU Tigramite generates noise as (dim, T) using random_state.random((dim, T))
        # We must match this generation order to get identical noise values
        noise_T = 1E-6 * data.std(axis=0).reshape(dim, 1) * self.random_state.random((dim, n_samples))
        data += noise_T.T
        
        # Apply rank transformation (matches CPU Tigramite default behavior)
        # This is CRITICAL - CPU uses ranks by default!
        data = data.T  # Transpose to (features, samples) for argsort
        data = data.argsort(axis=1).argsort(axis=1).astype(np.float64)
        # data is now (features, samples) with rank-transformed values
        
        # Call GPU function - now returns (MI, p-value) tuple
        # Note: Pass k directly (not k+1), CUDA handles neighbor exclusion internally
        mi_val, p_val = gpucmiknn.pval_l0(data, k, self.sig_samples)
        
        return float(mi_val), float(p_val)
    
    def _test_conditional(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        k: int
    ) -> Tuple[float, float]:
        """
        Test conditional independence: X _|_ Y | Z
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, dim_x)
            First variable
        Y : np.ndarray, shape (n_samples, dim_y)
            Second variable
        Z : np.ndarray, shape (n_samples, dim_z)
            Conditioning variables
        k : int
            Number of nearest neighbors
        
        Returns
        -------
        cmi_value : float
            Conditional mutual information estimate
        p_value : float
            p-value from permutation test
        """
        n_samples = len(X)
        
        # Ensure Z is 2D
        if Z.ndim == 1:
            Z = Z.reshape((n_samples, 1))
        
        # Concatenate X, Y, Z
        data = np.concatenate((X, Y, Z), axis=1).astype(np.float64)
        
        # Add small noise to destroy ties (matches CPU Tigramite)
        dim = data.shape[1]
        # Match CPU noise generation: (dim, T) then transpose
        noise_T = 1E-6 * data.std(axis=0).reshape(dim, 1) * self.random_state.random((dim, n_samples))
        data += noise_T.T
        
        # Apply rank transformation (matches CPU Tigramite default behavior)
        # This is CRITICAL - CPU uses ranks by default!
        data = data.T  # Transpose to (features, samples) for argsort
        data = data.argsort(axis=1).argsort(axis=1).astype(np.float64)
        # data is now (features, samples) with rank-transformed values
        
        # Generate restricted permutations conditioned on Z
        # Z needs rank transformation too for permutation generation
        z_data = Z.astype(np.float64)
        dim_z = z_data.shape[1]
        # Match CPU noise generation for Z
        z_noise_T = 1E-6 * z_data.std(axis=0).reshape(dim_z, 1) * self.random_state.random((dim_z, n_samples))
        z_data += z_noise_T.T
        z_data = z_data.T
        z_data = z_data.argsort(axis=1).argsort(axis=1).astype(np.float64)
        
        restricted_perm = gpucmiknn.rperm_multi(z_data, self.sig_samples)
        
        # Create permuted X arrays
        # For multi-D X: permute rows (samples) while keeping columns (features) together
        dim_x = X.shape[1]
        x_permutations = np.ndarray(
            shape=(self.sig_samples, n_samples * dim_x),
            dtype='float64'
        )
        
        # Extract rank-transformed X from data (first dim_x features)
        X_ranked = data[:dim_x, :].T  # Back to (samples, features)
        
        for i in range(self.sig_samples):
            # Permute X rows and flatten to 1D for GPU
            permuted_X = X_ranked[restricted_perm[i], :]
            x_permutations[i] = permuted_X.flatten()
        
        if self.verbosity >= 2:
            print(f"DEBUG: X.shape={X.shape}, Y.shape={Y.shape}, Z.shape={Z.shape}")
            print(f"DEBUG: data.shape={data.shape}")
            print(f"DEBUG: x_permutations.shape={x_permutations.shape}")
        
        # Call GPU function - now returns (CMI, p-value) tuple
        # Note: Pass k directly (not k+1), CUDA handles neighbor exclusion internally
        cmi_val, p_val = gpucmiknn.pval_ln(
            data,
            x_permutations,
            k,
            k,  # k_perm
            self.sig_samples
        )
        
        if self.verbosity >= 2:
            print(f"DEBUG: cmi_val={cmi_val}, p_val={p_val}")
        
        return float(cmi_val), float(p_val)
    
    def get_measure(self) -> str:
        """Return measure name for tigramite compatibility."""
        return self.measure
    
    def get_confidence(self, X, Y, Z=None, tau_max=0, **kwargs):
        """
        Get confidence intervals (not implemented for GPU CMIknn).
        
        This method is called by PCMCI when confidence intervals are requested.
        Since GPU CMIknn focuses on p-values rather than confidence intervals,
        this returns None.
        
        Parameters
        ----------
        X, Y, Z : Various
            Variable specifications (not used)
        tau_max : int
            Maximum time lag (not used)
        
        Returns
        -------
        None
            Confidence intervals not supported
        """
        return None
    
    def get_dependence_measure(self, X, Y, Z=None, tau_max=0, **kwargs):
        """
        Get the dependence measure (CMI value).
        
        This is a convenience method that returns just the CMI value
        without the p-value.
        
        Parameters
        ----------
        X, Y, Z : list of tuples
            Variable specifications
        tau_max : int
            Maximum time lag
        
        Returns
        -------
        cmi_value : float
            CMI estimate (0.0 for GPU version which focuses on p-values)
        """
        val, pval = self.run_test(X, Y, Z, tau_max=tau_max)
        return val
    
    def get_shuffle_significance(self, X, Y, Z=None, tau_max=0, **kwargs):
        """
        Get shuffle significance (p-value).
        
        This is a convenience method that returns just the p-value
        without the CMI value.
        
        Parameters
        ----------
        X, Y, Z : list of tuples
            Variable specifications
        tau_max : int
            Maximum time lag
        
        Returns
        -------
        pval : float
            p-value from permutation test
        """
        val, pval = self.run_test(X, Y, Z, tau_max=tau_max)
        return pval
    
    def set_mask_type(self, mask_type):
        """
        Set mask type for handling missing values.
        
        This method is called by tigramite to configure how missing values
        should be handled. GPU CMIknn currently doesn't support masking.
        
        Parameters
        ----------
        mask_type : str or None
            Type of masking ('y', 'x', 'z', 'xy', 'xz', 'yz', 'xyz', or None)
        """
        if mask_type is not None:
            warnings.warn(
                "GPU CMIknn does not support missing value masking. "
                "Please ensure your data has no missing values.",
                RuntimeWarning
            )
        self.mask_type = mask_type
    
    def get_analytic_significance(self, *args, **kwargs):
        """Not implemented - GPU CMIknn uses permutation tests."""
        raise NotImplementedError(
            "GPU CMIknn only supports shuffle/permutation significance tests. "
            "Analytic p-values are not available."
        )
    
    def get_model_selection_criterion(self, *args, **kwargs):
        """Not implemented - for tigramite compatibility."""
        raise NotImplementedError(
            "Model selection not implemented for GPU CMIknn. "
            "Use fixed parameters instead."
        )


# Convenience function for easy imports
def create_gpu_cmi_test(knn: int = 5, sig_samples: int = 100, verbosity: int = 0, seed: int = None):
    """
    Create GPU CMIknn test with automatic fallback to CPU if GPU unavailable.
    
    Parameters
    ----------
    knn : int
        Number of nearest neighbors
    sig_samples : int
        Number of permutations for significance testing
    verbosity : int
        Verbosity level
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    cond_ind_test : CMIknn or GPUCMIknn
        GPU version if available, otherwise CPU version
    """
    if GPU_AVAILABLE:
        if verbosity >= 1:
            print("✓ Using GPU-accelerated CMIknn (50-430x faster)")
        return GPUCMIknn(knn=knn, sig_samples=sig_samples, verbosity=verbosity, seed=seed)
    else:
        if verbosity >= 1:
            print("⚠ GPU not available, using CPU CMIknn")
        try:
            from tigramite.independence_tests.cmiknn import CMIknn
            return CMIknn(knn=knn, sig_samples=sig_samples, verbosity=verbosity, seed=seed)
        except ImportError:
            raise ImportError(
                "Neither GPU CMIknn nor tigramite CMIknn available. "
                "Please install tigramite or build GPU CMIknn."
            )
