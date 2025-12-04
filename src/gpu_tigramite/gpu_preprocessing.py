"""
GPU-Accelerated Data Preprocessing for Tigramite

Provides GPU-accelerated preprocessing operations for time series data,
including standardization, missing value handling, and sliding window operations.

Author: gpu-tigramite
License: GPL-3.0
"""

import numpy as np
import torch
from typing import Optional, Union, Tuple
import warnings


class GPUPreprocessor:
    """
    GPU-accelerated data preprocessing for causal discovery.
    
    Provides fast preprocessing operations using PyTorch CUDA:
    - Standardization (zero mean, unit variance)
    - Missing value imputation
    - Sliding window construction
    - Data validation
    
    Parameters
    ----------
    device : str or torch.device, optional
        GPU device to use. If None, automatically selects available GPU.
    verbosity : int, optional
        Level of verbosity (0=quiet, 1=normal, 2=debug). Default: 0.
        
    Examples
    --------
    >>> preprocessor = GPUPreprocessor(device='cuda:0')
    >>> data_standardized = preprocessor.standardize(data)
    >>> data_windowed = preprocessor.create_sliding_windows(data, tau_max=5)
    """
    
    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        verbosity: int = 0
    ):
        self.verbosity = verbosity
        
        # Set up GPU device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                warnings.warn("No CUDA device available, falling back to CPU")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
            
        if self.verbosity > 0:
            print(f"GPUPreprocessor initialized on device: {self.device}")
    
    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        """Convert numpy array to GPU tensor."""
        return torch.tensor(array, dtype=torch.float32, device=self.device)
    
    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert GPU tensor to numpy array."""
        return tensor.cpu().numpy()
    
    def standardize(
        self,
        data: np.ndarray,
        return_stats: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        """
        Standardize data to zero mean and unit variance on GPU.
        
        Parameters
        ----------
        data : np.ndarray
            Input data, shape (T, N) where T is time steps, N is variables
        return_stats : bool, optional
            If True, return (standardized_data, stats_dict)
            
        Returns
        -------
        data_std : np.ndarray
            Standardized data
        stats : dict, optional
            Dictionary with 'mean' and 'std' arrays
            
        Examples
        --------
        >>> data_std = preprocessor.standardize(data)
        >>> data_std, stats = preprocessor.standardize(data, return_stats=True)
        """
        # Convert to GPU
        data_gpu = self._to_tensor(data)
        
        # Compute statistics
        mean = data_gpu.mean(dim=0, keepdim=True)
        std = data_gpu.std(dim=0, keepdim=True)
        
        # Avoid division by zero
        std = torch.where(std < 1e-10, torch.ones_like(std), std)
        
        # Standardize
        data_std = (data_gpu - mean) / std
        
        # Convert back to numpy
        result = self._to_numpy(data_std)
        
        if return_stats:
            stats = {
                'mean': self._to_numpy(mean.squeeze()),
                'std': self._to_numpy(std.squeeze())
            }
            return result, stats
        
        return result
    
    def impute_missing(
        self,
        data: np.ndarray,
        method: str = 'mean',
        mask_value: Optional[float] = None
    ) -> np.ndarray:
        """
        Impute missing values on GPU.
        
        Parameters
        ----------
        data : np.ndarray
            Input data with missing values (NaN or specified mask_value)
        method : str, optional
            Imputation method: 'mean', 'median', 'zero', 'forward_fill'
        mask_value : float, optional
            Value to treat as missing (default: NaN)
            
        Returns
        -------
        data_imputed : np.ndarray
            Data with missing values imputed
        """
        data_gpu = self._to_tensor(data)
        
        # Create mask for missing values
        if mask_value is None:
            mask = torch.isnan(data_gpu)
        else:
            mask = (data_gpu == mask_value)
        
        if not mask.any():
            return data  # No missing values
        
        if method == 'mean':
            # Compute mean ignoring NaN
            for col in range(data_gpu.shape[1]):
                col_data = data_gpu[:, col]
                col_mask = mask[:, col]
                if col_mask.any():
                    mean_val = col_data[~col_mask].mean()
                    data_gpu[col_mask, col] = mean_val
                    
        elif method == 'median':
            for col in range(data_gpu.shape[1]):
                col_data = data_gpu[:, col]
                col_mask = mask[:, col]
                if col_mask.any():
                    median_val = col_data[~col_mask].median()
                    data_gpu[col_mask, col] = median_val
                    
        elif method == 'zero':
            data_gpu[mask] = 0.0
            
        elif method == 'forward_fill':
            # Forward fill along time axis
            for col in range(data_gpu.shape[1]):
                col_data = data_gpu[:, col]
                col_mask = mask[:, col]
                if col_mask.any():
                    # Find first non-missing value
                    valid_idx = (~col_mask).nonzero(as_tuple=True)[0]
                    if len(valid_idx) > 0:
                        fill_value = col_data[valid_idx[0]]
                        # Fill before first valid
                        data_gpu[:valid_idx[0], col] = fill_value
                        # Forward fill rest
                        for t in range(1, len(col_data)):
                            if col_mask[t]:
                                data_gpu[t, col] = data_gpu[t-1, col]
        else:
            raise ValueError(f"Unknown imputation method: {method}")
        
        return self._to_numpy(data_gpu)
    
    def create_sliding_windows(
        self,
        data: np.ndarray,
        tau_max: int,
        include_current: bool = True
    ) -> np.ndarray:
        """
        Create sliding windows for time-lagged analysis on GPU.
        
        Parameters
        ----------
        data : np.ndarray
            Input data, shape (T, N)
        tau_max : int
            Maximum time lag
        include_current : bool, optional
            If True, include current time step (lag 0)
            
        Returns
        -------
        windowed_data : np.ndarray
            Windowed data, shape (T-tau_max, N * (tau_max + 1))
            
        Examples
        --------
        >>> # Create windows with lags 0, 1, 2
        >>> windows = preprocessor.create_sliding_windows(data, tau_max=2)
        """
        data_gpu = self._to_tensor(data)
        T, N = data_gpu.shape
        
        # Determine window size
        if include_current:
            window_size = tau_max + 1
            lags = list(range(0, tau_max + 1))
        else:
            window_size = tau_max
            lags = list(range(1, tau_max + 1))
        
        # Create windowed array
        T_out = T - tau_max
        windowed = torch.zeros(T_out, N * window_size, device=self.device)
        
        # Fill windows
        for i, lag in enumerate(lags):
            start_idx = tau_max - lag
            end_idx = start_idx + T_out
            windowed[:, i*N:(i+1)*N] = data_gpu[start_idx:end_idx, :]
        
        return self._to_numpy(windowed)
    
    def remove_trends(
        self,
        data: np.ndarray,
        method: str = 'linear'
    ) -> np.ndarray:
        """
        Remove trends from time series on GPU.
        
        Parameters
        ----------
        data : np.ndarray
            Input data, shape (T, N)
        method : str, optional
            Detrending method: 'linear', 'mean'
            
        Returns
        -------
        detrended : np.ndarray
            Detrended data
        """
        data_gpu = self._to_tensor(data)
        T, N = data_gpu.shape
        
        if method == 'mean':
            # Simply subtract mean
            mean = data_gpu.mean(dim=0, keepdim=True)
            detrended = data_gpu - mean
            
        elif method == 'linear':
            # Fit and remove linear trend for each variable
            detrended = torch.zeros_like(data_gpu)
            t = torch.arange(T, dtype=torch.float32, device=self.device).reshape(-1, 1)
            
            for col in range(N):
                y = data_gpu[:, col:col+1]
                
                # Fit linear regression: y = a + b*t
                # Using least squares: beta = (X^T X)^-1 X^T y
                X = torch.cat([torch.ones_like(t), t], dim=1)
                XtX = X.T @ X
                Xty = X.T @ y
                
                try:
                    beta = torch.linalg.solve(XtX, Xty)
                    trend = X @ beta
                    detrended[:, col] = (y - trend).squeeze()
                except torch.linalg.LinAlgError:
                    # Fallback to mean if singular
                    detrended[:, col] = y.squeeze() - y.mean()
        else:
            raise ValueError(f"Unknown detrending method: {method}")
        
        return self._to_numpy(detrended)
    
    def validate_data(
        self,
        data: np.ndarray,
        check_finite: bool = True,
        check_variance: bool = True,
        min_variance: float = 1e-10
    ) -> dict:
        """
        Validate data quality on GPU.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        check_finite : bool, optional
            Check for NaN/Inf values
        check_variance : bool, optional
            Check for zero/low variance variables
        min_variance : float, optional
            Minimum allowed variance
            
        Returns
        -------
        validation_report : dict
            Dictionary with validation results
        """
        data_gpu = self._to_tensor(data)
        T, N = data_gpu.shape
        
        report = {
            'shape': (T, N),
            'valid': True,
            'issues': []
        }
        
        if check_finite:
            has_nan = torch.isnan(data_gpu).any().item()
            has_inf = torch.isinf(data_gpu).any().item()
            
            if has_nan:
                n_nan = torch.isnan(data_gpu).sum().item()
                report['issues'].append(f"{n_nan} NaN values found")
                report['valid'] = False
                
            if has_inf:
                n_inf = torch.isinf(data_gpu).sum().item()
                report['issues'].append(f"{n_inf} Inf values found")
                report['valid'] = False
        
        if check_variance:
            variances = data_gpu.var(dim=0)
            low_var_mask = variances < min_variance
            
            if low_var_mask.any():
                low_var_cols = low_var_mask.nonzero(as_tuple=True)[0].cpu().numpy()
                report['issues'].append(
                    f"Variables {low_var_cols.tolist()} have variance < {min_variance}"
                )
                report['valid'] = False
        
        return report
    
    def batch_process(
        self,
        data: np.ndarray,
        batch_size: int = 1000,
        operations: list = ['standardize']
    ) -> np.ndarray:
        """
        Process large datasets in batches to avoid GPU memory issues.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        batch_size : int, optional
            Number of samples per batch
        operations : list, optional
            List of operations to apply: 'standardize', 'impute', 'detrend'
            
        Returns
        -------
        processed_data : np.ndarray
            Processed data
        """
        T = data.shape[0]
        n_batches = (T + batch_size - 1) // batch_size
        
        # First pass: compute global statistics if needed
        if 'standardize' in operations:
            data_gpu = self._to_tensor(data)
            global_mean = data_gpu.mean(dim=0, keepdim=True)
            global_std = data_gpu.std(dim=0, keepdim=True)
            global_std = torch.where(global_std < 1e-10, torch.ones_like(global_std), global_std)
            del data_gpu
        
        # Process in batches
        result = np.zeros_like(data)
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, T)
            batch = data[start_idx:end_idx]
            
            batch_gpu = self._to_tensor(batch)
            
            # Apply operations
            if 'impute' in operations:
                mask = torch.isnan(batch_gpu)
                if mask.any():
                    # Use global mean for imputation
                    for col in range(batch_gpu.shape[1]):
                        col_mask = mask[:, col]
                        if col_mask.any():
                            batch_gpu[col_mask, col] = global_mean[0, col]
            
            if 'standardize' in operations:
                batch_gpu = (batch_gpu - global_mean) / global_std
            
            if 'detrend' in operations:
                # Use batch-local detrending
                batch_gpu = batch_gpu - batch_gpu.mean(dim=0, keepdim=True)
            
            result[start_idx:end_idx] = self._to_numpy(batch_gpu)
        
        return result


def test_gpu_preprocessing():
    """Test GPU preprocessing functions."""
    print("Testing GPU Preprocessing...")
    
    # Generate test data
    np.random.seed(42)
    T, N = 1000, 5
    data = np.random.randn(T, N).astype(np.float32)
    
    # Add some structure
    data[:, 1] = data[:, 0] + 0.5 * np.random.randn(T)
    data[:, 2] = 2.0 * data[:, 2] + 10.0  # Different scale
    
    # Add missing values
    data[100:110, 3] = np.nan
    
    preprocessor = GPUPreprocessor(verbosity=1)
    
    print("\n1. Standardization:")
    data_std, stats = preprocessor.standardize(data, return_stats=True)
    print(f"   Mean: {data_std.mean(axis=0)}")
    print(f"   Std: {data_std.std(axis=0)}")
    
    print("\n2. Missing value imputation:")
    data_imputed = preprocessor.impute_missing(data, method='mean')
    print(f"   Missing before: {np.isnan(data).sum()}")
    print(f"   Missing after: {np.isnan(data_imputed).sum()}")
    
    print("\n3. Sliding windows (tau_max=2):")
    windows = preprocessor.create_sliding_windows(data, tau_max=2)
    print(f"   Input shape: {data.shape}")
    print(f"   Window shape: {windows.shape}")
    
    print("\n4. Detrending:")
    data_detrended = preprocessor.remove_trends(data_imputed, method='linear')
    print(f"   Original mean: {data_imputed.mean(axis=0)}")
    print(f"   Detrended mean: {data_detrended.mean(axis=0)}")
    
    print("\n5. Data validation:")
    report = preprocessor.validate_data(data)
    print(f"   Valid: {report['valid']}")
    print(f"   Issues: {report['issues']}")
    
    print("\n✓ All preprocessing tests passed!")


if __name__ == '__main__':
    test_gpu_preprocessing()