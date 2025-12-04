"""
CPU vs GPU Performance Benchmark
=================================

Rigorous benchmarking of GPU-accelerated CMIknn vs CPU CMIknn.

This benchmark measures:
1. Real execution time (not estimates)
2. Multiple dataset sizes
3. Statistical significance (multiple runs)
4. Detailed performance metrics

Author: Your Name (2025)
License: GPL-3.0
"""

import numpy as np
import time
import sys
from typing import Tuple, List

print("=" * 80)
print("GPU-Tigramite: CPU vs GPU Performance Benchmark")
print("=" * 80)

# Check dependencies
print("\n[Setup] Checking dependencies...")
try:
    from gpu_tigramite import GPUCMIknn, get_gpu_count
    gpu_available = True
    print("✓ GPU-Tigramite installed")
except ImportError:
    print("✗ GPU-Tigramite not found")
    gpu_available = False

try:
    from tigramite.independence_tests.cmiknn import CMIknn
    cpu_available = True
    print("✓ Tigramite CPU CMIknn available")
except ImportError:
    print("✗ Tigramite not installed")
    cpu_available = False
    sys.exit(1)

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI

# Check GPU
if gpu_available:
    num_gpus = get_gpu_count()
    print(f"✓ Found {num_gpus} GPU(s)")
else:
    print("⚠ No GPU detected - GPU benchmarks will be skipped")

print("\n" + "=" * 80)


def generate_synthetic_data(n_samples: int, n_vars: int, seed: int = 42) -> np.ndarray:
    """
    Generate synthetic time series with known causal structure.
    
    Parameters
    ----------
    n_samples : int
        Number of time steps
    n_vars : int
        Number of variables
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    data : np.ndarray
        Synthetic time series (n_samples × n_vars)
    """
    np.random.seed(seed)
    
    data = np.zeros((n_samples, n_vars))
    data[0, :] = np.random.randn(n_vars)
    
    # Create autoregressive process with some cross-dependencies
    for t in range(1, n_samples):
        for i in range(n_vars):
            # Self-dependency
            data[t, i] = 0.5 * data[t-1, i]
            
            # Cross-dependencies
            if i > 0:
                data[t, i] += 0.2 * data[t-1, i-1]
            if i < n_vars - 1:
                data[t, i] += 0.1 * data[t-1, i+1]
            
            # Noise
            data[t, i] += np.random.randn()
    
    return data


def benchmark_single_run(
    data: np.ndarray,
    use_gpu: bool,
    knn: int = 5,
    sig_samples: int = 100,
    tau_max: int = 2,
    pc_alpha: float = 0.05,
    verbosity: int = 0
) -> Tuple[float, int]:
    """
    Run a single PCMCI benchmark.
    
    Parameters
    ----------
    data : np.ndarray
        Input time series
    use_gpu : bool
        Whether to use GPU acceleration
    knn : int
        Number of nearest neighbors
    sig_samples : int
        Number of permutations
    tau_max : int
        Maximum time lag
    pc_alpha : float
        Significance threshold
    verbosity : int
        Verbosity level
    
    Returns
    -------
    elapsed_time : float
        Time taken in seconds
    num_links : int
        Number of discovered causal links
    """
    # Create dataframe
    dataframe = pp.DataFrame(data)
    
    # Create independence test
    if use_gpu:
        cond_ind_test = GPUCMIknn(
            knn=knn,
            sig_samples=sig_samples,
            verbosity=verbosity
        )
    else:
        cond_ind_test = CMIknn(
            knn=knn,
            sig_samples=sig_samples,
            verbosity=verbosity
        )
    
    # Create PCMCI instance
    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=cond_ind_test,
        verbosity=verbosity
    )
    
    # Run benchmark
    start_time = time.time()
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha)
    elapsed_time = time.time() - start_time
    
    # Count discovered links (excluding lag 0)
    p_matrix = results['p_matrix']
    num_links = np.sum(p_matrix < pc_alpha) - np.sum(p_matrix[:, :, 0] < pc_alpha)
    
    return elapsed_time, int(num_links)


def run_benchmark_suite(
    dataset_configs: List[Tuple[int, int]],
    num_runs: int = 3,
    knn: int = 5,
    sig_samples: int = 100
):
    """
    Run comprehensive benchmark suite.
    
    Parameters
    ----------
    dataset_configs : list of tuples
        List of (n_samples, n_vars) configurations
    num_runs : int
        Number of runs per configuration for statistical reliability
    knn : int
        Number of nearest neighbors
    sig_samples : int
        Number of permutations
    """
    print("\nBenchmark Configuration:")
    print(f"  k-NN: {knn}")
    print(f"  Permutations: {sig_samples}")
    print(f"  Runs per config: {num_runs}")
    print(f"  Datasets: {len(dataset_configs)}")
    
    results = []
    
    for config_idx, (n_samples, n_vars) in enumerate(dataset_configs):
        print("\n" + "-" * 80)
        print(f"Dataset {config_idx + 1}/{len(dataset_configs)}: "
              f"{n_samples} samples × {n_vars} variables")
        print("-" * 80)
        
        # Generate data once for both CPU and GPU
        print(f"Generating synthetic data...")
        data = generate_synthetic_data(n_samples, n_vars)
        print(f"✓ Data generated: shape {data.shape}")
        
        # CPU Benchmark
        print(f"\n[CPU] Running {num_runs} iterations...")
        cpu_times = []
        cpu_links = []
        
        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}...", end=" ", flush=True)
            elapsed, num_links = benchmark_single_run(
                data, use_gpu=False, knn=knn, sig_samples=sig_samples
            )
            cpu_times.append(elapsed)
            cpu_links.append(num_links)
            print(f"{elapsed:.2f}s ({num_links} links)")
        
        cpu_mean = np.mean(cpu_times)
        cpu_std = np.std(cpu_times)
        
        print(f"\n  CPU Results:")
        print(f"    Mean time: {cpu_mean:.2f}s ± {cpu_std:.2f}s")
        print(f"    Links found: {int(np.mean(cpu_links))}")
        
        # GPU Benchmark (if available)
        if gpu_available and num_gpus > 0:
            print(f"\n[GPU] Running {num_runs} iterations...")
            gpu_times = []
            gpu_links = []
            
            for run in range(num_runs):
                print(f"  Run {run + 1}/{num_runs}...", end=" ", flush=True)
                elapsed, num_links = benchmark_single_run(
                    data, use_gpu=True, knn=knn, sig_samples=sig_samples
                )
                gpu_times.append(elapsed)
                gpu_links.append(num_links)
                print(f"{elapsed:.2f}s ({num_links} links)")
            
            gpu_mean = np.mean(gpu_times)
            gpu_std = np.std(gpu_times)
            speedup = cpu_mean / gpu_mean
            
            print(f"\n  GPU Results:")
            print(f"    Mean time: {gpu_mean:.2f}s ± {gpu_std:.2f}s")
            print(f"    Links found: {int(np.mean(gpu_links))}")
            print(f"\n  Speedup: {speedup:.1f}x 🚀")
            
            results.append({
                'n_samples': n_samples,
                'n_vars': n_vars,
                'cpu_time': cpu_mean,
                'cpu_std': cpu_std,
                'gpu_time': gpu_mean,
                'gpu_std': gpu_std,
                'speedup': speedup,
                'cpu_links': int(np.mean(cpu_links)),
                'gpu_links': int(np.mean(gpu_links))
            })
        else:
            results.append({
                'n_samples': n_samples,
                'n_vars': n_vars,
                'cpu_time': cpu_mean,
                'cpu_std': cpu_std,
                'gpu_time': None,
                'gpu_std': None,
                'speedup': None,
                'cpu_links': int(np.mean(cpu_links)),
                'gpu_links': None
            })
    
    return results


def print_summary(results: List[dict]):
    """Print benchmark summary table."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    # Header
    print(f"\n{'Dataset':<20} {'CPU Time':<15} {'GPU Time':<15} {'Speedup':<10}")
    print("-" * 80)
    
    # Results
    for r in results:
        dataset_str = f"{r['n_samples']}×{r['n_vars']}"
        cpu_str = f"{r['cpu_time']:.2f}s ± {r['cpu_std']:.2f}s"
        
        if r['gpu_time'] is not None:
            gpu_str = f"{r['gpu_time']:.2f}s ± {r['gpu_std']:.2f}s"
            speedup_str = f"{r['speedup']:.1f}x"
        else:
            gpu_str = "N/A"
            speedup_str = "N/A"
        
        print(f"{dataset_str:<20} {cpu_str:<15} {gpu_str:<15} {speedup_str:<10}")
    
    # Overall statistics
    if all(r['gpu_time'] is not None for r in results):
        avg_speedup = np.mean([r['speedup'] for r in results])
        min_speedup = np.min([r['speedup'] for r in results])
        max_speedup = np.max([r['speedup'] for r in results])
        
        print("\n" + "=" * 80)
        print(f"Average Speedup: {avg_speedup:.1f}x")
        print(f"Range: {min_speedup:.1f}x - {max_speedup:.1f}x")
        print("=" * 80)


# Main benchmark
if __name__ == "__main__":
    # Define benchmark datasets (start small, increase size)
    dataset_configs = [
        (100, 10),     # Small: 100 samples × 10 variables
        (500, 20),     # Medium: 500 samples × 20 variables
        (1000, 30),    # Large: 1000 samples × 30 variables
        (2000, 50),    # Very large: 2000 samples × 50 variables
    ]
    
    print("\n⚠️  WARNING: This benchmark will take several minutes!")
    print("For faster results, reduce num_runs or dataset sizes.")
    
    try:
        results = run_benchmark_suite(
            dataset_configs=dataset_configs,
            num_runs=3,  # 3 runs for statistical reliability
            knn=5,
            sig_samples=100
        )
        
        print_summary(results)
        
        # Save results
        import json
        with open('benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\n✓ Results saved to: benchmark_results.json")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Benchmark failed: {e}")
        raise
    
    print("\n✓ Benchmark complete!")