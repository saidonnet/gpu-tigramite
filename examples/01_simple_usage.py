"""
Simple GPU-Tigramite Example
=============================

Basic example showing how to use GPU-accelerated CMIknn with Tigramite PCMCI.

This example demonstrates:
1. Creating synthetic time series data
2. Using GPU-accelerated independence test
3. Running PCMCI causal discovery
4. Visualizing results

Author: Your Name (2025)
License: GPL-3.0
"""

import numpy as np
from gpu_tigramite import GPUCMIknn
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("GPU-Tigramite Simple Example")
print("=" * 70)

# 1. Create synthetic time series data
print("\n[1/4] Generating synthetic time series data...")
n_samples = 1000
n_vars = 5

# Create simple autoregressive process with some causal links
data = np.zeros((n_samples, n_vars))
data[0, :] = np.random.randn(n_vars)

for t in range(1, n_samples):
    # X0(t) = 0.5 * X0(t-1) + noise
    data[t, 0] = 0.5 * data[t-1, 0] + np.random.randn()
    
    # X1(t) = 0.3 * X0(t-1) + noise (causal link: X0 -> X1)
    data[t, 1] = 0.3 * data[t-1, 0] + np.random.randn()
    
    # X2(t) = 0.4 * X1(t-1) + noise (causal link: X1 -> X2)
    data[t, 2] = 0.4 * data[t-1, 1] + np.random.randn()
    
    # X3(t) = independent noise
    data[t, 3] = np.random.randn()
    
    # X4(t) = 0.2 * X0(t-1) + 0.2 * X2(t-1) + noise (links: X0 -> X4, X2 -> X4)
    data[t, 4] = 0.2 * data[t-1, 0] + 0.2 * data[t-1, 2] + np.random.randn()

print(f"✓ Generated data: {n_samples} samples × {n_vars} variables")
print(f"  True causal structure:")
print(f"    X0 -> X1 (strength: 0.3)")
print(f"    X1 -> X2 (strength: 0.4)")
print(f"    X0 -> X4 (strength: 0.2)")
print(f"    X2 -> X4 (strength: 0.2)")
print(f"    X3 is independent")

# 2. Create Tigramite dataframe
print("\n[2/4] Creating Tigramite dataframe...")
var_names = [f'X{i}' for i in range(n_vars)]
dataframe = pp.DataFrame(
    data,
    var_names=var_names
)
print(f"✓ Dataframe created with variables: {var_names}")

# 3. Create GPU-accelerated independence test
print("\n[3/4] Initializing GPU-accelerated CMIknn...")
cond_ind_test = GPUCMIknn(
    knn=5,                  # Number of nearest neighbors
    sig_samples=100,        # Number of permutations for significance
    verbosity=1             # Show initialization info
)

# 4. Run PCMCI causal discovery
print("\n[4/4] Running PCMCI causal discovery...")
pcmci = PCMCI(
    dataframe=dataframe,
    cond_ind_test=cond_ind_test,
    verbosity=1
)

results = pcmci.run_pcmci(
    tau_max=2,              # Maximum time lag
    pc_alpha=0.05          # Significance threshold
)

print("\n" + "=" * 70)
print("Results")
print("=" * 70)

# Print discovered links
print("\nDiscovered causal links (p < 0.05):")
p_matrix = results['p_matrix']
val_matrix = results['val_matrix']

for i in range(n_vars):
    for j in range(n_vars):
        for tau in range(p_matrix.shape[2]):
            if p_matrix[i, j, tau] < 0.05 and tau > 0:
                print(f"  {var_names[j]} (t-{tau}) -> {var_names[i]} (t)  "
                      f"(p-value: {p_matrix[i, j, tau]:.4f})")

print("\n" + "=" * 70)
print("✓ Analysis complete!")
print("=" * 70)

# Optional: Visualize results (requires matplotlib)
try:
    import matplotlib.pyplot as plt
    
    print("\nGenerating visualization...")
    
    # Plot time series
    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 8), sharex=True)
    for i in range(n_vars):
        axes[i].plot(data[:200, i])  # Plot first 200 samples
        axes[i].set_ylabel(var_names[i])
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time')
    plt.suptitle('Synthetic Time Series Data')
    plt.tight_layout()
    plt.savefig('gpu_tigramite_timeseries.png', dpi=150)
    print("✓ Saved plot: gpu_tigramite_timeseries.png")
    
    # Plot causal graph using Tigramite's built-in function
    from tigramite import plotting as tp
    tp.plot_graph(
        val_matrix=val_matrix,
        graph=results['graph'],
        var_names=var_names,
        figsize=(8, 6)
    )
    plt.savefig('gpu_tigramite_graph.png', dpi=150)
    print("✓ Saved plot: gpu_tigramite_graph.png")
    
except ImportError:
    print("\nNote: Install matplotlib to generate visualizations")
    print("  pip install matplotlib")

print("\nDone! GPU-accelerated causal discovery completed successfully.")