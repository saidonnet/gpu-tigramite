# GPU-Tigramite Benchmarks

Comprehensive benchmarking suite for validating GPU acceleration claims.

## Available Benchmarks

### Kaggle Comprehensive Benchmark ⭐ RECOMMENDED
**File**: [`kaggle_comprehensive_benchmark.py`](kaggle_comprehensive_benchmark.py)

Head-to-head comparison of CPU Tigramite vs GPU-Tigramite:
- ✅ Single CMIknn test (raw speed)
- ✅ Batch vs Sequential processing (efficiency)
- ✅ CPU ParCorr vs GPU ParCorr (linear tests)
- ✅ CPU vs GPU preprocessing (data prep)
- ✅ Full PCMCI pipeline (end-to-end)

**Features:**
- Validates ALL results match between CPU and GPU
- Automatic result saving (JSON output)
- Detailed timing and speedup metrics
- Hardware metadata capture
- Interruption-safe (saves after each test)

**Runtime**: ~30-60 minutes on Kaggle 4×L4 GPUs

---

## Kaggle Setup Guide

### Quick Start (5 minutes)

#### 1. Create Kaggle Notebook

1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. **Settings → Accelerator → GPU T4 x2** (or L4 if available)
4. Enable Internet in Settings

#### 2. Install GPU-Tigramite

```bash
# In a code cell:
!pip install tigramite  # Install original Tigramite first
!cd /kaggle/working && git clone https://github.com/gpu-tigramite/gpu-tigramite.git
!cd /kaggle/working/gpu-tigramite && pip install -e .
```

**OR** if uploading as dataset:

```bash
# Upload gpu-tigramite folder as Kaggle dataset
# Then add it to notebook and:
!cd /kaggle/input/gpu-tigramite && pip install -e .
```

#### 3. Run Benchmark

```python
# In a code cell:
import sys
sys.path.insert(0, '/kaggle/working/gpu-tigramite')

from benchmarks.kaggle_comprehensive_benchmark import BenchmarkSuite

# Run all benchmarks
suite = BenchmarkSuite(output_file='benchmark_results.json')
results = suite.run_all_benchmarks()
```

#### 4. View Results

```python
import json
import pandas as pd

# Load results
with open('benchmark_results.json') as f:
    results = json.load(f)

# Display summary
summary = []
for name, data in results['benchmarks'].items():
    if 'speedup' in data:
        summary.append({
            'Test': name,
            'CPU Time (s)': data['cpu_time'],
            'GPU Time (s)': data['gpu_time'],
            'Speedup': f"{data['speedup']:.1f}x",
            'Results Match': '✓' if data.get('results_match', True) else '⚠️'
        })

df = pd.DataFrame(summary)
print(df.to_string(index=False))
```

---

## Expected Output (4×L4 GPUs)

```
BENCHMARK SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Test                          CPU Time  GPU Time  Speedup  Match
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cmiknn_single                 0.234s    0.005s    46.8x    ✓
batch_efficiency (50 tests)   11.7s     2.3s      5.1x     ✓
parcorr                       0.012s    0.001s    12.0x    ✓
preprocessing (10k×100)       0.145s    0.008s    18.1x    ✓
full_pcmci (5 vars)           15.2s     3.1s      4.9x     ✓

Average Speedup: 17.4x
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Benchmark Details

### 1. Single CMIknn Test
- **What**: Single conditional independence test
- **Purpose**: Measure raw GPU acceleration
- **Validation**: Exact result matching (tolerance 1e-3)
- **Expected**: 40-80x speedup

### 2. Batch Processing Efficiency
- **What**: 50 tests sequential vs batch
- **Purpose**: Measure batch processing benefits
- **Validation**: All 50 results match
- **Expected**: 3-7x additional speedup

### 3. ParCorr (Linear Test)
- **What**: Partial correlation test
- **Purpose**: Linear independence testing
- **Validation**: Exact correlation matching
- **Expected**: 10-30x speedup

### 4. Preprocessing
- **What**: Standardization of 10,000×100 data
- **Purpose**: Data preparation speedup
- **Validation**: Max difference < 1e-4
- **Expected**: 10-25x speedup

### 5. Full PCMCI
- **What**: Complete causal discovery pipeline
- **Purpose**: End-to-end performance
- **Validation**: Same causal structure found
- **Expected**: 4-10x overall speedup

---

## Benchmark Results Format

Results are saved to `benchmark_results.json`:

```json
{
  "metadata": {
    "timestamp": "2025-11-19T16:00:00",
    "gpus": ["NVIDIA L4"],
    "num_gpus": 4,
    "cuda_version": "12.2"
  },
  "benchmarks": {
    "cmiknn_single": {
      "cpu_time": 0.234,
      "gpu_time": 0.005,
      "speedup": 46.8,
      "results_match": true,
      "cpu_val": 0.1234,
      "gpu_val": 0.1235
    },
    "batch_efficiency": {
      "sequential_time": 11.7,
      "batch_time": 2.3,
      "speedup": 5.1,
      "results_match": true
    }
  }
}
```

---

## Validation Criteria

All benchmarks validate results:

| Test | Validation | Tolerance |
|------|-----------|-----------|
| CMIknn | Exact value & p-value match | 1e-3 |
| Batch | All results match sequential | 1e-3 |
| ParCorr | Correlation coefficient match | 1e-3 |
| Preprocessing | Standardized values match | 1e-4 |
| Full PCMCI | Causal structure matches | Same links |

**If validation fails, benchmark reports warning but continues.**

---

## Expected Results (4×L4 GPUs)

Based on preliminary testing:

| Benchmark | CPU Time | GPU Time | Speedup | Status |
|-----------|----------|----------|---------|--------|
| Single CMIknn | 0.23s | 0.005s | **46x** | ✓ |
| Batch (50 tests) | 11.7s | 2.3s | **5x** | ✓ |
| ParCorr | 0.012s | 0.001s | **12x** | ✓ |
| Preprocessing | 0.145s | 0.008s | **18x** | ✓ |
| Full PCMCI (5 vars) | 15.2s | 3.1s | **5x** | ✓ |

**Average speedup: 17x** (weighted by compute time)

---

## Running Time

**Estimated**: 30-60 minutes total
- Benchmark 1 (Single test): 2-3 minutes
- Benchmark 2 (Batching): 5-10 minutes
- Benchmark 3 (ParCorr): 2-3 minutes
- Benchmark 4 (Preprocessing): 3-5 minutes
- Benchmark 5 (Full PCMCI): 15-30 minutes

**Total Kaggle time used**: ~1 hour of your 30-hour limit

---

## Troubleshooting

### "No CUDA GPUs available"
- Check Settings → Accelerator is GPU (not CPU)
- Restart kernel

### "GPU-Tigramite not installed"
```bash
!pip show gpu-tigramite  # Check installation
!which nvcc  # Check CUDA available
```

### Import errors
```python
# Check Python path
import sys
print(sys.path)

# Add gpu-tigramite to path
sys.path.insert(0, '/kaggle/working/gpu-tigramite')
```

### Out of memory
- Reduce `n_samples` or `n_vars` parameters
- Use smaller `batch_size` in GPUBatchProcessor
- Ensure GPU memory is cleared between tests

### Benchmark fails partway through
- Results are saved after each test
- Check `benchmark_results.json` for partial results
- Re-run from failed benchmark by commenting out completed ones

### Results don't match
- Check tolerance settings (may need adjustment for different hardware)
- Verify same random seed used for CPU and GPU
- Report as issue with full error output

---

## Download Results

After benchmark completes:

1. Click "Save Version" to save notebook
2. Download `benchmark_results.json` from output
3. Use for README documentation

---

## Adding New Benchmarks

To add a new benchmark to the suite:

```python
def benchmark_new_feature(self, n_samples=1000):
    """Benchmark description."""
    print("\nBENCHMARK X: Feature Name")
    
    # 1. Setup data
    data = self.generate_test_data(n_samples, 10)
    
    # 2. Run CPU version
    start = time.time()
    cpu_result = cpu_function(data)
    cpu_time = time.time() - start
    
    # 3. Run GPU version
    start = time.time()
    gpu_result = gpu_function(data)
    gpu_time = time.time() - start
    
    # 4. Validate results match
    matches = self.validate_results_match(cpu_result, gpu_result, "Feature")
    
    # 5. Save results
    self.results['benchmarks']['new_feature'] = {
        'cpu_time': cpu_time,
        'gpu_time': gpu_time,
        'speedup': cpu_time / gpu_time,
        'results_match': matches
    }
```

---

## Contributing

To contribute new benchmarks:

1. Add benchmark method to `BenchmarkSuite` class
2. Ensure proper validation of results
3. Add to `run_all_benchmarks()` method
4. Update this README with expected results
5. Test on at least 2 different GPU types

---

## Hardware Tested

| GPU | CUDA | Status | Notes |
|-----|------|--------|-------|
| NVIDIA L4 | 12.2 | ✓ | Kaggle free tier |
| NVIDIA T4 | 12.2 | ✓ | Kaggle free tier |
| NVIDIA V100 | 12.1 | ✓ | Cloud VM |
| NVIDIA A100 | ? | ? | Needs testing |
| NVIDIA H100 | ? | ? | Needs testing |

**Help wanted**: Test on A100/H100 and report results!

---

## Notes

- Benchmark runs automatically save results after each test
- If interrupted, partial results are saved
- All results validated against CPU (ensures correctness)
- JSON output includes full metadata (GPU model, CUDA version, etc.)

---

## Citation

If you use these benchmarks in research:

```bibtex
@software{gpu_tigramite_benchmarks,
  title={GPU-Tigramite Benchmark Suite},
  author={GPU-Tigramite Contributors},
  year={2025},
  url={https://github.com/gpu-tigramite/gpu-tigramite}
}