# H²-Compressible Self-Attention

This repository implements H²-compressible self-attention using hierarchical low-rank structure with recursive partitioning and SVD for compression. The implementation validates the conjecture that under mild conditions (smooth token embeddings), the attention matrix can be approximated using a hierarchical low-rank structure, enabling subquadratic (near-linear) time complexity.

## Conjecture (H²-Compressibility of Self-Attention)

The key claim is that for analytically varying token embeddings, the far-field blocks of the attention matrix have numerical rank that scales as O(log(1/ε)), where ε is the desired error tolerance. This enables O(n log n) time complexity for self-attention computation, which is a significant improvement over the standard O(n²) complexity.

## Features

- Hierarchical matrix approximation for attention
- Randomized SVD for faster block compression
- Memory-efficient implementation for very long sequences (up to 131K tokens tested)
- Hybrid approach for building on CUDA and inference on any device (including Apple Silicon)
- Comprehensive benchmarking tools

## Repository Structure

```
h2_attention/
├── scripts/                   # Python scripts
│   ├── utils.py               # Common utilities
│   ├── analyze_rank.py        # Analyze numerical rank of far-field blocks
│   ├── h2_attention.py        # Basic H² matrix approx for attention
│   ├── h2_attention_direct.py # Memory-efficient implementation
│   ├── h2_attention_mps.py    # MPS-optimized implementation for Apple Silicon
│   ├── h2_attention_hybrid.py # Hybrid approach (build on CUDA, inference anywhere)
│   ├── benchmark_rsvd.py      # Benchmark different SVD implementations
│   ├── benchmark_mps.py       # Benchmark CPU vs MPS on Apple Silicon
│   ├── scaling_test.py        # Test scaling properties with sequence length
│   └── analyze_pretrained.py  # Analyze real pretrained models
├── results/                   # JSON results from experiments
├── models/                    # Saved H² models
└── visualizations/            # Generated plots and figures
```

## Usage

### Basic H² Attention

```bash
python scripts/h2_attention.py --n 16384 --block_size 256 --ranks 5 10 15 20 --rsvd_oversampling 10 --rsvd_n_iter 2 --rsvd_min_block_size 5000
```

### Memory-Efficient Implementation for Very Long Sequences

```bash
python scripts/h2_attention_direct.py --n 65536 --block_size 1024 --ranks 5 10 15 --rsvd_oversampling 10 --rsvd_n_iter 2 --rsvd_min_block_size 5000
```

### Hybrid Approach (Build on CUDA, Inference on Any Device)

Build on a CUDA-enabled machine:
```bash
python scripts/h2_attention_hybrid.py --mode build --n 16384 --block_size 256 --rank 10 --device cuda --model_path h2_model_16k.pkl
```

Run inference on any device:
```bash
python scripts/h2_attention_hybrid.py --mode inference --n 16384 --device cpu --model_path h2_model_16k.pkl
```

### Benchmarking

```bash
# Benchmark different SVD implementations
python scripts/benchmark_rsvd.py --sizes 64x64,128x128,256x256,512x512 --rank 10 --trials 5

# Benchmark CPU vs MPS on Apple Silicon
python scripts/benchmark_mps.py --sizes 4096,8192,16384 --rank 10
```

## Results

The H² attention implementation achieves:
- Compression ratios of up to 24.86x for 131K sequences
- Speedups of up to 46.96x compared to full attention
- Low approximation errors (around 0.00007-0.00008)

## Apple Silicon Optimization

For Apple Silicon (M-series) processors:
1. Build H² matrices on CUDA-enabled machines for maximum efficiency
2. Save the compressed models and transfer them to MacBooks for inference
3. Use CPU inference for smaller sequences, test MPS for larger sequences

## Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy
- Matplotlib
- Transformers (for analyze_pretrained.py)

## Quick Start

1. Clone this repository
2. Install dependencies: `pip install torch numpy matplotlib transformers`
3. Run the rank analysis script: `python scripts/analyze_rank.py`
4. Check the results in the `results/` and `visualizations/` directories

## References

The H²-compressibility implementation builds on hierarchical matrix theory from numerical analysis, particularly:

- H-matrices and H²-matrices for efficient numerical computation
- Fast multipole methods for N-body problems
- Randomized SVD algorithms (Halko et al. 2009)
- Analytical properties of kernel functions in reproducing kernel Hilbert spaces
