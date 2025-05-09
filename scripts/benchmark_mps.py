#!/usr/bin/env python
"""
Benchmark script to compare the performance of CPU and MPS implementations
of H² matrix approximation for attention.
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import json
from time import time
from utils import (
    generate_smooth_embeddings, ensure_dir
)
from h2_attention_direct import H2AttentionDirect
from h2_attention_mps import H2AttentionMPS

def run_benchmark(n, d, dv, block_size, rank, rsvd_params, device_types=['cpu', 'mps']):
    """
    Run benchmark comparing CPU and MPS implementations.
    
    Args:
        n: Sequence length
        d: Embedding dimension
        dv: Value dimension
        block_size: Block size
        rank: Rank for far-field approximation
        rsvd_params: Dictionary with randomized SVD parameters
        device_types: List of device types to benchmark
        
    Returns:
        results: Dictionary with benchmark results
    """
    results = {
        'n': n,
        'd': d,
        'dv': dv,
        'block_size': block_size,
        'rank': rank,
        'rsvd_params': rsvd_params,
        'cpu': {},
        'mps': {}
    }
    
    # Generate test data
    print(f"Generating test data for sequence length {n}...")
    q, k = generate_smooth_embeddings(n, d, noise_level=0.01)
    v = torch.randn(n, dv)
    
    # Run benchmarks for each device type
    for device_type in device_types:
        if device_type == 'mps' and not torch.backends.mps.is_available():
            print("MPS not available, skipping MPS benchmark")
            continue
            
        print(f"\nRunning benchmark on {device_type.upper()}...")
        
        # Create H² attention object
        if device_type == 'mps':
            h2_attention = H2AttentionMPS(
                n, d, block_size, distance=1, rank=rank, eps=rsvd_params['eps'],
                rsvd_oversampling=rsvd_params['oversampling'],
                rsvd_n_iter=rsvd_params['n_iter'],
                rsvd_min_block_size=rsvd_params['min_block_size']
            )
        else:
            h2_attention = H2AttentionDirect(
                n, d, block_size, distance=1, rank=rank, eps=rsvd_params['eps'], device='cpu',
                rsvd_oversampling=rsvd_params['oversampling'],
                rsvd_n_iter=rsvd_params['n_iter'],
                rsvd_min_block_size=rsvd_params['min_block_size']
            )
        
        # Measure build time
        start_time = time()
        h2_attention.build(q, k)
        build_time = time() - start_time
        print(f"  Build time: {build_time:.3f}s")
        
        # Measure matrix-vector multiplication time
        start_time = time()
        for _ in range(5):  # Average over 5 runs
            h2_attention.matvec(v)
        matvec_time = (time() - start_time) / 5
        print(f"  Matrix-vector multiplication time: {matvec_time:.6f}s")
        
        # Measure attention output time
        start_time = time()
        for _ in range(5):  # Average over 5 runs
            h2_attention.attention_output(v)
        attention_time = (time() - start_time) / 5
        print(f"  Attention output time: {attention_time:.6f}s")
        
        # Store results
        results[device_type] = {
            'build_time': build_time,
            'matvec_time': matvec_time,
            'attention_time': attention_time,
            'compression_ratio': h2_attention.stats['compression_ratio']
        }
    
    # Calculate speedups if both CPU and MPS results are available
    if 'cpu' in results and 'mps' in results and results['mps']:
        results['speedups'] = {
            'build': results['cpu']['build_time'] / results['mps']['build_time'],
            'matvec': results['cpu']['matvec_time'] / results['mps']['matvec_time'],
            'attention': results['cpu']['attention_time'] / results['mps']['attention_time']
        }
        print("\nSpeedups (MPS vs CPU):")
        print(f"  Build: {results['speedups']['build']:.2f}x")
        print(f"  Matrix-vector multiplication: {results['speedups']['matvec']:.2f}x")
        print(f"  Attention output: {results['speedups']['attention']:.2f}x")
    
    return results

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark CPU vs MPS for H² attention')
    parser.add_argument('--sizes', type=str, default='4096,8192,16384,32768',
                       help='Comma-separated list of sequence lengths to benchmark')
    parser.add_argument('--d', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--dv', type=int, default=64, help='Value dimension')
    parser.add_argument('--rank', type=int, default=10, help='Rank for far-field approximation')
    parser.add_argument('--rsvd_oversampling', type=int, default=10,
                       help='Oversampling parameter for randomized SVD')
    parser.add_argument('--rsvd_n_iter', type=int, default=2,
                       help='Number of power iterations for randomized SVD')
    parser.add_argument('--rsvd_min_block_size', type=int, default=5000,
                       help='Minimum block size to use randomized SVD')
    parser.add_argument('--eps', type=float, default=1e-6, help='Tolerance for SVD')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Benchmarking CPU vs MPS for H² attention with parameters: {args}")
    
    # Setup output directories
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    viz_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visualizations')
    ensure_dir(results_dir)
    ensure_dir(viz_dir)
    
    # Parse sequence lengths
    sizes = [int(size) for size in args.sizes.split(',')]
    
    # Randomized SVD parameters
    rsvd_params = {
        'oversampling': args.rsvd_oversampling,
        'n_iter': args.rsvd_n_iter,
        'min_block_size': args.rsvd_min_block_size,
        'eps': args.eps
    }
    
    # Run benchmarks for each sequence length
    all_results = []
    
    for n in sizes:
        # Determine appropriate block size based on sequence length
        if n <= 4096:
            block_size = 128
        elif n <= 16384:
            block_size = 256
        elif n <= 65536:
            block_size = 512
        else:
            block_size = 1024
        
        print(f"\n{'='*50}")
        print(f"Benchmarking sequence length: {n} with block size: {block_size}")
        print(f"{'='*50}")
        
        result = run_benchmark(
            n, args.d, args.dv, block_size, args.rank, rsvd_params,
            device_types=['cpu', 'mps']
        )
        all_results.append(result)
    
    # Save results
    with open(os.path.join(results_dir, 'h2_cpu_vs_mps_benchmark.json'), 'w') as f:
        # Convert to serializable format
        serializable_results = []
        for result in all_results:
            serializable_result = {k: v for k, v in result.items() if k not in ['cpu', 'mps', 'speedups']}
            if 'cpu' in result:
                serializable_result['cpu'] = {k: float(v) for k, v in result['cpu'].items()}
            if 'mps' in result:
                serializable_result['mps'] = {k: float(v) for k, v in result['mps'].items()}
            if 'speedups' in result:
                serializable_result['speedups'] = {k: float(v) for k, v in result['speedups'].items()}
            serializable_results.append(serializable_result)
            
        json.dump({
            'params': vars(args),
            'results': serializable_results
        }, f, indent=2)
    
    # Plot results
    if all_results:
        plot_benchmark_results(all_results, viz_dir)
    
    print("\nBenchmark complete! Results saved to:", results_dir)
    print("Visualizations saved to:", viz_dir)

def plot_benchmark_results(results, viz_dir):
    """Plot benchmark results comparing CPU and MPS performance."""
    sizes = [result['n'] for result in results]
    
    # Extract data for plotting
    build_times_cpu = [result['cpu']['build_time'] if 'cpu' in result else 0 for result in results]
    build_times_mps = [result['mps']['build_time'] if 'mps' in result else 0 for result in results]
    
    matvec_times_cpu = [result['cpu']['matvec_time'] if 'cpu' in result else 0 for result in results]
    matvec_times_mps = [result['mps']['matvec_time'] if 'mps' in result else 0 for result in results]
    
    attention_times_cpu = [result['cpu']['attention_time'] if 'cpu' in result else 0 for result in results]
    attention_times_mps = [result['mps']['attention_time'] if 'mps' in result else 0 for result in results]
    
    speedups_build = [result['speedups']['build'] if 'speedups' in result else 0 for result in results]
    speedups_matvec = [result['speedups']['matvec'] if 'speedups' in result else 0 for result in results]
    speedups_attention = [result['speedups']['attention'] if 'speedups' in result else 0 for result in results]
    
    # Create x-axis labels
    labels = [f"{size//1000}K" if size >= 1000 else str(size) for size in sizes]
    x = np.arange(len(labels))
    width = 0.35
    
    # Plot timing comparison
    plt.figure(figsize=(15, 15))
    
    # Build time comparison
    plt.subplot(3, 2, 1)
    plt.bar(x - width/2, build_times_cpu, width, label='CPU')
    plt.bar(x + width/2, build_times_mps, width, label='MPS')
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (seconds)')
    plt.title('Build Time')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Matrix-vector multiplication time comparison
    plt.subplot(3, 2, 3)
    plt.bar(x - width/2, matvec_times_cpu, width, label='CPU')
    plt.bar(x + width/2, matvec_times_mps, width, label='MPS')
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (seconds)')
    plt.title('Matrix-Vector Multiplication Time')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Attention output time comparison
    plt.subplot(3, 2, 5)
    plt.bar(x - width/2, attention_times_cpu, width, label='CPU')
    plt.bar(x + width/2, attention_times_mps, width, label='MPS')
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (seconds)')
    plt.title('Attention Output Time')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Speedup plots
    plt.subplot(3, 2, 2)
    plt.bar(x, speedups_build, width, color='green')
    plt.xlabel('Sequence Length')
    plt.ylabel('Speedup Factor (CPU/MPS)')
    plt.title('Build Time Speedup')
    plt.xticks(x, labels)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 4)
    plt.bar(x, speedups_matvec, width, color='green')
    plt.xlabel('Sequence Length')
    plt.ylabel('Speedup Factor (CPU/MPS)')
    plt.title('Matrix-Vector Multiplication Speedup')
    plt.xticks(x, labels)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 6)
    plt.bar(x, speedups_attention, width, color='green')
    plt.xlabel('Sequence Length')
    plt.ylabel('Speedup Factor (CPU/MPS)')
    plt.title('Attention Output Speedup')
    plt.xticks(x, labels)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'h2_cpu_vs_mps_benchmark.png'))

if __name__ == '__main__':
    main()
