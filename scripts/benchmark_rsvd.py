#!/usr/bin/env python
"""
Benchmark script to compare the performance of different SVD implementations
for H² matrix approximation in attention.
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import json
from time import time
from utils import (
    generate_smooth_embeddings, compute_attention_matrix,
    get_block_indices, is_far_field, randomized_svd, ensure_dir
)

def standard_svd(A, rank, eps=1e-6):
    """Standard SVD implementation using torch.svd."""
    u, s, v = torch.svd(A)
    
    # Determine numerical rank (up to rank parameter)
    tol = eps * s[0].item() if s[0].item() > 0 else eps
    k = min(rank, torch.sum(s > tol).item())
    k = int(k)
    
    # Truncate to the numerical rank
    u = u[:, :k]
    s = s[:k]
    v = v[:, :k]
    
    return u, s, v

def simple_randomized_svd(A, rank, oversampling=10):
    """Simple randomized SVD implementation (without power iteration)."""
    m, n = A.shape
    l = min(m, n, rank + oversampling)
    
    # Random projection
    Omega = torch.randn(n, l, device=A.device)
    Y = A @ Omega
    
    # Orthogonalize
    Q, _ = torch.linalg.qr(Y)
    
    # Project
    B = Q.T @ A
    
    # SVD on smaller matrix
    u_hat, s, v = torch.svd(B)
    u = Q @ u_hat
    
    # Truncate to the desired rank
    u = u[:, :rank]
    s = s[:rank]
    v = v[:, :rank]
    
    return u, s, v

def benchmark_svd_methods(matrix_sizes, rank=10, n_trials=5, device='cpu'):
    """
    Benchmark different SVD implementations on matrices of various sizes.
    
    Args:
        matrix_sizes: List of (m, n) tuples for matrix dimensions
        rank: Target rank for approximation
        n_trials: Number of trials for each size
        device: Computation device
        
    Returns:
        results: Dictionary with benchmark results
    """
    results = {
        'matrix_sizes': matrix_sizes,
        'standard_svd': [],
        'simple_rsvd': [],
        'advanced_rsvd': [],
        'errors_simple': [],
        'errors_advanced': []
    }
    
    for m, n in matrix_sizes:
        print(f"Benchmarking matrix size: {m}x{n}")
        
        # Generate random matrices for testing
        times_standard = []
        times_simple = []
        times_advanced = []
        errors_simple = []
        errors_advanced = []
        
        for trial in range(n_trials):
            # Create a random matrix with decaying singular values
            # to simulate attention-like matrices
            U = torch.randn(m, min(m, n), device=device)
            U, _ = torch.linalg.qr(U)
            V = torch.randn(n, min(m, n), device=device)
            V, _ = torch.linalg.qr(V)
            
            # Create singular values with exponential decay
            s = torch.exp(-torch.arange(min(m, n), device=device) / 10)
            
            # Form the test matrix
            A = U @ torch.diag(s) @ V.T
            
            # Standard SVD
            start_time = time()
            u_std, s_std, v_std = standard_svd(A, rank)
            times_standard.append(time() - start_time)
            
            # Simple randomized SVD
            start_time = time()
            u_simple, s_simple, v_simple = simple_randomized_svd(A, rank)
            times_simple.append(time() - start_time)
            
            # Advanced randomized SVD (with power iteration)
            start_time = time()
            u_adv, s_adv, v_adv = randomized_svd(A, rank, n_iter=2)
            times_advanced.append(time() - start_time)
            
            # Compute approximation errors
            A_simple = u_simple @ torch.diag(s_simple) @ v_simple.T
            A_adv = u_adv @ torch.diag(s_adv) @ v_adv.T
            
            error_simple = torch.norm(A - A_simple) / torch.norm(A)
            error_adv = torch.norm(A - A_adv) / torch.norm(A)
            
            errors_simple.append(error_simple.item())
            errors_advanced.append(error_adv.item())
        
        # Average results
        results['standard_svd'].append(np.mean(times_standard))
        results['simple_rsvd'].append(np.mean(times_simple))
        results['advanced_rsvd'].append(np.mean(times_advanced))
        results['errors_simple'].append(np.mean(errors_simple))
        results['errors_advanced'].append(np.mean(errors_advanced))
        
        # Print results for this size
        print(f"  Standard SVD: {results['standard_svd'][-1]:.6f}s")
        print(f"  Simple RSVD: {results['simple_rsvd'][-1]:.6f}s (speedup: {results['standard_svd'][-1]/results['simple_rsvd'][-1]:.2f}x)")
        print(f"  Advanced RSVD: {results['advanced_rsvd'][-1]:.6f}s (speedup: {results['standard_svd'][-1]/results['advanced_rsvd'][-1]:.2f}x)")
        print(f"  Error (Simple): {results['errors_simple'][-1]:.6f}")
        print(f"  Error (Advanced): {results['errors_advanced'][-1]:.6f}")
    
    return results

def plot_results(results, output_dir):
    """Plot benchmark results."""
    # Convert matrix sizes to labels
    labels = [f"{m}x{n}" for m, n in results['matrix_sizes']]
    x = np.arange(len(labels))
    width = 0.25
    
    # Plot timing comparison
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    plt.bar(x - width, results['standard_svd'], width, label='Standard SVD')
    plt.bar(x, results['simple_rsvd'], width, label='Simple RSVD')
    plt.bar(x + width, results['advanced_rsvd'], width, label='Advanced RSVD')
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.title('SVD Computation Time')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot error comparison
    plt.subplot(2, 1, 2)
    plt.bar(x - width/2, results['errors_simple'], width, label='Simple RSVD')
    plt.bar(x + width/2, results['errors_advanced'], width, label='Advanced RSVD')
    plt.xlabel('Matrix Size')
    plt.ylabel('Relative Error')
    plt.title('Approximation Error')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'svd_benchmark.png'))
    
    # Plot speedup
    plt.figure(figsize=(10, 6))
    speedup_simple = [std/simple for std, simple in zip(results['standard_svd'], results['simple_rsvd'])]
    speedup_advanced = [std/adv for std, adv in zip(results['standard_svd'], results['advanced_rsvd'])]
    
    plt.bar(x - width/2, speedup_simple, width, label='Simple RSVD')
    plt.bar(x + width/2, speedup_advanced, width, label='Advanced RSVD')
    plt.xlabel('Matrix Size')
    plt.ylabel('Speedup Factor')
    plt.title('SVD Speedup Relative to Standard SVD')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'svd_speedup.png'))

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark SVD implementations')
    parser.add_argument('--sizes', type=str, default='64x64,128x128,256x256,512x512,1024x1024',
                       help='Comma-separated list of matrix sizes in format mxn')
    parser.add_argument('--rank', type=int, default=10, help='Target rank for approximation')
    parser.add_argument('--trials', type=int, default=5, help='Number of trials for each size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device for computations')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Benchmarking SVD implementations with parameters: {args}")
    
    # Parse matrix sizes
    matrix_sizes = []
    for size_str in args.sizes.split(','):
        m, n = map(int, size_str.split('x'))
        matrix_sizes.append((m, n))
    
    # Setup output directories
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    viz_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visualizations')
    ensure_dir(results_dir)
    ensure_dir(viz_dir)
    
    # Run benchmarks
    results = benchmark_svd_methods(
        matrix_sizes, 
        rank=args.rank,
        n_trials=args.trials,
        device=args.device
    )
    
    # Save results
    with open(os.path.join(results_dir, 'svd_benchmark_results.json'), 'w') as f:
        # Convert numpy values to Python native types for JSON serialization
        serializable_results = {
            'matrix_sizes': [(int(m), int(n)) for m, n in results['matrix_sizes']],
            'standard_svd': [float(x) for x in results['standard_svd']],
            'simple_rsvd': [float(x) for x in results['simple_rsvd']],
            'advanced_rsvd': [float(x) for x in results['advanced_rsvd']],
            'errors_simple': [float(x) for x in results['errors_simple']],
            'errors_advanced': [float(x) for x in results['errors_advanced']]
        }
        json.dump({
            'params': vars(args),
            'results': serializable_results
        }, f, indent=2)
    
    # Plot results
    plot_results(results, viz_dir)
    
    print("Benchmark complete! Results saved to:", results_dir)
    print("Visualizations saved to:", viz_dir)

if __name__ == '__main__':
    main()
