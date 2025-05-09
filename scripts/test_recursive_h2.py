#!/usr/bin/env python
"""
Test script for the RecursiveH2Matrix implementation.
Compares the results and performance against full attention.
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import json
from time import time
from utils import generate_smooth_embeddings, compute_attention_matrix, ensure_dir
from recursive_h2_matrix import RecursiveH2Matrix

def compare_attention(q, k, v, h2_matrix, device='cpu'):
    """
    Compare H² attention to full attention.
    
    Args:
        q, k: Query and key embeddings
        v: Value embeddings
        h2_matrix: RecursiveH2Matrix object
        device: Computation device
    
    Returns:
        error: Relative error between full and H² attention
        time_full: Time for full attention
        time_h2: Time for H² attention
    """
    n = q.shape[0]
    
    # Full attention
    start_time = time()
    A = compute_attention_matrix(q, k)
    Av_full = torch.matmul(A, v)
    row_sums = torch.sum(A, dim=1, keepdim=True)
    output_full = Av_full / row_sums
    time_full = time() - start_time
    
    # H² attention
    start_time = time()
    output_h2 = h2_matrix.attention_output(v)
    time_h2 = time() - start_time
    
    # Compute error
    error = torch.norm(output_full - output_h2) / torch.norm(output_full)
    
    return error.item(), time_full, time_h2

def parse_args():
    parser = argparse.ArgumentParser(description='Recursive H² matrix approximation for attention')
    parser.add_argument('--n', type=int, default=1024, help='Sequence length')
    parser.add_argument('--d', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--dv', type=int, default=64, help='Value dimension')
    parser.add_argument('--nmin', type=int, default=32, help='Minimum leaf size')
    parser.add_argument('--ranks', type=int, nargs='+', default=[5, 10, 15, 20], 
                       help='Ranks for far-field approximation')
    parser.add_argument('--noise_level', type=float, default=0.01, help='Noise level')
    parser.add_argument('--eta', type=float, default=0.8, 
                       help='Admissibility parameter (higher = more compression)')
    parser.add_argument('--eps', type=float, default=1e-6, help='Tolerance for SVD')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device for computations')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Testing Recursive H² attention with parameters: {args}")
    
    # Setup output directories
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    viz_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visualizations')
    ensure_dir(results_dir)
    ensure_dir(viz_dir)
    
    # Generate smooth embeddings
    q, k = generate_smooth_embeddings(args.n, args.d, noise_level=args.noise_level)
    v = torch.randn(args.n, args.dv)
    
    # Move to device
    q = q.to(args.device)
    k = k.to(args.device)
    v = v.to(args.device)
    
    # Compute full attention matrix for validation
    A = compute_attention_matrix(q, k)
    
    # Test with different ranks
    results = []
    
    for rank in args.ranks:
        print(f"Testing rank: {rank}")
        
        # Create Recursive H² approximation
        h2_matrix = RecursiveH2Matrix(
            args.n, args.nmin, rank, args.eta, args.eps, args.device
        )
        
        # Build H² matrix
        build_time = h2_matrix.build(q, k)
        
        # Compare results
        error, time_full, time_h2 = compare_attention(q, k, v, h2_matrix, args.device)
        
        result = {
            'rank': rank,
            'error': error,
            'time_full': time_full,
            'time_h2': time_h2,
            'speedup': time_full / time_h2,
            'compression_ratio': h2_matrix.stats['compression_ratio'],
            'build_time': build_time,
            'n_dense_blocks': h2_matrix.stats['n_dense_blocks'],
            'n_low_rank_blocks': h2_matrix.stats['n_low_rank_blocks'],
            'average_rank': h2_matrix.stats['average_rank']
        }
        
        results.append(result)
        print(f"  Error: {error:.6f}")
        print(f"  Time (full): {time_full:.6f}s")
        print(f"  Time (H²): {time_h2:.6f}s")
        print(f"  Speedup: {result['speedup']:.2f}x")
    
    # Save results
    with open(os.path.join(results_dir, 'recursive_h2_results.json'), 'w') as f:
        json.dump({
            'params': vars(args),
            'results': results
        }, f, indent=2)
    
    # Plot results
    ranks = [r['rank'] for r in results]
    errors = [r['error'] for r in results]
    speedups = [r['speedup'] for r in results]
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(ranks, errors, marker='o', linewidth=2)
    plt.title('Approximation Error vs. Rank')
    plt.xlabel('Rank')
    plt.ylabel('Relative Error')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(ranks, speedups, marker='o', linewidth=2)
    plt.title('Speedup vs. Rank')
    plt.xlabel('Rank')
    plt.ylabel('Speedup Factor')
    plt.grid(True)
    
    # Plot compression ratio
    compression_ratios = [r['compression_ratio'] for r in results]
    plt.subplot(2, 2, 3)
    plt.plot(ranks, compression_ratios, marker='o', linewidth=2)
    plt.title('Compression Ratio vs. Rank')
    plt.xlabel('Rank')
    plt.ylabel('Compression Ratio')
    plt.grid(True)
    
    # Plot block distribution
    dense_blocks = [r['n_dense_blocks'] for r in results]
    low_rank_blocks = [r['n_low_rank_blocks'] for r in results]
    plt.subplot(2, 2, 4)
    width = 0.35
    x = np.arange(len(ranks))
    plt.bar(x - width/2, dense_blocks, width, label='Dense Blocks')
    plt.bar(x + width/2, low_rank_blocks, width, label='Low-Rank Blocks')
    plt.title('Block Distribution')
    plt.xlabel('Rank')
    plt.ylabel('Number of Blocks')
    plt.xticks(x, ranks)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'recursive_h2_performance.png'))
    
    print("Analysis complete! Results saved to:", results_dir)
    print("Visualizations saved to:", viz_dir)

if __name__ == '__main__':
    main()
