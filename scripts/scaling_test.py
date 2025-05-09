#!/usr/bin/env python
"""
Test the scaling properties of H² attention vs. full attention
with different sequence lengths.
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import json
import signal
from time import time
from contextlib import contextmanager
from utils import generate_smooth_embeddings, compute_attention_matrix, ensure_dir
from h2_attention import H2Attention

def parse_args():
    parser = argparse.ArgumentParser(description='Test scaling of H² attention')
    parser.add_argument('--d', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--dv', type=int, default=64, help='Value dimension')
    parser.add_argument('--seq_lengths', type=int, nargs='+', 
                       default=[512, 1024, 2048, 4096, 8192],
                       help='Sequence lengths to test')
    parser.add_argument('--block_size', type=int, default=64, help='Block size')
    parser.add_argument('--rank', type=int, default=10, help='Rank for far-field approximation')
    parser.add_argument('--noise_level', type=float, default=0.01, help='Noise level')
    parser.add_argument('--eps', type=float, default=1e-6, help='Tolerance for SVD')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device for computations')
    parser.add_argument('--repeats', type=int, default=3, help='Number of repeats for timing')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout in seconds for each sequence length test')
    return parser.parse_args()

def benchmark_full_attention(q, k, v, repeats=3, device='cpu'):
    """Benchmark full attention."""
    times = []
    
    for _ in range(repeats):
        torch.cuda.synchronize() if device == 'cuda' else None
        start_time = time()
        
        # Full attention
        A = compute_attention_matrix(q, k)
        Av = torch.matmul(A, v)
        row_sums = torch.sum(A, dim=1, keepdim=True)
        output = Av / row_sums
        
        torch.cuda.synchronize() if device == 'cuda' else None
        times.append(time() - start_time)
    
    return min(times)  # Use minimum time for fairest comparison

def benchmark_h2_attention(q, k, v, rank, block_size, eps, repeats=3, device='cpu'):
    """Benchmark H² attention."""
    n = q.shape[0]
    
    # Compute full attention matrix (needed to build H²)
    A = compute_attention_matrix(q, k)
    
    # Create and build H² approximation
    h2_attention = H2Attention(
        n, block_size, distance=1, rank=rank, eps=eps, device=device
    )
    
    # Build time
    torch.cuda.synchronize() if device == 'cuda' else None
    build_start = time()
    h2_attention.build(A)
    torch.cuda.synchronize() if device == 'cuda' else None
    build_time = time() - build_start
    
    # Matvec time
    matvec_times = []
    for _ in range(repeats):
        torch.cuda.synchronize() if device == 'cuda' else None
        start_time = time()
        
        output = h2_attention.attention_output(v)
        
        torch.cuda.synchronize() if device == 'cuda' else None
        matvec_times.append(time() - start_time)
    
    # Error
    A = compute_attention_matrix(q, k)
    Av_full = torch.matmul(A, v)
    row_sums = torch.sum(A, dim=1, keepdim=True)
    output_full = Av_full / row_sums
    
    error = torch.norm(output_full - output) / torch.norm(output_full)
    
    return {
        'build_time': build_time,
        'matvec_time': min(matvec_times),
        'total_time': build_time + min(matvec_times),
        'error': error.item(),
        'stats': h2_attention.stats
    }

@contextmanager
def timeout(seconds):
    """Context manager for timeouts"""
    def handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set the timeout handler
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Cancel the timeout
        signal.alarm(0)

def main():
    args = parse_args()
    print(f"Testing scaling with parameters: {args}")
    
    # Setup output directories
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    viz_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visualizations')
    ensure_dir(results_dir)
    ensure_dir(viz_dir)
    
    results = []
    
    for n in args.seq_lengths:
        print(f"Testing sequence length: {n}")
        
        try:
            with timeout(args.timeout):
                # Generate embeddings
                q, k = generate_smooth_embeddings(n, args.d, noise_level=args.noise_level)
                v = torch.randn(n, args.dv)
                
                # Move to device
                q = q.to(args.device)
                k = k.to(args.device)
                v = v.to(args.device)
                
                # Benchmark full attention
                try:
                    time_full = benchmark_full_attention(q, k, v, args.repeats, args.device)
                    full_ok = True
                except RuntimeError as e:
                    print(f"  Full attention failed: {e}")
                    time_full = float('inf')
                    full_ok = False
                
                # Benchmark H² attention
                try:
                    h2_results = benchmark_h2_attention(
                        q, k, v, args.rank, args.block_size, args.eps, args.repeats, args.device
                    )
                    h2_ok = True
                except RuntimeError as e:
                    print(f"  H² attention failed: {e}")
                    h2_results = {
                        'build_time': float('inf'),
                        'matvec_time': float('inf'),
                        'total_time': float('inf'),
                        'error': float('inf'),
                        'stats': {}
                    }
                    h2_ok = False
        except TimeoutError:
            print(f"  Timeout reached for sequence length {n}")
            time_full = float('inf')
            h2_results = {
                'build_time': float('inf'),
                'matvec_time': float('inf'),
                'total_time': float('inf'),
                'error': float('inf'),
                'stats': {}
            }
            full_ok = False
            h2_ok = False
        
        result = {
            'n': n,
            'full_time': time_full,
            'h2_build_time': h2_results['build_time'],
            'h2_matvec_time': h2_results['matvec_time'],
            'h2_total_time': h2_results['total_time'],
            'speedup': time_full / h2_results['total_time'] if full_ok and h2_ok else float('nan'),
            'error': h2_results['error'],
            'h2_stats': h2_results['stats'] if h2_ok else {},
            'full_ok': full_ok,
            'h2_ok': h2_ok
        }
        
        results.append(result)
        
        print(f"  Full attention time: {time_full:.6f}s")
        print(f"  H² attention time: {h2_results['total_time']:.6f}s")
        if full_ok and h2_ok:
            print(f"  Speedup: {result['speedup']:.2f}x")
            print(f"  Error: {h2_results['error']:.6f}")
    
    # Save results
    with open(os.path.join(results_dir, 'scaling_results.json'), 'w') as f:
        json.dump({
            'params': vars(args),
            'results': results
        }, f, indent=2)
    
    # Plot scaling results
    n_values = [r['n'] for r in results]
    full_times = [r['full_time'] for r in results if r['full_ok']]
    h2_times = [r['h2_total_time'] for r in results if r['h2_ok']]
    n_full = [n for n, ok in zip(n_values, [r['full_ok'] for r in results]) if ok]
    n_h2 = [n for n, ok in zip(n_values, [r['h2_ok'] for r in results]) if ok]
    
    plt.figure(figsize=(12, 10))
    
    # Time vs. sequence length (log-log)
    plt.subplot(2, 2, 1)
    plt.loglog(n_full, full_times, 'o-', label='Full Attention')
    plt.loglog(n_h2, h2_times, 'o-', label='H² Attention')
    
    # Add theoretical scaling
    x = np.array(n_values)
    plt.loglog(x, 1e-8 * x**2, '--', label='O(n²)')
    plt.loglog(x, 1e-6 * x * np.log(x), '--', label='O(n log n)')
    
    plt.title('Runtime vs. Sequence Length')
    plt.xlabel('Sequence Length (n)')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.grid(True)
    
    # Error vs. sequence length
    plt.subplot(2, 2, 2)
    errors = [r['error'] for r in results if r['h2_ok']]
    plt.semilogx(n_h2, errors, 'o-')
    plt.title('Approximation Error vs. Sequence Length')
    plt.xlabel('Sequence Length (n)')
    plt.ylabel('Relative Error')
    plt.grid(True)
    
    # Speedup vs. sequence length
    plt.subplot(2, 2, 3)
    speedups = [r['speedup'] for r in results if r['full_ok'] and r['h2_ok']]
    n_both = [n for n, r in zip(n_values, results) if r['full_ok'] and r['h2_ok']]
    plt.semilogx(n_both, speedups, 'o-')
    plt.title('Speedup vs. Sequence Length')
    plt.xlabel('Sequence Length (n)')
    plt.ylabel('Speedup Factor')
    plt.grid(True)
    
    # Compression ratio vs. sequence length
    plt.subplot(2, 2, 4)
    compression = [r['h2_stats'].get('compression_ratio', float('nan')) for r in results]
    plt.semilogx(n_values, compression, 'o-')
    plt.title('Compression Ratio vs. Sequence Length')
    plt.xlabel('Sequence Length (n)')
    plt.ylabel('Compression Ratio')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'scaling_results.png'))
    
    print("Analysis complete! Results saved to:", results_dir)
    print("Visualizations saved to:", viz_dir)

if __name__ == '__main__':
    main()
