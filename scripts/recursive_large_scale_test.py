#!/usr/bin/env python
"""
Test the recursive H² approximation for very large sequences.
This script uses a memory-efficient approach that avoids forming the full attention matrix.
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
from utils import generate_smooth_embeddings, ensure_dir
from recursive_h2_matrix import RecursiveH2Matrix

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

def measure_approximation_error(q, k, v, h2_matrix, device='cpu', max_samples=1000):
    """
    Measure the approximation error for a subset of randomly sampled rows.
    
    Args:
        q, k: Query and key embeddings
        v: Value embeddings
        h2_matrix: RecursiveH2Matrix object
        device: Computation device
        max_samples: Maximum number of rows to sample for error calculation
    
    Returns:
        error: Relative approximation error
    """
    n = q.shape[0]
    
    # Randomly sample rows for error calculation
    sample_size = min(max_samples, n)
    indices = torch.randperm(n)[:sample_size]
    
    # Compute exact attention for sampled rows
    exact_rows = []
    for idx in indices:
        # Compute one row of the attention matrix
        q_i = q[idx:idx+1]
        qk = torch.matmul(q_i, k.transpose(0, 1)) / np.sqrt(q.shape[1])
        a_row = torch.exp(qk)
        
        # Apply row normalization
        a_row = a_row / torch.sum(a_row)
        
        # Compute attention output for this row
        out_row = torch.matmul(a_row, v)
        exact_rows.append(out_row)
    
    exact_output = torch.cat(exact_rows, dim=0)
    
    # Compute H² approximation for sampled rows
    h2_output = h2_matrix.attention_output(v)
    approx_output = h2_output[indices]
    
    # Compute relative error
    error = torch.norm(exact_output - approx_output) / torch.norm(exact_output)
    
    return error.item()

def parse_args():
    parser = argparse.ArgumentParser(description='Test recursive H² attention for very large sequences')
    parser.add_argument('--d', type=int, default=16, help='Embedding dimension')
    parser.add_argument('--dv', type=int, default=16, help='Value dimension')
    parser.add_argument('--seq_lengths', type=int, nargs='+', 
                       default=[1024, 2048, 4096, 8192, 16384],
                       help='Sequence lengths to test')
    parser.add_argument('--nmin', type=int, default=64, help='Minimum leaf size')
    parser.add_argument('--rank', type=int, default=10, help='Rank for far-field approximation')
    parser.add_argument('--eta', type=float, default=0.8, help='Admissibility parameter')
    parser.add_argument('--noise_level', type=float, default=0.01, help='Noise level')
    parser.add_argument('--eps', type=float, default=1e-5, help='Tolerance for SVD')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device for computations')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout in seconds for each sequence length test')
    parser.add_argument('--error_samples', type=int, default=100, 
                       help='Number of random samples for error calculation')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Testing recursive H² attention for large sequences with parameters: {args}")
    
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
                start_time = time()
                
                # Generate embeddings
                print(f"  Generating embeddings...")
                q, k = generate_smooth_embeddings(n, args.d, noise_level=args.noise_level)
                v = torch.randn(n, args.dv)
                
                # Move to device
                q = q.to(args.device)
                k = k.to(args.device)
                v = v.to(args.device)
                
                # Build H² matrix directly
                print(f"  Building H² matrix...")
                build_start = time()
                h2_matrix = RecursiveH2Matrix(
                    n, args.nmin, args.rank, args.eta, args.eps, args.device
                )
                build_time = h2_matrix.build(q, k)
                print(f"  H² matrix built in {build_time:.2f} seconds")
                
                # Print statistics
                print(f"  H² statistics:")
                print(f"    Near-field blocks: {h2_matrix.stats['n_dense_blocks']}")
                print(f"    Far-field blocks: {h2_matrix.stats['n_low_rank_blocks']}")
                print(f"    Compression ratio: {h2_matrix.stats['compression_ratio']:.2f}x")
                
                # Measure time for H² matrix-vector product
                print(f"  Measuring H² matvec time...")
                matvec_start = time()
                _ = h2_matrix.attention_output(v)
                matvec_time = time() - matvec_start
                print(f"  H² matvec completed in {matvec_time:.2f} seconds")
                
                # Measure approximation error
                print(f"  Measuring approximation error...")
                error = measure_approximation_error(q, k, v, h2_matrix, args.device, args.error_samples)
                print(f"  Approximation error: {error:.6f}")
                
                total_time = time() - start_time
                print(f"  Total test time: {total_time:.2f} seconds")
                
                # Theoretical full attention time (O(n²))
                theoretical_n2_time = (n / 1000) ** 2 * 0.001  # Approximate scaling
                theoretical_nlogn_time = (n / 1000) * np.log(n / 1000) * 0.001  # Approximate scaling
                
                speedup_vs_theoretical = theoretical_n2_time / (build_time + matvec_time)
                
                result = {
                    'n': n,
                    'build_time': build_time,
                    'matvec_time': matvec_time,
                    'total_time': build_time + matvec_time,
                    'error': error,
                    'n_dense_blocks': h2_matrix.stats['n_dense_blocks'],
                    'n_low_rank_blocks': h2_matrix.stats['n_low_rank_blocks'],
                    'compression_ratio': h2_matrix.stats['compression_ratio'],
                    'theoretical_n2_time': theoretical_n2_time,
                    'theoretical_nlogn_time': theoretical_nlogn_time,
                    'speedup_vs_theoretical': speedup_vs_theoretical
                }
                
                results.append(result)
                
            # End of with timeout block
        except TimeoutError:
            print(f"  Timeout reached for sequence length {n}")
            results.append({
                'n': n,
                'build_time': float('inf'),
                'matvec_time': float('inf'),
                'total_time': float('inf'),
                'error': float('inf'),
                'n_dense_blocks': 0,
                'n_low_rank_blocks': 0,
                'compression_ratio': 0,
                'theoretical_n2_time': (n / 1000) ** 2 * 0.001,
                'theoretical_nlogn_time': (n / 1000) * np.log(n / 1000) * 0.001,
                'speedup_vs_theoretical': 0
            })
        except Exception as e:
            import traceback
            print(f"  Error processing sequence length {n}: {e}")
            print(f"  Traceback: {traceback.format_exc()}")
            results.append({
                'n': n,
                'build_time': float('inf'),
                'matvec_time': float('inf'),
                'total_time': float('inf'),
                'error': float('inf'),
                'n_dense_blocks': 0,
                'n_low_rank_blocks': 0,
                'compression_ratio': 0,
                'theoretical_n2_time': (n / 1000) ** 2 * 0.001,
                'theoretical_nlogn_time': (n / 1000) * np.log(n / 1000) * 0.001,
                'speedup_vs_theoretical': 0
            })
    
    # Save results
    with open(os.path.join(results_dir, 'recursive_large_scale_results.json'), 'w') as f:
        json.dump({
            'params': vars(args),
            'results': results
        }, f, indent=2)
    
    # Plot results if we have at least one successful run
    successful_results = [r for r in results if r['total_time'] != float('inf')]
    
    if successful_results:
        # Extract data for plotting
        n_values = [r['n'] for r in successful_results]
        build_times = [r['build_time'] for r in successful_results]
        matvec_times = [r['matvec_time'] for r in successful_results]
        total_times = [r['total_time'] for r in successful_results]
        errors = [r['error'] for r in successful_results]
        compression_ratios = [r['compression_ratio'] for r in successful_results]
        theoretical_n2_times = [r['theoretical_n2_time'] for r in successful_results]
        theoretical_nlogn_times = [r['theoretical_nlogn_time'] for r in successful_results]
        
        # Plot time vs. sequence length
        plt.figure(figsize=(12, 10))
        
        # Time vs. sequence length (log-log)
        plt.subplot(2, 2, 1)
        plt.loglog(n_values, total_times, 'o-', label='H² Total Time')
        plt.loglog(n_values, build_times, 'o--', label='H² Build Time')
        plt.loglog(n_values, matvec_times, 'o-.', label='H² Matvec Time')
        plt.loglog(n_values, theoretical_n2_times, '--', label='Theoretical O(n²)')
        plt.loglog(n_values, theoretical_nlogn_times, '--', label='Theoretical O(n log n)')
        plt.title('Runtime vs. Sequence Length')
        plt.xlabel('Sequence Length (n)')
        plt.ylabel('Time (s)')
        plt.legend()
        plt.grid(True)
        
        # Error vs. sequence length
        plt.subplot(2, 2, 2)
        plt.semilogx(n_values, errors, 'o-')
        plt.title('Approximation Error vs. Sequence Length')
        plt.xlabel('Sequence Length (n)')
        plt.ylabel('Relative Error')
        plt.grid(True)
        
        # Compression ratio vs. sequence length
        plt.subplot(2, 2, 3)
        plt.semilogx(n_values, compression_ratios, 'o-')
        plt.title('Compression Ratio vs. Sequence Length')
        plt.xlabel('Sequence Length (n)')
        plt.ylabel('Compression Ratio')
        plt.grid(True)
        
        # Speedup vs. sequence length
        plt.subplot(2, 2, 4)
        speedups = [r['speedup_vs_theoretical'] for r in successful_results]
        plt.semilogx(n_values, speedups, 'o-')
        plt.title('Speedup vs. Theoretical O(n²) Time')
        plt.xlabel('Sequence Length (n)')
        plt.ylabel('Speedup Factor')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'recursive_large_scale_results.png'))
        
        print("Analysis complete! Results saved to:", results_dir)
        print("Visualizations saved to:", viz_dir)
    else:
        print("No successful runs to plot!")

if __name__ == '__main__':
    main()
