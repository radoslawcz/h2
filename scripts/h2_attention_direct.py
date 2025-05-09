#!/usr/bin/env python
"""
Memory-efficient implementation of H² matrix approximation for attention.
Builds the H² matrix directly from query and key embeddings without forming the full attention matrix.
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

def compute_attention_block(q_block, k_block):
    """
    Compute a block of the attention matrix without forming the full matrix.

    Args:
        q_block: Query embeddings for a block
        k_block: Key embeddings for a block

    Returns:
        A_block: Block of the attention matrix
    """
    d = q_block.shape[1]
    qk = torch.matmul(q_block, k_block.transpose(0, 1)) / np.sqrt(d)
    A_block = torch.exp(qk)
    return A_block

class H2AttentionDirect:
    """
    Memory-efficient H² matrix approximation for attention.
    Builds the H² matrix directly from query and key embeddings without forming the full attention matrix.
    """

    def __init__(self, n, d, block_size=64, distance=1, rank=10, eps=1e-6, device='cpu',
                 rsvd_oversampling=10, rsvd_n_iter=2, rsvd_min_block_size=10000):
        """
        Initialize the H² attention approximation.

        Args:
            n: Sequence length
            d: Embedding dimension
            block_size: Size of leaf-level blocks
            distance: Minimum number of blocks separating far-field blocks
            rank: Maximum rank for far-field approximations
            eps: Tolerance for SVD approximation
            device: Computation device
            rsvd_oversampling: Oversampling parameter for randomized SVD
            rsvd_n_iter: Number of power iterations for randomized SVD
            rsvd_min_block_size: Minimum block size to use randomized SVD
        """
        self.n = n
        self.d = d
        self.block_size = block_size
        self.distance = distance
        self.rank = rank
        self.eps = eps
        self.device = device

        # Randomized SVD parameters
        self.rsvd_oversampling = rsvd_oversampling
        self.rsvd_n_iter = rsvd_n_iter
        self.rsvd_min_block_size = rsvd_min_block_size

        # Create block partition
        self.block_indices = get_block_indices(n, block_size)
        self.n_blocks = len(self.block_indices)

        # Initialize data structures
        self.near_field = {}  # Maps (i,j) -> dense block
        self.far_field = {}   # Maps (i,j) -> (U, V) low-rank factors

        # Initialize stats
        self.stats = {}

    def build(self, q, k):
        """
        Build the H² approximation directly from query and key embeddings.

        Args:
            q: Query embeddings (n, d)
            k: Key embeddings (n, d)
        """
        start_time = time()

        for i in range(self.n_blocks):
            for j in range(self.n_blocks):
                block_i = self.block_indices[i]
                block_j = self.block_indices[j]

                if is_far_field(block_i, block_j, self.distance):
                    # Far-field block: create low-rank approximation
                    self._build_far_field_direct(q, k, i, j)
                else:
                    # Near-field block: store as dense
                    self._build_near_field_direct(q, k, i, j)

        build_time = time() - start_time
        print(f"H² matrix built in {build_time:.3f} seconds")

        # Calculate compression statistics
        self._calculate_stats()

        return build_time

    def _build_near_field_direct(self, q, k, i, j):
        """Store near-field block as dense matrix, computed directly from q and k."""
        block_i = self.block_indices[i]
        block_j = self.block_indices[j]

        q_block = q[block_i[0]:block_i[1]]
        k_block = k[block_j[0]:block_j[1]]

        A_block = compute_attention_block(q_block, k_block)
        self.near_field[(i, j)] = A_block

    def _build_far_field_direct(self, q, k, i, j):
        """Create low-rank approximation for far-field block directly from q and k."""
        block_i = self.block_indices[i]
        block_j = self.block_indices[j]

        q_block = q[block_i[0]:block_i[1]]
        k_block = k[block_j[0]:block_j[1]]

        # Compute the block of the attention matrix
        A_block = compute_attention_block(q_block, k_block)

        # Determine if we should use randomized SVD based on block size
        block_size = A_block.shape[0] * A_block.shape[1]

        if block_size > self.rsvd_min_block_size:
            # Use our improved randomized SVD implementation
            u, s, v = randomized_svd(
                A_block,
                rank=self.rank,
                n_oversamples=self.rsvd_oversampling,
                n_iter=self.rsvd_n_iter,
                eps=self.eps,
                device=self.device
            )
        else:
            # Regular SVD for smaller blocks
            u, s, v = torch.svd(A_block)

            # Determine numerical rank (up to rank parameter)
            tol = self.eps * s[0].item() if s[0].item() > 0 else self.eps
            k = min(self.rank, torch.sum(s > tol).item())
            k = int(k)

            # Truncate to the numerical rank
            u = u[:, :k]
            s = s[:k]
            v = v[:, :k]

        # Create low-rank factors
        U = u @ torch.diag(s)
        V = v

        self.far_field[(i, j)] = (U, V)

    def _calculate_stats(self):
        """Calculate compression statistics."""
        n_near = len(self.near_field)
        n_far = len(self.far_field)

        # Memory usage
        mem_full = self.n ** 2

        mem_near = 0
        for _, block in self.near_field.items():
            mem_near += block.numel()

        mem_far = 0
        for _, (U, V) in self.far_field.items():
            mem_far += U.numel() + V.numel()

        mem_h2 = mem_near + mem_far

        self.stats = {
            'n_near': n_near,
            'n_far': n_far,
            'mem_full': mem_full,
            'mem_h2': mem_h2,
            'compression_ratio': mem_full / mem_h2 if mem_h2 > 0 else float('inf')
        }

        print(f"H² statistics:")
        print(f"  Near-field blocks: {n_near}")
        print(f"  Far-field blocks: {n_far}")
        print(f"  Memory usage: {mem_h2} elements ({self.stats['compression_ratio']:.2f}x compression)")

    def matvec(self, x):
        """
        Apply the H² matrix to a vector.

        Args:
            x: Input vector (n,) or matrix (n, d)

        Returns:
            y: Output vector (n,) or matrix (n, d)
        """
        if x.dim() == 1:
            y = torch.zeros(self.n, device=self.device)
        else:
            y = torch.zeros(self.n, x.shape[1], device=self.device)

        # Apply near-field blocks
        for (i, j), block in self.near_field.items():
            block_i = self.block_indices[i]
            block_j = self.block_indices[j]

            if x.dim() == 1:
                y[block_i[0]:block_i[1]] += torch.matmul(block, x[block_j[0]:block_j[1]])
            else:
                y[block_i[0]:block_i[1]] += torch.matmul(block, x[block_j[0]:block_j[1], :])

        # Apply far-field blocks
        for (i, j), (U, V) in self.far_field.items():
            block_i = self.block_indices[i]
            block_j = self.block_indices[j]

            if x.dim() == 1:
                tmp = torch.matmul(V.t(), x[block_j[0]:block_j[1]])
                y[block_i[0]:block_i[1]] += torch.matmul(U, tmp)
            else:
                tmp = torch.matmul(V.t(), x[block_j[0]:block_j[1], :])
                y[block_i[0]:block_i[1]] += torch.matmul(U, tmp)

        return y

    def attention_output(self, x):
        """
        Apply H² attention to input, including softmax normalization.

        Args:
            x: Input vector or matrix

        Returns:
            output: Normalized attention output
        """
        # Apply H² matrix
        Ax = self.matvec(x)

        # Apply row-wise softmax normalization
        # For softmax, we need to compute row sums of A
        ones = torch.ones(self.n, device=self.device)
        row_sums = self.matvec(ones)

        # Normalize
        if x.dim() == 1:
            output = Ax / row_sums
        else:
            output = Ax / row_sums.unsqueeze(1)

        return output

def compare_attention_direct(q, k, v, h2_attention, device='cpu', max_error_samples=1000):
    """
    Compare H² attention to full attention without forming the full attention matrix.
    Uses sampling for error estimation with very large sequences.
    For very large sequences, estimates the theoretical full attention time.

    Args:
        q, k: Query and key embeddings
        v: Value embeddings
        h2_attention: H² attention object
        device: Computation device
        max_error_samples: Maximum number of samples to use for error estimation

    Returns:
        error: Relative error between full and H² attention
        time_full: Time for full attention (estimated for very large sequences)
        time_h2: Time for H² attention
    """
    n = q.shape[0]
    d = q.shape[1]

    # For very large sequences, we'll sample rows to estimate the error
    sample_size = min(n, max_error_samples)
    if sample_size < n:
        indices = torch.randperm(n)[:sample_size]
        q_sample = q[indices]
    else:
        indices = torch.arange(n)
        q_sample = q

    # Compute error using sampled rows
    # Full attention (memory-efficient implementation)
    start_time = time()

    # Compute full attention output for sampled rows
    output_full_sample = torch.zeros((sample_size, v.shape[1]), device=device)

    # Process in batches to avoid OOM
    batch_size = 100  # Adjust based on available memory
    for i in range(0, sample_size, batch_size):
        end = min(i + batch_size, sample_size)

        # Compute QK^T / sqrt(d) for this batch
        qk = torch.matmul(q_sample[i:end], k.transpose(0, 1)) / np.sqrt(d)

        # Apply exp to get attention weights
        A = torch.exp(qk)

        # Compute attention output
        Av = torch.matmul(A, v)

        # Normalize
        row_sums = torch.sum(A, dim=1, keepdim=True)
        output_full_sample[i:end] = Av / row_sums

        # Free memory
        del A, qk, Av, row_sums

    sample_time = time() - start_time

    # For very large sequences, estimate the full attention time based on the sample
    # Full attention is O(n²), so we scale the sample time accordingly
    if n > max_error_samples:
        # Estimate time for full attention based on the sample
        # The sample computes max_error_samples * n operations
        # Full attention would compute n * n operations
        time_full = sample_time * (n / max_error_samples)
        print(f"  Estimated full attention time: {time_full:.6f}s (based on sample of {max_error_samples} rows)")
    else:
        time_full = sample_time

    # H² attention
    start_time = time()
    output_h2 = h2_attention.attention_output(v)
    time_h2 = time() - start_time

    # Compute error on sampled rows
    output_h2_sample = output_h2[indices]
    error = torch.norm(output_full_sample - output_h2_sample) / torch.norm(output_full_sample)

    return error.item(), time_full, time_h2

def parse_args():
    parser = argparse.ArgumentParser(description='Memory-efficient H² matrix approximation for attention')
    parser.add_argument('--n', type=int, default=32768, help='Sequence length')
    parser.add_argument('--d', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--dv', type=int, default=64, help='Value dimension')
    parser.add_argument('--block_size', type=int, default=512, help='Block size')
    parser.add_argument('--ranks', type=int, nargs='+', default=[10],
                       help='Ranks for far-field approximation')
    parser.add_argument('--noise_level', type=float, default=0.01, help='Noise level')
    parser.add_argument('--distance', type=int, default=1,
                       help='Minimum number of blocks between far-field blocks')
    parser.add_argument('--eps', type=float, default=1e-6, help='Tolerance for SVD')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device for computations')

    # Randomized SVD parameters
    parser.add_argument('--rsvd_oversampling', type=int, default=10,
                       help='Oversampling parameter for randomized SVD')
    parser.add_argument('--rsvd_n_iter', type=int, default=2,
                       help='Number of power iterations for randomized SVD')
    parser.add_argument('--rsvd_min_block_size', type=int, default=5000,
                       help='Minimum block size to use randomized SVD')

    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Testing memory-efficient H² attention with parameters: {args}")

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

    # Test with different ranks
    results = []

    for rank in args.ranks:
        print(f"Testing rank: {rank}")

        # Create H² approximation
        h2_attention = H2AttentionDirect(
            args.n, args.d, args.block_size, args.distance, rank, args.eps, args.device,
            rsvd_oversampling=args.rsvd_oversampling,
            rsvd_n_iter=args.rsvd_n_iter,
            rsvd_min_block_size=args.rsvd_min_block_size
        )

        # Build H² matrix
        build_time = h2_attention.build(q, k)

        # Compare results
        error, time_full, time_h2 = compare_attention_direct(q, k, v, h2_attention, args.device)

        result = {
            'rank': rank,
            'error': error,
            'time_full': time_full,
            'time_h2': time_h2,
            'speedup': time_full / time_h2,
            'compression_ratio': h2_attention.stats['compression_ratio'],
            'build_time': build_time
        }

        results.append(result)
        print(f"  Error: {error:.6f}")
        print(f"  Time (full): {time_full:.6f}s")
        print(f"  Time (H²): {time_h2:.6f}s")
        print(f"  Speedup: {result['speedup']:.2f}x")

    # Save results
    with open(os.path.join(results_dir, 'h2_attention_direct_results.json'), 'w') as f:
        json.dump({
            'params': vars(args),
            'results': results
        }, f, indent=2)

    # Plot results
    ranks = [r['rank'] for r in results]
    errors = [r['error'] for r in results]
    speedups = [r['speedup'] for r in results]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(ranks, errors, marker='o', linewidth=2)
    plt.title('Approximation Error vs. Rank')
    plt.xlabel('Rank')
    plt.ylabel('Relative Error')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(ranks, speedups, marker='o', linewidth=2)
    plt.title('Speedup vs. Rank')
    plt.xlabel('Rank')
    plt.ylabel('Speedup Factor')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'h2_direct_performance.png'))

    print("Analysis complete! Results saved to:", results_dir)
    print("Visualizations saved to:", viz_dir)

if __name__ == '__main__':
    main()
