#!/usr/bin/env python
"""
Hybrid implementation of H² matrix approximation for attention.
Allows building on CUDA and inference on any device (including MPS).
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import json
import pickle
from time import time
from utils import (
    generate_smooth_embeddings, compute_attention_matrix,
    get_block_indices, is_far_field, randomized_svd, ensure_dir
)

class H2AttentionHybrid:
    """
    Hybrid H² matrix approximation for attention.
    Supports building on CUDA and inference on any device (including MPS).
    """

    def __init__(self, n, d, block_size=64, distance=1, rank=10, eps=1e-6, device='auto',
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
            device: Computation device ('auto', 'cpu', 'cuda', 'mps')
            rsvd_oversampling: Oversampling parameter for randomized SVD
            rsvd_n_iter: Number of power iterations for randomized SVD
            rsvd_min_block_size: Minimum block size to use randomized SVD
        """
        # Determine device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print("Using CUDA for computation")
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
                print("Using MPS for computation")
            else:
                self.device = torch.device('cpu')
                print("Using CPU for computation")
        else:
            self.device = torch.device(device)
            print(f"Using {device} for computation")

        self.n = n
        self.d = d
        self.block_size = block_size
        self.distance = distance
        self.rank = rank
        self.eps = eps

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

        # Flag to indicate if the model is built
        self.is_built = False

    def build(self, q, k):
        """
        Build the H² approximation directly from query and key embeddings.

        Args:
            q: Query embeddings (n, d)
            k: Key embeddings (n, d)

        Returns:
            build_time: Time taken to build the H² matrix
        """
        # Move data to device
        q = q.to(self.device)
        k = k.to(self.device)

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

        # Set built flag
        self.is_built = True

        return build_time

    def _build_near_field_direct(self, q, k, i, j):
        """Store near-field block as dense matrix, computed directly from q and k."""
        block_i = self.block_indices[i]
        block_j = self.block_indices[j]

        q_block = q[block_i[0]:block_i[1]]
        k_block = k[block_j[0]:block_j[1]]

        # Compute attention block
        d = q_block.shape[1]
        qk = torch.matmul(q_block, k_block.transpose(0, 1)) / np.sqrt(d)
        A_block = torch.exp(qk)

        self.near_field[(i, j)] = A_block

    def _build_far_field_direct(self, q, k, i, j):
        """Create low-rank approximation for far-field block directly from q and k."""
        block_i = self.block_indices[i]
        block_j = self.block_indices[j]

        q_block = q[block_i[0]:block_i[1]]
        k_block = k[block_j[0]:block_j[1]]

        # Compute attention block
        d = q_block.shape[1]
        qk = torch.matmul(q_block, k_block.transpose(0, 1)) / np.sqrt(d)
        A_block = torch.exp(qk)

        # Determine if we should use randomized SVD based on block size
        block_size = A_block.shape[0] * A_block.shape[1]

        if block_size > self.rsvd_min_block_size:
            # Use randomized SVD implementation
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
            u, s, v = torch.linalg.svd(A_block, full_matrices=False)

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

    def save(self, filepath):
        """
        Save the H² matrix to disk.

        Args:
            filepath: Path to save the H² matrix
        """
        if not self.is_built:
            raise ValueError("H² matrix must be built before saving")

        # Move data to CPU for saving
        near_field_cpu = {}
        for key, value in self.near_field.items():
            near_field_cpu[key] = value.cpu()

        far_field_cpu = {}
        for key, (U, V) in self.far_field.items():
            far_field_cpu[key] = (U.cpu(), V.cpu())

        # Create save dictionary
        save_dict = {
            'n': self.n,
            'd': self.d,
            'block_size': self.block_size,
            'distance': self.distance,
            'rank': self.rank,
            'eps': self.eps,
            'block_indices': self.block_indices,
            'n_blocks': self.n_blocks,
            'near_field': near_field_cpu,
            'far_field': far_field_cpu,
            'stats': self.stats
        }

        # Save to disk
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)

        print(f"H² matrix saved to {filepath}")

    @classmethod
    def load(cls, filepath, device='auto'):
        """
        Load a pre-built H² matrix from disk.

        Args:
            filepath: Path to the saved H² matrix
            device: Device to load the H² matrix onto

        Returns:
            h2_attention: Loaded H² attention object
        """
        # Load from disk
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)

        # Create H² attention object
        h2_attention = cls(
            save_dict['n'],
            save_dict['d'],
            save_dict['block_size'],
            save_dict['distance'],
            save_dict['rank'],
            save_dict['eps'],
            device
        )

        # Determine device
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        # Set attributes
        h2_attention.block_indices = save_dict['block_indices']
        h2_attention.n_blocks = save_dict['n_blocks']
        h2_attention.stats = save_dict['stats']

        # Move data to device
        h2_attention.near_field = {}
        for key, value in save_dict['near_field'].items():
            h2_attention.near_field[key] = value.to(h2_attention.device)

        h2_attention.far_field = {}
        for key, (U, V) in save_dict['far_field'].items():
            h2_attention.far_field[key] = (U.to(h2_attention.device), V.to(h2_attention.device))

        # Set built flag
        h2_attention.is_built = True

        print(f"H² matrix loaded from {filepath} to {device}")

        return h2_attention

    def matvec(self, x):
        """
        Apply the H² matrix to a vector.

        Args:
            x: Input vector (n,) or matrix (n, d)

        Returns:
            y: Output vector (n,) or matrix (n, d)
        """
        if not self.is_built:
            raise ValueError("H² matrix must be built before applying")

        # Move input to device
        x = x.to(self.device)

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
                tmp = torch.matmul(V.T, x[block_j[0]:block_j[1]])
                y[block_i[0]:block_i[1]] += torch.matmul(U, tmp)
            else:
                tmp = torch.matmul(V.T, x[block_j[0]:block_j[1], :])
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
        if not self.is_built:
            raise ValueError("H² matrix must be built before applying attention")

        # Move input to device
        x = x.to(self.device)

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

def parse_args():
    parser = argparse.ArgumentParser(description='Hybrid H² matrix approximation for attention')
    parser.add_argument('--mode', type=str, choices=['build', 'inference'], required=True,
                       help='Mode: build (on CUDA) or inference (on any device)')
    parser.add_argument('--n', type=int, default=16384, help='Sequence length')
    parser.add_argument('--d', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--dv', type=int, default=64, help='Value dimension')
    parser.add_argument('--block_size', type=int, default=256, help='Block size')
    parser.add_argument('--rank', type=int, default=10, help='Rank for far-field approximation')
    parser.add_argument('--distance', type=int, default=1,
                       help='Minimum number of blocks between far-field blocks')
    parser.add_argument('--eps', type=float, default=1e-6, help='Tolerance for SVD')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device for computation (auto, cpu, cuda, mps)')
    parser.add_argument('--model_path', type=str, default='models/h2_model.pkl',
                       help='Path to save/load the H² model')

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
    print(f"Running hybrid H² attention in {args.mode} mode with parameters: {args}")

    # Setup output directories
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    viz_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visualizations')
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    ensure_dir(results_dir)
    ensure_dir(viz_dir)
    ensure_dir(model_dir)

    model_path = os.path.join(model_dir, os.path.basename(args.model_path))

    if args.mode == 'build':
        # Build mode: Create and save the H² matrix
        print(f"Building H² matrix for sequence length {args.n}...")

        # Generate embeddings
        q, k = generate_smooth_embeddings(args.n, args.d, noise_level=0.01)

        # Create H² attention object
        h2_attention = H2AttentionHybrid(
            args.n, args.d, args.block_size, args.distance, args.rank, args.eps, args.device,
            rsvd_oversampling=args.rsvd_oversampling,
            rsvd_n_iter=args.rsvd_n_iter,
            rsvd_min_block_size=args.rsvd_min_block_size
        )

        # Build H² matrix
        build_time = h2_attention.build(q, k)

        # Save model
        h2_attention.save(model_path)

        print(f"H² matrix built in {build_time:.3f} seconds and saved to {model_path}")

    elif args.mode == 'inference':
        # Inference mode: Load the H² matrix and run inference
        if not os.path.exists(model_path):
            raise ValueError(f"Model file {model_path} does not exist. Run in build mode first.")

        print(f"Loading H² matrix from {model_path}...")

        # Load H² attention object
        h2_attention = H2AttentionHybrid.load(model_path, device=args.device)

        # Generate test data
        v = torch.randn(args.n, args.dv)

        # Run inference
        print("Running inference...")
        start_time = time()
        output = h2_attention.attention_output(v)
        inference_time = time() - start_time

        print(f"Inference completed in {inference_time:.6f} seconds")
        print(f"Output shape: {output.shape}")

        # Save results
        results = {
            'params': vars(args),
            'stats': h2_attention.stats,
            'inference_time': inference_time
        }

        with open(os.path.join(results_dir, 'h2_hybrid_inference_results.json'), 'w') as f:
            # Convert to serializable format
            serializable_results = {k: v for k, v in results.items() if k != 'stats'}
            serializable_results['stats'] = {k: float(v) if isinstance(v, (int, float)) else v
                                           for k, v in results['stats'].items()}
            json.dump(serializable_results, f, indent=2)

        print(f"Results saved to {os.path.join(results_dir, 'h2_hybrid_inference_results.json')}")

if __name__ == '__main__':
    main()
