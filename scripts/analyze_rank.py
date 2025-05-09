#!/usr/bin/env python
"""
Analyze the numerical rank of far-field blocks in the attention matrix.
This script validates the key claim of the H²-compressibility conjecture:
that far-field blocks have low numerical rank.
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import json
from utils import (
    generate_smooth_embeddings, compute_attention_matrix,
    get_block_indices, is_far_field, measure_block_rank, ensure_dir
)

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze rank of attention matrix blocks')
    parser.add_argument('--n', type=int, default=1024, help='Sequence length')
    parser.add_argument('--d', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--block_size', type=int, default=64, help='Block size')
    parser.add_argument('--eps', type=float, default=1e-6, help='Tolerance for numerical rank')
    parser.add_argument('--noise_levels', type=float, nargs='+', default=[0.0, 0.01, 0.05, 0.1],
                       help='Noise levels to test')
    parser.add_argument('--distance', type=int, default=1, 
                       help='Minimum number of blocks between far-field blocks')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device for computations')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Analyzing attention matrix blocks with parameters: {args}")
    
    # Setup output directories
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    viz_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visualizations')
    ensure_dir(results_dir)
    ensure_dir(viz_dir)
    
    block_indices = get_block_indices(args.n, args.block_size)
    n_blocks = len(block_indices)
    
    # Identify far-field block pairs
    far_field_pairs = []
    for i in range(n_blocks):
        for j in range(n_blocks):
            if is_far_field(block_indices[i], block_indices[j], args.distance):
                far_field_pairs.append((i, j))
    
    print(f"Found {len(far_field_pairs)} far-field block pairs out of {n_blocks*n_blocks} total")
    
    # Test with different noise levels
    results = {}
    
    for noise_level in args.noise_levels:
        print(f"Testing noise level: {noise_level}")
        q, k = generate_smooth_embeddings(args.n, args.d, noise_level=noise_level)
        q = q.to(args.device)
        k = k.to(args.device)
        
        A = compute_attention_matrix(q, k)
        
        # Measure rank of far-field blocks
        far_field_ranks = []
        singular_values = []
        
        for i, j in far_field_pairs[:min(100, len(far_field_pairs))]:  # Limit to 100 pairs for speed
            block_i = block_indices[i]
            block_j = block_indices[j]
            rank, s_norm = measure_block_rank(A, block_i, block_j, args.eps)
            far_field_ranks.append(rank)
            singular_values.append(s_norm[:20])  # Keep top 20 singular values
        
        # Save results
        result_data = {
            'params': vars(args),
            'noise_level': noise_level,
            'far_field_ranks': far_field_ranks,
            'singular_values': singular_values,
            'avg_rank': np.mean(far_field_ranks),
            'max_rank': np.max(far_field_ranks),
            'block_size': args.block_size
        }
        
        results[f'noise_{noise_level}'] = result_data
        
        # Plot singular value decay for a few blocks
        plt.figure(figsize=(10, 6))
        for i in range(min(10, len(singular_values))):
            s = singular_values[i]
            plt.semilogy(range(1, len(s)+1), s, marker='o', markersize=3, alpha=0.7)
        
        plt.title(f'Singular Value Decay (Noise Level: {noise_level})')
        plt.xlabel('Index')
        plt.ylabel('Normalized Singular Value')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f'singular_value_decay_noise_{noise_level}.png'))
        plt.close()
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Convert results for JSON
    json_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            json_results[key] = {k: convert_for_json(v) for k, v in value.items()}
        else:
            json_results[key] = convert_for_json(value)
    
    # Save all results
    try:
        with open(os.path.join(results_dir, 'rank_analysis_results.json'), 'w') as f:
            json.dump(json_results, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save results to JSON: {e}")
        print("Continuing with visualization...")
    
    # Plot summary of average ranks vs noise levels
    noise_levels = list(args.noise_levels)
    avg_ranks = [results[f'noise_{nl}']['avg_rank'] for nl in noise_levels]
    
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, avg_ranks, marker='o', linewidth=2)
    plt.title('Average Far-Field Block Rank vs. Noise Level')
    plt.xlabel('Noise Level')
    plt.ylabel('Average Numerical Rank')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'avg_rank_vs_noise.png'))
    
    print("Analysis complete! Results saved to:", results_dir)
    print("Visualizations saved to:", viz_dir)

if __name__ == '__main__':
    main()
