#!/usr/bin/env python
"""
Analyze the far-field rank properties of attention matrices from pre-trained models.
This helps determine if the H²-compressibility conjecture holds in practice.
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import json
from time import time
from utils import get_block_indices, is_far_field, measure_block_rank, ensure_dir
from transformers import AutoModel, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze attention matrices from pre-trained models')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                      help='Hugging Face model name')
    parser.add_argument('--max_length', type=int, default=512,
                      help='Maximum sequence length')
    parser.add_argument('--block_size', type=int, default=32, help='Block size')
    parser.add_argument('--distance', type=int, default=1, 
                      help='Minimum number of blocks between far-field blocks')
    parser.add_argument('--eps', type=float, default=1e-6, help='Tolerance for numerical rank')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device for computations')
    parser.add_argument('--num_samples', type=int, default=10,
                      help='Number of input samples to test')
    return parser.parse_args()

def generate_random_input(tokenizer, max_length):
    """Generate random text input for testing."""
    vocab_size = tokenizer.vocab_size
    random_ids = torch.randint(0, vocab_size, (max_length,))
    return tokenizer.decode(random_ids, skip_special_tokens=True)

def extract_attention_matrices(model, tokenizer, text, device='cpu'):
    """Extract attention matrices from all layers and heads of a model."""
    # Tokenize input
    inputs = tokenizer(text, return_tensors='pt').to(device)
    
    # Forward pass with output_attentions=True
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Get attention matrices
    attention_matrices = outputs.attentions
    
    return attention_matrices, inputs.input_ids.size(1)

def analyze_attention_matrices(attention_matrices, block_size, distance, eps):
    """Analyze the rank properties of attention matrices."""
    results = []
    
    for layer_idx, layer_attention in enumerate(attention_matrices):
        layer_results = []
        
        for head_idx, head_attention in enumerate(layer_attention[0]):
            # Get attention matrix
            A = head_attention.cpu()
            n = A.shape[0]
            
            # Get block indices
            block_indices = get_block_indices(n, block_size)
            n_blocks = len(block_indices)
            
            # Identify far-field block pairs
            far_field_pairs = []
            for i in range(n_blocks):
                for j in range(n_blocks):
                    if is_far_field(block_indices[i], block_indices[j], distance):
                        far_field_pairs.append((i, j))
            
            # Measure rank of far-field blocks
            far_field_ranks = []
            singular_values = []
            
            for i, j in far_field_pairs[:min(20, len(far_field_pairs))]:  # Limit to 20 pairs for speed
                block_i = block_indices[i]
                block_j = block_indices[j]
                rank, s_norm = measure_block_rank(A, block_i, block_j, eps)
                far_field_ranks.append(rank)
                singular_values.append(s_norm[:10])  # Keep top 10 singular values
            
            head_result = {
                'layer': layer_idx,
                'head': head_idx,
                'far_field_ranks': far_field_ranks,
                'singular_values': singular_values,
                'avg_rank': np.mean(far_field_ranks) if far_field_ranks else float('nan'),
                'max_rank': np.max(far_field_ranks) if far_field_ranks else float('nan'),
                'min_rank': np.min(far_field_ranks) if far_field_ranks else float('nan'),
            }
            
            layer_results.append(head_result)
        
        results.append(layer_results)
    
    return results

def main():
    args = parse_args()
    print(f"Analyzing attention matrices from {args.model_name}")
    
    # Setup output directories
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    viz_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visualizations')
    ensure_dir(results_dir)
    ensure_dir(viz_dir)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name, output_attentions=True)
    model.to(args.device)
    model.eval()
    
    all_results = []
    
    for sample_idx in range(args.num_samples):
        print(f"Processing sample {sample_idx+1}/{args.num_samples}")
        
        # Generate random input text
        text = generate_random_input(tokenizer, args.max_length)
        
        # Extract attention matrices
        attention_matrices, seq_length = extract_attention_matrices(model, tokenizer, text, args.device)
        
        # Analyze attention matrices
        results = analyze_attention_matrices(attention_matrices, args.block_size, args.distance, args.eps)
        
        all_results.append({
            'sample_idx': sample_idx,
            'seq_length': seq_length,
            'results': results
        })
    
    # Save results
    with open(os.path.join(results_dir, f'pretrained_analysis_{args.model_name.replace("/", "_")}.json'), 'w') as f:
        json.dump({
            'params': vars(args),
            'all_results': all_results
        }, f, indent=2)
    
    # Compute average ranks across all samples
    avg_ranks = []
    for layer_idx in range(len(all_results[0]['results'])):
        layer_ranks = []
        for head_idx in range(len(all_results[0]['results'][layer_idx])):
            head_ranks = []
            for sample in all_results:
                rank = sample['results'][layer_idx][head_idx]['avg_rank']
                if not np.isnan(rank):
                    head_ranks.append(rank)
            
            if head_ranks:
                layer_ranks.append(np.mean(head_ranks))
            else:
                layer_ranks.append(float('nan'))
        
        avg_ranks.append(layer_ranks)
    
    # Plot average ranks by layer and head
    plt.figure(figsize=(12, 8))
    plt.imshow(avg_ranks, cmap='viridis')
    plt.colorbar(label='Average Rank')
    plt.xlabel('Head')
    plt.ylabel('Layer')
    plt.title(f'Average Far-Field Block Rank in {args.model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'pretrained_ranks_{args.model_name.replace("/", "_")}.png'))
    
    # Plot singular value decay for a few representative blocks
    plt.figure(figsize=(12, 8))
    
    # Pick a middle layer, middle head, and first sample
    layer_idx = len(all_results[0]['results']) // 2
    head_idx = len(all_results[0]['results'][layer_idx]) // 2
    sample_idx = 0
    
    singular_values = all_results[sample_idx]['results'][layer_idx][head_idx]['singular_values']
    
    for i, s in enumerate(singular_values[:5]):  # Plot first 5 blocks
        plt.semilogy(range(1, len(s)+1), s, marker='o', label=f'Block {i+1}')
    
    plt.title(f'Singular Value Decay in {args.model_name} (Layer {layer_idx}, Head {head_idx})')
    plt.xlabel('Index')
    plt.ylabel('Normalized Singular Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'pretrained_svd_{args.model_name.replace("/", "_")}.png'))
    
    print("Analysis complete! Results saved to:", results_dir)
    print("Visualizations saved to:", viz_dir)

if __name__ == '__main__':
    main()
