#!/usr/bin/env python
"""
Needle-in-Haystack benchmark for H² attention.
Tests the ability to find specific information in long sequences.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import json
import random
from time import time
from tqdm import tqdm
import psutil
from utils import ensure_dir
from recursive_h2_matrix import RecursiveH2Matrix

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class H2SelfAttention(nn.Module):
    """Self-attention layer using RecursiveH2Matrix for efficient computation."""

    def __init__(self, dim, nmin=32, rank=10, eta=0.8, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.nmin = nmin
        self.rank = rank
        self.eta = eta
        self.eps = eps

        # Projection matrices
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)

        Returns:
            Output tensor of shape (batch_size, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Project queries, keys, values
        q = self.q_proj(x)  # (batch_size, seq_len, dim)
        k = self.k_proj(x)  # (batch_size, seq_len, dim)
        v = self.v_proj(x)  # (batch_size, seq_len, dim)

        # Process each item in the batch
        outputs = []
        for b in range(batch_size):
            # Create H² matrix for this batch item
            h2_matrix = RecursiveH2Matrix(
                seq_len, self.nmin, self.rank, self.eta, self.eps, device
            )

            # Build H² matrix from q and k
            h2_matrix.build(q[b], k[b])

            # Apply attention using H² matrix
            output = h2_matrix.attention_output(v[b])
            outputs.append(output)

        # Stack batch outputs
        output = torch.stack(outputs, dim=0)

        # Final projection
        output = self.out_proj(output)

        return output

class StandardSelfAttention(nn.Module):
    """Standard self-attention layer for comparison."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # Projection matrices
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)

        Returns:
            Output tensor of shape (batch_size, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project queries, keys, values
        q = self.q_proj(x)  # (batch_size, seq_len, dim)
        k = self.k_proj(x)  # (batch_size, seq_len, dim)
        v = self.v_proj(x)  # (batch_size, seq_len, dim)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.dim)

        # Apply softmax
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Apply attention to values
        output = torch.matmul(attn_probs, v)

        # Final projection
        output = self.out_proj(output)

        return output

class SimpleTransformer(nn.Module):
    """Simple transformer model for needle-in-haystack task."""

    def __init__(self,
                 vocab_size,
                 dim=128,
                 num_classes=10,
                 use_h2=False,
                 nmin=32,
                 rank=10,
                 eta=0.8,
                 eps=1e-6):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dim)

        # Choose attention mechanism
        if use_h2:
            self.attention = H2SelfAttention(dim, nmin, rank, eta, eps)
        else:
            self.attention = StandardSelfAttention(dim)

        # Layer norm and MLP
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

        # Classification head
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len)

        # Embedding
        x = self.embedding(x)  # (batch_size, seq_len, dim)

        # Attention with residual connection
        x = x + self.attention(self.norm1(x))

        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))

        # Global average pooling
        x = x.mean(dim=1)  # (batch_size, dim)

        # Classification
        x = self.classifier(x)  # (batch_size, num_classes)

        return x

def generate_random_text(length, vocab_size=1000):
    """Generate random token IDs to serve as filler text."""
    return torch.randint(2, vocab_size, (length,))

def generate_needle_in_haystack_dataset(
    context_lengths=[1024, 4096, 8192, 16384],
    needle_positions=[0.1, 0.5, 0.9],  # As fraction of context length
    num_examples=10,
    vocab_size=1000,
    num_classes=10
):
    """Generate a needle-in-haystack dataset."""
    dataset = []

    # Create class-specific token IDs
    class_tokens = list(range(2, 2 + num_classes))  # Reserve 0 for padding, 1 for special token

    for length in context_lengths:
        for position_frac in needle_positions:
            for class_idx in range(num_classes):
                for _ in range(num_examples):
                    # Create irrelevant tokens (random text)
                    context = generate_random_text(length, vocab_size)

                    # Insert needle at specified position
                    pos = int(length * position_frac)
                    context[pos] = class_tokens[class_idx]  # Insert class-specific token

                    # Add to dataset
                    dataset.append({
                        "context": context,
                        "target": class_idx,
                        "metadata": {
                            "context_length": length,
                            "needle_position": pos,
                            "position_fraction": position_frac
                        }
                    })

    return dataset

def evaluate_model(model, dataset, batch_size=1, device='cpu'):
    """Evaluate model on needle-in-haystack dataset."""
    model.eval()
    correct = 0
    total = 0
    results = []

    with torch.no_grad():
        for item in tqdm(dataset, desc="Evaluating"):
            context = item["context"].unsqueeze(0).to(device)  # Add batch dimension
            target = torch.tensor([item["target"]], device=device)

            # Measure memory before forward pass
            memory_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

            # Measure time for forward pass
            start_time = time()
            output = model(context)
            forward_time = time() - start_time

            # Measure memory after forward pass
            memory_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            memory_used = memory_after - memory_before

            # Get prediction
            _, predicted = output.max(1)

            # Update metrics
            total += 1
            correct += (predicted == target).sum().item()

            # Record result
            result = {
                "context_length": item["metadata"]["context_length"],
                "needle_position": item["metadata"]["needle_position"],
                "position_fraction": item["metadata"]["position_fraction"],
                "correct": (predicted == target).item(),
                "forward_time": forward_time,
                "memory_used": memory_used
            }
            results.append(result)

    accuracy = 100.0 * correct / total
    return accuracy, results

def parse_args():
    parser = argparse.ArgumentParser(description='Needle-in-haystack benchmark for H² attention')
    parser.add_argument('--context_lengths', type=int, nargs='+', default=[1024, 2048, 4096, 8192],
                       help='Context lengths to test')
    parser.add_argument('--needle_positions', type=float, nargs='+', default=[0.1, 0.5, 0.9],
                       help='Needle positions as fraction of context length')
    parser.add_argument('--num_examples', type=int, default=5,
                       help='Number of examples per class per context length per position')
    parser.add_argument('--vocab_size', type=int, default=1000, help='Vocabulary size')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--dim', type=int, default=128, help='Model dimension')
    parser.add_argument('--nmin', type=int, default=32, help='Minimum leaf size for H²')
    parser.add_argument('--rank', type=int, default=10, help='Rank for H² approximation')
    parser.add_argument('--eta', type=float, default=0.8, help='Admissibility parameter')
    parser.add_argument('--eps', type=float, default=1e-6, help='Tolerance for SVD')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device for computations')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Needle-in-haystack benchmark with parameters: {args}")

    # Setup output directories
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    viz_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visualizations')
    ensure_dir(results_dir)
    ensure_dir(viz_dir)

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Generate dataset
    print("Generating needle-in-haystack dataset...")
    dataset = generate_needle_in_haystack_dataset(
        context_lengths=args.context_lengths,
        needle_positions=args.needle_positions,
        num_examples=args.num_examples,
        vocab_size=args.vocab_size,
        num_classes=args.num_classes
    )
    print(f"Generated {len(dataset)} examples")

    # Initialize models
    print("Initializing models...")
    standard_model = SimpleTransformer(
        vocab_size=args.vocab_size,
        dim=args.dim,
        num_classes=args.num_classes,
        use_h2=False
    ).to(device)

    h2_model = SimpleTransformer(
        vocab_size=args.vocab_size,
        dim=args.dim,
        num_classes=args.num_classes,
        use_h2=True,
        nmin=args.nmin,
        rank=args.rank,
        eta=args.eta,
        eps=args.eps
    ).to(device)

    # Evaluate standard model
    print("\nEvaluating standard attention model...")
    try:
        standard_accuracy, standard_results = evaluate_model(
            standard_model, dataset, args.batch_size, device
        )
        print(f"Standard Model Accuracy: {standard_accuracy:.2f}%")
    except RuntimeError as e:
        print(f"Error evaluating standard model: {e}")
        print("This is likely due to memory constraints with long sequences.")
        standard_accuracy = 0
        standard_results = []

    # Evaluate H² model
    print("\nEvaluating H² attention model...")
    h2_accuracy, h2_results = evaluate_model(
        h2_model, dataset, args.batch_size, device
    )
    print(f"H² Model Accuracy: {h2_accuracy:.2f}%")

    # Save results
    results = {
        'params': vars(args),
        'standard': {
            'accuracy': standard_accuracy,
            'results': standard_results
        },
        'h2': {
            'accuracy': h2_accuracy,
            'results': h2_results
        }
    }

    with open(os.path.join(results_dir, 'needle_in_haystack_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Analyze results by context length and needle position
    if h2_results:
        # Group results by context length and position
        h2_by_length_pos = {}
        for result in h2_results:
            length = result['context_length']
            pos_frac = result['position_fraction']
            key = (length, pos_frac)
            if key not in h2_by_length_pos:
                h2_by_length_pos[key] = []
            h2_by_length_pos[key].append(result)

        # Calculate metrics for each group
        h2_metrics = {}
        for key, group in h2_by_length_pos.items():
            length, pos_frac = key
            accuracy = 100 * sum(r['correct'] for r in group) / len(group)
            avg_time = sum(r['forward_time'] for r in group) / len(group)
            avg_memory = sum(r['memory_used'] for r in group) / len(group)
            h2_metrics[key] = {
                'accuracy': accuracy,
                'avg_time': avg_time,
                'avg_memory': avg_memory
            }

        # Do the same for standard model if results exist
        standard_by_length_pos = {}
        standard_metrics = {}
        if standard_results:
            for result in standard_results:
                length = result['context_length']
                pos_frac = result['position_fraction']
                key = (length, pos_frac)
                if key not in standard_by_length_pos:
                    standard_by_length_pos[key] = []
                standard_by_length_pos[key].append(result)

            for key, group in standard_by_length_pos.items():
                length, pos_frac = key
                accuracy = 100 * sum(r['correct'] for r in group) / len(group)
                avg_time = sum(r['forward_time'] for r in group) / len(group)
                avg_memory = sum(r['memory_used'] for r in group) / len(group)
                standard_metrics[key] = {
                    'accuracy': accuracy,
                    'avg_time': avg_time,
                    'avg_memory': avg_memory
                }

        # Plot results
        plt.figure(figsize=(15, 15))

        # Accuracy by context length for different positions
        plt.subplot(2, 2, 1)
        for pos in args.needle_positions:
            h2_acc = [h2_metrics.get((length, pos), {}).get('accuracy', 0) for length in args.context_lengths]
            plt.plot(args.context_lengths, h2_acc, 'o-', label=f'H² (pos={pos})')

            if standard_metrics:
                std_acc = [standard_metrics.get((length, pos), {}).get('accuracy', 0) for length in args.context_lengths]
                plt.plot(args.context_lengths, std_acc, 'x--', label=f'Standard (pos={pos})')

        plt.title('Accuracy vs. Context Length')
        plt.xlabel('Context Length')
        plt.ylabel('Accuracy (%)')
        plt.xscale('log')
        plt.grid(True)
        plt.legend()

        # Inference time by context length
        plt.subplot(2, 2, 2)
        for pos in args.needle_positions:
            h2_time = [h2_metrics.get((length, pos), {}).get('avg_time', 0) for length in args.context_lengths]
            plt.plot(args.context_lengths, h2_time, 'o-', label=f'H² (pos={pos})')

            if standard_metrics:
                std_time = [standard_metrics.get((length, pos), {}).get('avg_time', 0) for length in args.context_lengths]
                plt.plot(args.context_lengths, std_time, 'x--', label=f'Standard (pos={pos})')

        plt.title('Inference Time vs. Context Length')
        plt.xlabel('Context Length')
        plt.ylabel('Time (s)')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()

        # Memory usage by context length
        plt.subplot(2, 2, 3)
        for pos in args.needle_positions:
            h2_mem = [h2_metrics.get((length, pos), {}).get('avg_memory', 0) for length in args.context_lengths]
            plt.plot(args.context_lengths, h2_mem, 'o-', label=f'H² (pos={pos})')

            if standard_metrics:
                std_mem = [standard_metrics.get((length, pos), {}).get('avg_memory', 0) for length in args.context_lengths]
                plt.plot(args.context_lengths, std_mem, 'x--', label=f'Standard (pos={pos})')

        plt.title('Memory Usage vs. Context Length')
        plt.xlabel('Context Length')
        plt.ylabel('Memory (MB)')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()

        # Accuracy by needle position for different context lengths
        plt.subplot(2, 2, 4)
        for length in args.context_lengths:
            h2_acc = [h2_metrics.get((length, pos), {}).get('accuracy', 0) for pos in args.needle_positions]
            plt.plot(args.needle_positions, h2_acc, 'o-', label=f'H² (len={length})')

            if standard_metrics:
                std_acc = [standard_metrics.get((length, pos), {}).get('accuracy', 0) for pos in args.needle_positions]
                plt.plot(args.needle_positions, std_acc, 'x--', label=f'Standard (len={length})')

        plt.title('Accuracy vs. Needle Position')
        plt.xlabel('Position (fraction of context)')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'needle_in_haystack_results.png'))

    print("\nBenchmark complete!")
    print(f"Results saved to: {results_dir}")
    print(f"Visualizations saved to: {viz_dir}")

if __name__ == '__main__':
    main()
