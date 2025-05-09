#!/usr/bin/env python
"""
Benchmark the RecursiveH2Matrix implementation on MNIST classification task.
Compares a Vision Transformer with H² attention against standard attention.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
import matplotlib.pyplot as plt
import os
import json
from time import time
from tqdm import tqdm
import psutil
from utils import ensure_dir
from recursive_h2_matrix import RecursiveH2Matrix

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

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

class TransformerBlock(nn.Module):
    """Transformer block with either H² or standard attention."""

    def __init__(self, dim, use_h2=False, nmin=32, rank=10, eta=0.8, eps=1e-6):
        super().__init__()

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

    def forward(self, x):
        # Attention with residual connection
        x = x + self.attention(self.norm1(x))

        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))

        return x

class SimpleViT(nn.Module):
    """Simple Vision Transformer for MNIST classification."""

    def __init__(self,
                 image_size=28,
                 patch_size=2,
                 num_classes=10,
                 dim=128,
                 depth=4,
                 use_h2=False,
                 nmin=32,
                 rank=10,
                 eta=0.8,
                 eps=1e-6):
        super().__init__()

        assert image_size % patch_size == 0, "Image size must be divisible by patch size"

        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = 1 * patch_size ** 2  # 1 channel for MNIST

        # Patch embedding
        self.patch_embed = nn.Linear(patch_dim, dim)

        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim, use_h2, nmin, rank, eta, eps)
            for _ in range(depth)
        ])

        # Classification head
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        batch_size = x.shape[0]

        # Extract patches
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, 1, -1, self.patch_size * self.patch_size)
        x = x.transpose(1, 2).reshape(batch_size, -1, self.patch_size * self.patch_size)

        # Patch embedding
        x = self.patch_embed(x)

        # Add position embedding
        x = x + self.pos_embed

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Classification
        x = self.norm(x)
        x = self.head(x)

        return x

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(dataloader)

    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(dataloader)

    return avg_loss, accuracy

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def parse_args():
    parser = argparse.ArgumentParser(description='MNIST benchmark for H² attention')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patch_size', type=int, default=2, help='Patch size')
    parser.add_argument('--dim', type=int, default=128, help='Model dimension')
    parser.add_argument('--depth', type=int, default=4, help='Number of transformer blocks')
    parser.add_argument('--nmin', type=int, default=16, help='Minimum leaf size for H²')
    parser.add_argument('--rank', type=int, default=10, help='Rank for H² approximation')
    parser.add_argument('--eta', type=float, default=0.8, help='Admissibility parameter')
    parser.add_argument('--eps', type=float, default=1e-6, help='Tolerance for SVD')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device for computations')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"MNIST benchmark with parameters: {args}")

    # Setup output directories
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    viz_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visualizations')
    ensure_dir(results_dir)
    ensure_dir(viz_dir)

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load MNIST dataset from Hugging Face
    print("Loading MNIST dataset...")
    mnist_dataset = load_dataset("mnist")

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Create a custom dataset class
    class MNISTDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, transform=None):
            self.dataset = dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            image = np.array(self.dataset[idx]["image"]).astype(np.uint8)
            label = self.dataset[idx]["label"]

            if self.transform:
                image = self.transform(image.reshape(28, 28, 1))

            return image, label

    # Create datasets
    train_dataset = MNISTDataset(mnist_dataset["train"], transform)
    test_dataset = MNISTDataset(mnist_dataset["test"], transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Initialize models
    print("Initializing models...")
    standard_model = SimpleViT(
        image_size=28,
        patch_size=args.patch_size,
        num_classes=10,
        dim=args.dim,
        depth=args.depth,
        use_h2=False
    ).to(device)

    h2_model = SimpleViT(
        image_size=28,
        patch_size=args.patch_size,
        num_classes=10,
        dim=args.dim,
        depth=args.depth,
        use_h2=True,
        nmin=args.nmin,
        rank=args.rank,
        eta=args.eta,
        eps=args.eps
    ).to(device)

    # Define optimizers and loss function
    standard_optimizer = optim.Adam(standard_model.parameters(), lr=args.lr)
    h2_optimizer = optim.Adam(h2_model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Training and evaluation results
    standard_results = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_times': [],
        'memory_usage': []
    }

    h2_results = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_times': [],
        'memory_usage': []
    }

    # Train and evaluate standard model
    print("\nTraining standard attention model...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Measure memory before training
        memory_before = get_memory_usage()

        # Train
        start_time = time()
        train_loss, train_acc = train_epoch(standard_model, train_loader, standard_optimizer, criterion, device)
        epoch_time = time() - start_time

        # Measure memory after training
        memory_after = get_memory_usage()
        memory_used = memory_after - memory_before

        # Evaluate
        val_loss, val_acc = evaluate(standard_model, test_loader, criterion, device)

        # Record results
        standard_results['train_loss'].append(train_loss)
        standard_results['train_acc'].append(train_acc)
        standard_results['val_loss'].append(val_loss)
        standard_results['val_acc'].append(val_acc)
        standard_results['epoch_times'].append(epoch_time)
        standard_results['memory_usage'].append(memory_used)

        print(f"Standard Model - Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Time: {epoch_time:.2f}s, Memory: {memory_used:.2f}MB")

    # Train and evaluate H² model
    print("\nTraining H² attention model...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Measure memory before training
        memory_before = get_memory_usage()

        # Train
        start_time = time()
        train_loss, train_acc = train_epoch(h2_model, train_loader, h2_optimizer, criterion, device)
        epoch_time = time() - start_time

        # Measure memory after training
        memory_after = get_memory_usage()
        memory_used = memory_after - memory_before

        # Evaluate
        val_loss, val_acc = evaluate(h2_model, test_loader, criterion, device)

        # Record results
        h2_results['train_loss'].append(train_loss)
        h2_results['train_acc'].append(train_acc)
        h2_results['val_loss'].append(val_loss)
        h2_results['val_acc'].append(val_acc)
        h2_results['epoch_times'].append(epoch_time)
        h2_results['memory_usage'].append(memory_used)

        print(f"H² Model - Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Time: {epoch_time:.2f}s, Memory: {memory_used:.2f}MB")

    # Save results
    results = {
        'params': vars(args),
        'standard': standard_results,
        'h2': h2_results
    }

    with open(os.path.join(results_dir, 'mnist_benchmark_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot results
    plt.figure(figsize=(15, 10))

    # Accuracy plot
    plt.subplot(2, 2, 1)
    plt.plot(range(1, args.epochs+1), standard_results['val_acc'], 'o-', label='Standard Attention')
    plt.plot(range(1, args.epochs+1), h2_results['val_acc'], 'o-', label='H² Attention')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Loss plot
    plt.subplot(2, 2, 2)
    plt.plot(range(1, args.epochs+1), standard_results['val_loss'], 'o-', label='Standard Attention')
    plt.plot(range(1, args.epochs+1), h2_results['val_loss'], 'o-', label='H² Attention')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Training time plot
    plt.subplot(2, 2, 3)
    plt.bar(['Standard', 'H²'],
            [np.mean(standard_results['epoch_times']), np.mean(h2_results['epoch_times'])],
            yerr=[np.std(standard_results['epoch_times']), np.std(h2_results['epoch_times'])])
    plt.title('Average Epoch Time')
    plt.ylabel('Time (s)')
    plt.grid(True)

    # Memory usage plot
    plt.subplot(2, 2, 4)
    plt.bar(['Standard', 'H²'],
            [np.mean(standard_results['memory_usage']), np.mean(h2_results['memory_usage'])],
            yerr=[np.std(standard_results['memory_usage']), np.std(h2_results['memory_usage'])])
    plt.title('Average Memory Usage')
    plt.ylabel('Memory (MB)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'mnist_benchmark_results.png'))

    print("\nBenchmark complete!")
    print(f"Final Standard Model Accuracy: {standard_results['val_acc'][-1]:.2f}%")
    print(f"Final H² Model Accuracy: {h2_results['val_acc'][-1]:.2f}%")
    print(f"Average Standard Model Epoch Time: {np.mean(standard_results['epoch_times']):.2f}s")
    print(f"Average H² Model Epoch Time: {np.mean(h2_results['epoch_times']):.2f}s")
    print(f"Results saved to: {results_dir}")
    print(f"Visualizations saved to: {viz_dir}")

if __name__ == '__main__':
    main()
