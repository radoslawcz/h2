import numpy as np
import torch
import matplotlib.pyplot as plt
from time import time
import os
from typing import Callable, Tuple, List, Dict, Any, Optional

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_smooth_embeddings(
    n: int,
    d: int,
    func: Callable[[np.ndarray], np.ndarray] = None,
    noise_level: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate smooth token embeddings according to an analytic function.

    Args:
        n: Number of tokens
        d: Embedding dimension
        func: Function to generate embeddings (default: sinusoidal)
        noise_level: Amount of noise to add (0.0 = no noise)

    Returns:
        q, k: Query and key embeddings as tensors of shape (n, d)
    """
    # Default function: sinusoidal embedding similar to positional encoding
    if func is None:
        def default_func(x):
            result = np.zeros((len(x), d))
            for i in range(d // 2):
                result[:, 2*i] = np.sin(x / (10000 ** (2*i/d)))
                result[:, 2*i+1] = np.cos(x / (10000 ** (2*i/d)))
            if d % 2 == 1:  # Handle odd dimensions
                result[:, -1] = np.sin(x / (10000 ** ((d-1)/d)))
            return result
        func = default_func

    # Generate positions in [0, 1]
    x = np.linspace(0, 1, n)

    # Apply function to get base embeddings
    emb = func(x)

    # Add noise if specified
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, emb.shape)
        emb = emb + noise

    # Normalize to [-1, 1] range
    max_val = np.max(np.abs(emb))
    if max_val > 0:
        emb = emb / max_val

    # Convert to tensors
    q = torch.tensor(emb, dtype=torch.float32)
    k = torch.tensor(emb, dtype=torch.float32)  # Using same embeddings for Q and K

    return q, k

def compute_attention_matrix(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Compute the unnormalized attention matrix.

    Args:
        q: Query embeddings (n, d)
        k: Key embeddings (n, d)

    Returns:
        A: Attention matrix (n, n)
    """
    d = q.shape[1]
    qk = torch.matmul(q, k.transpose(0, 1)) / np.sqrt(d)
    A = torch.exp(qk)
    return A

def get_block_indices(n: int, block_size: int) -> List[Tuple[int, int]]:
    """
    Get start and end indices for blocks of size block_size.

    Args:
        n: Total size
        block_size: Size of each block

    Returns:
        List of (start, end) tuples
    """
    return [(i, min(i + block_size, n)) for i in range(0, n, block_size)]

def is_far_field(block_i: Tuple[int, int], block_j: Tuple[int, int], distance: int = 1) -> bool:
    """
    Determine if two blocks are in the far field.

    Args:
        block_i: (start, end) indices of first block
        block_j: (start, end) indices of second block
        distance: Minimum number of blocks separating far-field blocks

    Returns:
        True if blocks are in far field, False otherwise
    """
    return min(abs(block_i[0] - block_j[1]), abs(block_i[1] - block_j[0])) >= distance * (block_i[1] - block_i[0])

def measure_block_rank(
    A: torch.Tensor,
    block_i: Tuple[int, int],
    block_j: Tuple[int, int],
    eps: float = 1e-6
) -> Tuple[int, List[float]]:
    """
    Measure the numerical rank of a block in the attention matrix.

    Args:
        A: Attention matrix
        block_i: Row indices (start, end)
        block_j: Column indices (start, end)
        eps: Tolerance for determining numerical rank

    Returns:
        rank: Numerical rank of the block
        s_norm: Normalized singular values
    """
    A_block = A[block_i[0]:block_i[1], block_j[0]:block_j[1]]
    u, s, v = torch.svd(A_block)

    # Normalize singular values
    s_norm = s / s[0].item() if s[0].item() > 0 else s

    # Determine numerical rank
    rank = torch.sum(s > eps * s[0]).item()

    return int(rank), s_norm.cpu().numpy().tolist()

def hierarchical_partition(n: int, min_size: int = 16) -> List[List[Tuple[int, int]]]:
    """
    Create a hierarchical partition of the index set [0, n-1].

    Args:
        n: Total size
        min_size: Minimum block size

    Returns:
        List of levels, each level containing a list of (start, end) tuples
    """
    levels = []

    # Start with the full set
    current_level = [(0, n)]
    levels.append(current_level)

    # Subdivide each level until reaching min_size
    while current_level[0][1] - current_level[0][0] > min_size:
        next_level = []
        for start, end in current_level:
            mid = (start + end) // 2
            next_level.append((start, mid))
            next_level.append((mid, end))
        levels.append(next_level)
        current_level = next_level

    return levels

def randomized_svd(
    A: torch.Tensor,
    rank: int,
    n_oversamples: int = 10,
    n_iter: int = 2,
    eps: float = 1e-6,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute randomized SVD for faster low-rank approximation.

    This implementation follows the algorithm from Halko et al. 2009:
    "Finding structure with randomness: Probabilistic algorithms for constructing
    approximate matrix decompositions"

    Args:
        A: Input matrix of shape (m, n)
        rank: Target rank for the approximation
        n_oversamples: Oversampling parameter to improve accuracy (default: 10)
        n_iter: Number of power iterations to enhance accuracy (default: 2)
        eps: Tolerance for determining numerical rank
        device: Computation device (default: same as input tensor)

    Returns:
        U: Left singular vectors
        s: Singular values
        V: Right singular vectors (not transposed)
    """
    if device is None:
        device = A.device

    m, n = A.shape
    rank = min(rank, min(m, n))

    # Oversampled rank
    k = min(rank + n_oversamples, min(m, n))

    # Stage A: Compute an approximate range of A
    # Generate random Gaussian matrix
    Omega = torch.randn(n, k, device=device)

    # Form Y = A * Omega
    Y = A @ Omega

    # Optional: Perform power iterations to increase accuracy for slowly decaying spectra
    if n_iter > 0:
        Q, _ = torch.linalg.qr(Y)
        for _ in range(n_iter):
            Z = A.T @ Q
            Q, _ = torch.linalg.qr(Z)
            Z = A @ Q
            Q, _ = torch.linalg.qr(Z)
        Y = Q
    else:
        Q, _ = torch.linalg.qr(Y)

    # Stage B: Compute SVD on the reduced matrix
    # Form B = Q^T * A
    B = Q.T @ A

    # Compute SVD of the small matrix B
    U_hat, s, Vh = torch.linalg.svd(B, full_matrices=False)

    # Recover the left singular vectors
    U = Q @ U_hat

    # Get the right singular vectors (not transposed)
    V = Vh.T

    # Determine numerical rank based on tolerance
    tol = eps * s[0].item() if s[0].item() > 0 else eps
    numerical_rank = torch.sum(s > tol).item()
    numerical_rank = min(numerical_rank, rank)
    numerical_rank = int(numerical_rank)

    # Truncate to the desired rank
    U = U[:, :numerical_rank]
    s = s[:numerical_rank]
    V = V[:, :numerical_rank]

    return U, s, V
