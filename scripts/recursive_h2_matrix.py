#!/usr/bin/env python
"""
Implementation of a recursive H² matrix approximation for attention.
Uses a hierarchical tree structure with true recursive partitioning and SVD for low-rank compression.
"""

import torch
import numpy as np
from time import time
from typing import List, Optional, Tuple, Dict, Union, Any

class H2Node:
    """Represents a node in the hierarchical index tree."""
    def __init__(self, start: int, end: int, level: int):
        self.start = start  # Start index (inclusive)
        self.end = end      # End index (exclusive)
        self.size = end - start
        self.level = level
        self.children: List['H2Node'] = [] # Child nodes
        # Center/radius are useful for admissibility
        self.center = (start + end - 1) / 2.0
        self.radius = (end - start) / 2.0

    def is_leaf(self) -> bool:
        return not self.children

    def __repr__(self) -> str:
        return f"Node(L{self.level}, [{self.start}-{self.end-1}], size={self.size})"


def build_h2_tree(start: int, end: int, level: int = 0, nmin: int = 32) -> H2Node:
    """Recursively builds the hierarchical index tree."""
    node = H2Node(start, end, level)

    # Base case: stop recursion if node size is below minimum leaf size
    if node.size <= nmin:
        return node

    # Recursive step: split the node into two children
    mid = start + node.size // 2 # Ensure integer division behaves as expected
    if mid <= start or mid >= end: # Avoid infinite recursion if size is too small or split is bad
         return node # Treat as leaf if split is invalid

    child1 = build_h2_tree(start, mid, level + 1, nmin)
    child2 = build_h2_tree(mid, end, level + 1, nmin)
    node.children = [child1, child2]

    return node


def is_admissible(node_t: H2Node, node_s: H2Node, eta: float = 0.8) -> bool:
    """
    Checks if the interaction between two nodes is admissible (far-field).
    Standard admissibility condition: max(diam(t), diam(s)) <= eta * dist(t, s)
    For 1D indices, diameter = size, distance = distance between centers.
    """
    if node_t.size == 0 or node_s.size == 0:
        return False # Cannot approximate empty blocks

    # Diameter of clusters (in 1D, this is just the size/length)
    diam_t = float(node_t.size)
    diam_s = float(node_s.size)

    # Distance between the centers of the clusters
    # Add small epsilon to avoid division by zero if nodes overlap
    dist_ts = abs(node_t.center - node_s.center) + 1e-8

    # Adjust distance calculation slightly to represent distance between intervals
    # Minimum distance between the intervals [t.start, t.end) and [s.start, s.end)
    min_dist = max(0.0, node_t.start - node_s.end + 1, node_s.start - node_t.end + 1)
    # Use a distance metric robust to adjacent blocks
    # dist_ts = min_dist + (node_t.radius + node_s.radius) # Alt: distance between centers

    # Check the admissibility condition
    is_adm = max(diam_t, diam_s) <= eta * dist_ts

    # Add explicit check: adjacent blocks are usually NOT admissible
    # If the end of one block is the start of the other, they are adjacent.
    if node_t.end == node_s.start or node_s.end == node_t.start:
       is_adm = False
    # Overlapping blocks are never admissible
    if max(node_t.start, node_s.start) < min(node_t.end, node_s.end):
        is_adm = False

    return is_adm


class DenseMatrixBlock:
    def __init__(self, matrix: torch.Tensor):
        self.matrix = matrix # Shape (rows, cols)

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure x is compatible (e.g., 1D vector or 2D matrix cols)
        if x.dim() == 1:
            return torch.matmul(self.matrix, x)
        elif x.dim() == 2:
             # Check if dimensions match for matmul (cols of matrix vs rows of x)
            if self.matrix.shape[1] != x.shape[0]:
                raise ValueError(f"Shape mismatch in Dense matvec: {self.matrix.shape} vs {x.shape}")
            return torch.matmul(self.matrix, x)
        else:
             raise ValueError("Input tensor x must be 1D or 2D")


class LowRankMatrixBlock:
    def __init__(self, U: torch.Tensor, V: torch.Tensor):
        # Stores block B = U @ V.T
        self.U = U # Shape (rows, rank)
        self.V = V # Shape (cols, rank) -> We store V directly

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        # Compute y = U @ (V.T @ x)
        # Ensure x is compatible
        if x.dim() == 1:
             # Check V.T shape vs x shape
             if self.V.shape[0] != x.shape[0]:
                 raise ValueError(f"Shape mismatch in LowRank matvec (V.T @ x): {self.V.T.shape} vs {x.shape}")
             vt_x = torch.matmul(self.V.t(), x)
             return torch.matmul(self.U, vt_x)
        elif x.dim() == 2:
             # Check V.T shape vs x shape
             if self.V.shape[0] != x.shape[0]:
                 raise ValueError(f"Shape mismatch in LowRank matvec (V.T @ x): {self.V.T.shape} vs {x.shape}")
             vt_x = torch.matmul(self.V.t(), x)
             return torch.matmul(self.U, vt_x)
        else:
             raise ValueError("Input tensor x must be 1D or 2D")


MatrixBlock = Union[DenseMatrixBlock, LowRankMatrixBlock] # Type hint


class RecursiveH2Matrix:
    def __init__(self, n: int, nmin: int = 32, rank: int = 10, eta: float = 0.8, eps: float = 1e-6, device: str = 'cpu'):
        self.n = n
        self.nmin = nmin
        self.rank = rank
        self.eta = eta
        self.eps = eps
        self.device = device

        print(f"Building H2 Tree (n={n}, nmin={nmin})...")
        self.root_node = build_h2_tree(0, n, nmin=nmin)
        print("H2 Tree built.")

        # Store matrix blocks, mapping (target_node_id, source_node_id) -> MatrixBlock
        # Using node objects directly as keys might be tricky, use start/end tuples or unique IDs if needed
        # For simplicity here, let's use (target_start, target_end, source_start, source_end) tuple as key
        self.blocks: Dict[Tuple[int, int, int, int], MatrixBlock] = {}
        self.stats = {} # To store compression stats later

    def _get_block_key(self, node_t: H2Node, node_s: H2Node) -> Tuple[int, int, int, int]:
         """Generates a unique key for the block interaction."""
         return (node_t.start, node_t.end, node_s.start, node_s.end)

    def _compute_and_store_dense_block(self, node_t: H2Node, node_s: H2Node, q: torch.Tensor, k: torch.Tensor):
        """Computes and stores a dense matrix block."""
        # Extract relevant Q/K slices
        q_slice = q[node_t.start:node_t.end, :]
        k_slice = k[node_s.start:node_s.end, :]

        # Compute the dense attention block (unnormalized)
        d_model = q.shape[1]
        qk = torch.matmul(q_slice, k_slice.transpose(0, 1)) / torch.sqrt(torch.tensor(d_model, device=self.device))
        A_block_dense = torch.exp(qk)

        key = self._get_block_key(node_t, node_s)
        self.blocks[key] = DenseMatrixBlock(A_block_dense.to(self.device))
        # print(f"  Stored Dense Block for {key}")

    def _compute_and_store_low_rank_block(self, node_t: H2Node, node_s: H2Node, q: torch.Tensor, k: torch.Tensor):
        """Computes and stores a low-rank matrix block via SVD."""
        # Extract relevant Q/K slices
        q_slice = q[node_t.start:node_t.end, :]
        k_slice = k[node_s.start:node_s.end, :]

        # Compute the dense attention block temporarily for SVD
        d_model = q.shape[1]
        qk = torch.matmul(q_slice, k_slice.transpose(0, 1)) / torch.sqrt(torch.tensor(d_model, device=self.device))
        A_block_dense = torch.exp(qk)

        # Perform SVD (Consider randomized SVD for very large blocks)
        try:
            U, S, Vh = torch.linalg.svd(A_block_dense, full_matrices=False)
            V = Vh.T #linalg.svd returns Vh (V conjugate transpose)
        except torch.linalg.LinAlgError:
             print(f"SVD failed for block ({node_t}, {node_s}). Storing as dense.")
             # Fallback to dense if SVD fails
             key = self._get_block_key(node_t, node_s)
             self.blocks[key] = DenseMatrixBlock(A_block_dense.to(self.device))
             return

        # Determine rank for truncation
        if S.numel() == 0 or S[0].item() <= self.eps:
             actual_rank = 0
        else:
             tol = self.eps * S[0].item()
             # Ensure we don't try to find rank beyond available singular values
             max_possible_rank = S.numel()
             # Check against tolerance
             rank_candidates = torch.where(S > tol)[0]
             if rank_candidates.numel() == 0:
                 numerical_rank = 1 # Keep at least rank 1 if possible
             else:
                 numerical_rank = rank_candidates[-1].item() + 1
             actual_rank = min(self.rank, numerical_rank, max_possible_rank)

        # Store low-rank factors U and V (Note: B = U @ diag(S) @ V.T)
        # We store U' = U[:, :k] @ diag(S[:k]) and V' = V[:, :k]
        # So B = U' @ V'.T
        if actual_rank > 0:
            U_k = U[:, :actual_rank] @ torch.diag(S[:actual_rank])
            V_k = V[:, :actual_rank] # V is already V, not Vh
            key = self._get_block_key(node_t, node_s)
            self.blocks[key] = LowRankMatrixBlock(U_k.to(self.device), V_k.to(self.device))
            # print(f"  Stored LowRank Block (rank {actual_rank}) for {key}")
        # else: store nothing or a zero block if rank is 0

    # Recursive build function
    def build(self, q: torch.Tensor, k: torch.Tensor):
        """Builds the H² matrix representation recursively."""
        if q.shape[0] != self.n or k.shape[0] != self.n:
            raise ValueError("Q/K dimensions must match matrix dimension n")
        q = q.to(self.device)
        k = k.to(self.device)
        print("Starting recursive build...")
        start_time = time()
        self._recursive_build(self.root_node, self.root_node, q, k)
        build_time = time() - start_time
        print(f"Recursive build finished in {build_time:.3f} seconds.")
        self._calculate_stats() # Calculate stats after build
        return build_time

    def _recursive_build(self, node_t: H2Node, node_s: H2Node, q: torch.Tensor, k: torch.Tensor):
        """Helper function for recursive build."""
        # Base case: Interaction involves leaf nodes
        is_leaf_interaction = node_t.is_leaf() or node_s.is_leaf()

        # Check admissibility
        is_adm = is_admissible(node_t, node_s, self.eta)

        if is_adm:
            # Admissible (Far-field): Compute and store low-rank approximation
            # print(f"Admissible block: T={node_t}, S={node_s}")
            self._compute_and_store_low_rank_block(node_t, node_s, q, k)
        elif is_leaf_interaction:
             # Inadmissible Leaf (Near-field leaf): Compute and store dense block
             # print(f"Inadmissible Leaf block: T={node_t}, S={node_s}")
             self._compute_and_store_dense_block(node_t, node_s, q, k)
        else:
             # Inadmissible Non-Leaf: Recurse on children
             # print(f"Inadmissible Non-leaf block, recursing: T={node_t}, S={node_s}")
             for child_t in node_t.children:
                 for child_s in node_s.children:
                     self._recursive_build(child_t, child_s, q, k)

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the matrix-vector product y = A @ x using the H² structure."""
        if x.shape[0] != self.n:
            raise ValueError(f"Input vector x dimension {x.shape[0]} must match matrix dimension {self.n}")

        # Ensure x is on the correct device and has the right shape (n, batch_or_feat)
        x = x.to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(1) # Convert to (n, 1)

        y = torch.zeros((self.n, x.shape[1]), device=self.device) # Output accumulator

        # Start the recursive multiplication
        self._recursive_matvec(self.root_node, self.root_node, x, y)

        return y.squeeze(1) if y.shape[1] == 1 else y # Return shape consistent with input

    def _recursive_matvec(self, node_t: H2Node, node_s: H2Node, x: torch.Tensor, y: torch.Tensor):
        """Helper function for recursive matrix-vector product."""

        key = self._get_block_key(node_t, node_s)

        if key in self.blocks:
            # Base case: A block exists for this interaction (dense or low-rank)
            block = self.blocks[key]
            # Extract the relevant slice of x corresponding to the source node s
            x_slice = x[node_s.start:node_s.end, :]
            # Perform the block's matvec
            y_update = block.matvec(x_slice)
            # Add the result to the relevant slice of y corresponding to the target node t
            y[node_t.start:node_t.end, :] += y_update
        elif not (node_t.is_leaf() or node_s.is_leaf()):
            # Recursive case: No block stored directly, must have been inadmissible non-leaf.
            # Recurse into children.
            for child_t in node_t.children:
                for child_s in node_s.children:
                    # No need to pass slices of x/y, indices are handled by the base case
                    self._recursive_matvec(child_t, child_s, x, y)
        # Else: Leaf interaction that was admissible (rank 0?) or inadmissible but somehow not stored?
        # This case shouldn't typically happen if build is correct. Can add a warning.
        # elif node_t.is_leaf() or node_s.is_leaf():
        #    print(f"Warning: Reached leaf node pair without stored block: T={node_t}, S={node_s}")

    def attention_output(self, x: torch.Tensor) -> torch.Tensor:
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

    def _calculate_stats(self):
        """Calculate compression statistics."""
        print("Calculating H2 stats...")
        n_dense = 0
        n_low_rank = 0
        mem_dense = 0
        mem_low_rank = 0
        total_rank = 0

        for key, block in self.blocks.items():
            if isinstance(block, DenseMatrixBlock):
                n_dense += 1
                mem_dense += block.matrix.numel()
            elif isinstance(block, LowRankMatrixBlock):
                n_low_rank += 1
                mem_low_rank += block.U.numel() + block.V.numel()
                total_rank += block.U.shape[1] # Rank is stored in U's second dim

        mem_h2 = mem_dense + mem_low_rank
        mem_full = self.n * self.n
        avg_rank = total_rank / n_low_rank if n_low_rank > 0 else 0

        self.stats = {
            'n_dense_blocks': n_dense,
            'n_low_rank_blocks': n_low_rank,
            'mem_dense': mem_dense,
            'mem_low_rank': mem_low_rank,
            'mem_h2': mem_h2,
            'mem_full': mem_full,
            'compression_ratio': mem_full / mem_h2 if mem_h2 > 0 else float('inf'),
            'average_rank': avg_rank
        }

        print(f"H² statistics:")
        print(f"  Near-field blocks: {n_dense}")
        print(f"  Far-field blocks: {n_low_rank}")
        print(f"  Memory usage: {mem_h2} elements ({self.stats['compression_ratio']:.2f}x compression)")
        print(f"  Average rank: {avg_rank:.2f}")
