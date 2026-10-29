"""Small, reproducible Random Walk with Restart (RWR) helper.

The supplied PathDSP NetPEA pathway matrices already contain RWR-derived
DG-Net, MUT-Net and CNV-Net features. This helper is for re-running the core
diffusion step when a sparse adjacency matrix and seed-node indices are given.
It does not replace NetPEA's pathway permutation and z-score procedure.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def random_walk_with_restart(
    adjacency: sparse.spmatrix | np.ndarray,
    seed_nodes: list[int] | np.ndarray,
    restart_probability: float = 0.3,
    tolerance: float = 1e-10,
    max_iterations: int = 500,
) -> np.ndarray:
    """Return stationary node probabilities for an undirected/weighted graph."""
    if not 0.0 < restart_probability <= 1.0:
        raise ValueError("restart_probability must be in (0, 1].")
    graph = sparse.csr_matrix(adjacency, dtype=np.float64)
    if graph.shape[0] != graph.shape[1]:
        raise ValueError("adjacency must be square")
    n_nodes = graph.shape[0]
    seeds = np.asarray(seed_nodes, dtype=np.int64)
    seeds = seeds[(seeds >= 0) & (seeds < n_nodes)]
    if len(seeds) == 0:
        raise ValueError("at least one valid restart node is required")

    p0 = np.zeros(n_nodes, dtype=np.float64)
    p0[np.unique(seeds)] = 1.0 / len(np.unique(seeds))
    degree = np.asarray(graph.sum(axis=1)).ravel()
    inv_degree = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
    transition = sparse.diags(inv_degree) @ graph
    current = p0.copy()
    for _ in range(max_iterations):
        updated = (1.0 - restart_probability) * (transition.T @ current) + restart_probability * p0
        if np.linalg.norm(updated - current, ord=1) < tolerance:
            current = np.asarray(updated).ravel()
            break
        current = np.asarray(updated).ravel()
    total = current.sum()
    return current / total if total else current

