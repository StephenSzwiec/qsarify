"""Cophenetic clustering for descriptor pre-selection in GA-MLR.

Algorithm
---------
1. Compute Pearson autocorrelation matrix R of X columns (p × p).
2. Convert to distance matrix: D_ij = 1 − |R_ij|.  Highly correlated
   descriptors (positive *or* negative) get small distances.
3. Average-linkage (UPGMA) hierarchical clustering via
   :func:`scipy.cluster.hierarchy.linkage`.
4. Cut dendrogram at ``cut_d`` using
   :func:`scipy.cluster.hierarchy.fcluster` with ``criterion='distance'``.

When ``cut_d`` is ``None`` the threshold is chosen automatically by sweeping
candidate cut points and maximising the cohesion score of the resulting
partition, subject to the constraint that the number of clusters ≥
``min_clusters``.

References
----------
Gramatica, P. et al. (2013). QSARINS: A new software for the development,
analysis, and validation of QSAR MLR models. J. Comput. Chem., 34, 2121–2132.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

__all__ = [
    "ClusterResult",
    "cohesion_score",
    "cophenetic_cluster",
    "normalized_shannon_entropy",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ClusterResult:
    """Output of :func:`cophenetic_cluster`.

    Parameters
    ----------
    cluster_labels : ndarray of shape (p,)
        Integer cluster assignment for each descriptor column, 1-indexed.
    cluster_map : dict[int, list[int]]
        Mapping ``cluster_id → [descriptor_column_indices]``.
    cut_d : float
        The distance threshold used to cut the dendrogram.
    n_clusters : int
        Total number of clusters produced.
    linkage_matrix : ndarray of shape (p-1, 4)
        Scipy linkage matrix for the hierarchical clustering.
    cohesion : dict[int, float]
        Mean absolute pairwise Pearson correlation within each multi-member
        cluster.  Singletons are excluded.  Keys are cluster IDs.
    score : float
        Cohesion score for this partition (see :func:`cohesion_score`).
    """

    cluster_labels: NDArray[np.int32]
    cluster_map: dict[int, list[int]]
    cut_d: float
    n_clusters: int
    linkage_matrix: NDArray[np.float64] = field(repr=False)
    cohesion: dict[int, float] = field(repr=False, default_factory=dict)
    score: float = 0.0


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def normalized_shannon_entropy(labels: NDArray[np.generic]) -> float:
    """Normalised Shannon entropy of a cluster-size distribution.

    .. math::

        H_{norm} = -\\frac{\\sum_k \\frac{n_k}{N} \\log \\frac{n_k}{N}}{\\log C}

    where :math:`n_k` is the size of cluster *k*, :math:`N` is the total
    number of elements, and :math:`C` is the number of distinct clusters.

    .. note::
        Retained for backward compatibility.  The automatic ``cut_d``
        selection now uses :func:`cohesion_score` instead.

    Parameters
    ----------
    labels : array-like of int
        Cluster assignments (any non-negative integers).

    Returns
    -------
    float
        Normalised entropy in [0, 1].  Returns 0 for a single-cluster
        partition.
    """
    _, counts = np.unique(labels, return_counts=True)
    c = len(counts)
    if c <= 1:
        return 0.0
    n = labels.shape[0]
    probs = counts / n
    h = -float(np.sum(probs * np.log(probs)))
    return float(h / np.log(c))


def cohesion_score(
    abs_R: NDArray[np.float64],
    labels: NDArray[np.int32],
) -> float:
    """Score a partition by size-uniform diversity and intra-cluster cohesion.

    .. math::

        \\text{score} = H_{\\text{norm}} \\times
        \\frac{1}{k} \\sum_{k} c_k

    where :math:`H_{\\text{norm}}` is the normalised Shannon entropy of the
    cluster-size distribution (see :func:`normalized_shannon_entropy`),
    :math:`k` is the total number of clusters, and :math:`c_k` is the mean
    absolute pairwise Pearson correlation within cluster *k* (0 for
    singletons, which therefore contribute 0 to the sum).

    Both factors are in [0, 1]:

    - :math:`H_{\\text{norm}}` is **0** for a single-cluster partition
      (degenerate) and **1** when all clusters are equal-sized.  It penalises
      both the one-cluster extreme and ragged size distributions (e.g. a few
      large clusters plus many singletons).
    - The global mean cohesion is **0** when no multi-member clusters exist
      (all-singletons, degenerate) and approaches 1 for perfectly correlated
      groups.  Singletons contribute 0, so partitions heavy with singletons
      are penalised here as well.

    The score is therefore **0** for both degenerate extremes (*k* = 1 and
    all singletons) and is maximised by partitions that are simultaneously
    size-uniform and internally cohesive.

    Parameters
    ----------
    abs_R : ndarray of shape (p, p)
        Element-wise absolute Pearson correlation matrix of descriptors.
    labels : ndarray of shape (p,) of int
        Cluster assignments (1-indexed, as returned by ``fcluster``).

    Returns
    -------
    float
        Score in [0, 1].  Returns 0.0 for degenerate partitions.
    """
    # Normalised Shannon entropy of the size distribution.
    # 0 for k=1 (single cluster), 1 for perfectly equal-sized clusters.
    h = normalized_shannon_entropy(labels)
    if h == 0.0:
        return 0.0

    k_total = len(np.unique(labels))

    # Global mean cohesion: singletons contribute 0, divided by k_total.
    cohesion_dict = _compute_cohesion_dict(abs_R, labels)
    if not cohesion_dict:
        return 0.0
    mean_cohesion = float(sum(cohesion_dict.values()) / k_total)

    return float(h * mean_cohesion)


# ---------------------------------------------------------------------------
# Core clustering function
# ---------------------------------------------------------------------------


def cophenetic_cluster(
    X: NDArray[np.float64],
    cut_d: float | None = None,
    min_clusters: int = 1,
) -> ClusterResult:
    """Cluster descriptor columns by Pearson-correlation-based distance.

    Parameters
    ----------
    X : ndarray of shape (n_samples, p)
        Descriptor matrix.  Columns are clustered, not rows.
    cut_d : float or None, optional
        Distance threshold for cutting the dendrogram.  When ``None`` the
        threshold is selected automatically to maximise :func:`cohesion_score`
        subject to ``n_clusters ≥ min_clusters``.
    min_clusters : int, optional
        Minimum acceptable number of clusters when ``cut_d`` is ``None``
        (automatic selection).  Default 1.

    Returns
    -------
    ClusterResult
        Cluster assignments, cohesion values, score, and associated metadata.

    Notes
    -----
    Distance matrix: ``D_ij = 1 − |R_ij|`` where ``R`` is the Pearson
    correlation matrix.  Values range from 0 (perfect correlation /
    anti-correlation) to 1 (no linear relationship).
    """
    X_arr = np.asarray(X, dtype=np.float64)
    _, p = X_arr.shape

    if p == 1:
        labels = np.array([1], dtype=np.int32)
        cluster_map: dict[int, list[int]] = {1: [0]}
        chosen_cut = cut_d if cut_d is not None else 1.0
        dummy_z: NDArray[np.float64] = np.empty((0, 4), dtype=np.float64)
        return ClusterResult(
            cluster_labels=labels,
            cluster_map=cluster_map,
            cut_d=chosen_cut,
            n_clusters=1,
            linkage_matrix=dummy_z,
            cohesion={},
            score=0.0,
        )

    # Step 1: Pearson correlation matrix of columns
    R: NDArray[np.float64] = np.asarray(
        np.corrcoef(X_arr, rowvar=False), dtype=np.float64
    )

    # Absolute correlation — reused for distance matrix and cohesion scoring
    abs_R: NDArray[np.float64] = np.abs(R)

    # Step 2: Distance matrix D_ij = 1 − |R_ij|; clip to [0, 1] for safety
    D: NDArray[np.float64] = np.clip(1.0 - abs_R, 0.0, 1.0)
    np.fill_diagonal(D, 0.0)

    # Step 3: Average-linkage hierarchical clustering
    D_condensed = squareform(D, checks=False)
    Z: NDArray[np.float64] = linkage(D_condensed, method="average")

    # Step 4: Cut dendrogram (auto-select or use provided value)
    if cut_d is None:
        cut_d = _auto_cut_d(Z, abs_R, p, min_clusters)

    raw_labels: NDArray[np.int32] = fcluster(Z, cut_d, criterion="distance").astype(
        np.int32
    )

    cluster_map = _build_cluster_map(raw_labels, p)
    n_clusters = len(cluster_map)

    cohesion_dict = _compute_cohesion_dict(abs_R, raw_labels)
    score = cohesion_score(abs_R, raw_labels)

    return ClusterResult(
        cluster_labels=raw_labels,
        cluster_map=cluster_map,
        cut_d=float(cut_d),
        n_clusters=n_clusters,
        linkage_matrix=Z,
        cohesion=cohesion_dict,
        score=score,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_cohesion_dict(
    abs_R: NDArray[np.float64],
    labels: NDArray[np.int32],
) -> dict[int, float]:
    """Mean absolute pairwise correlation per multi-member cluster.

    Parameters
    ----------
    abs_R : ndarray of shape (p, p)
        Element-wise absolute Pearson correlation matrix.
    labels : ndarray of shape (p,) of int
        Cluster assignments.

    Returns
    -------
    dict[int, float]
        ``cluster_id → mean |r|`` for every cluster with ≥ 2 members.
        Singleton clusters are omitted.
    """
    unique_labels = np.unique(labels)
    cohesion: dict[int, float] = {}
    for cid in unique_labels:
        members = np.where(labels == cid)[0]
        if len(members) < 2:
            continue
        sub = abs_R[np.ix_(members, members)]
        n = len(members)
        triu = np.triu_indices(n, k=1)
        cohesion[int(cid)] = float(np.mean(sub[triu]))
    return cohesion


def _auto_cut_d(
    Z: NDArray[np.float64],
    abs_R: NDArray[np.float64],
    p: int,
    min_clusters: int,
) -> float:
    """Select the cut distance that maximises :func:`cohesion_score`.

    Sweeps candidate cut values equal to the unique merge distances in *Z*
    (plus a small epsilon above each).  Returns the cut that maximises the
    cohesion score among partitions with ``n_clusters ≥ min_clusters``.

    Parameters
    ----------
    Z : ndarray of shape (p-1, 4)
        Scipy linkage matrix.
    abs_R : ndarray of shape (p, p)
        Element-wise absolute Pearson correlation matrix.
    p : int
        Number of descriptors.
    min_clusters : int
        Minimum acceptable number of clusters.

    Returns
    -------
    float
        Chosen cut distance.
    """
    merge_dists: NDArray[np.float64] = np.unique(Z[:, 2])

    eps = 1e-9
    candidates = np.concatenate([merge_dists + eps, [merge_dists[-1] + 1.0]])

    best_cut = float(candidates[0])
    best_score = -np.inf

    for cut in candidates:
        labels: NDArray[np.int32] = fcluster(
            Z, float(cut), criterion="distance"
        ).astype(np.int32)
        n_k = len(np.unique(labels))
        if n_k < min_clusters:
            continue
        s = cohesion_score(abs_R, labels)
        if s > best_score:
            best_score = s
            best_cut = float(cut)

    return best_cut


def _build_cluster_map(labels: NDArray[np.int32], p: int) -> dict[int, list[int]]:
    """Build cluster_id → [descriptor_indices] mapping.

    Parameters
    ----------
    labels : ndarray of shape (p,)
        1-indexed cluster assignments from ``fcluster``.
    p : int
        Number of descriptors.

    Returns
    -------
    dict[int, list[int]]
        Cluster map.
    """
    cluster_map: dict[int, list[int]] = {}
    for desc_idx in range(p):
        cid = int(labels[desc_idx])
        cluster_map.setdefault(cid, []).append(desc_idx)
    return cluster_map
