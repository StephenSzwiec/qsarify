"""Unit tests for qsarify.modeling.clustering."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qsarify.modeling.clustering import (
    ClusterResult,
    cohesion_score,
    cophenetic_cluster,
    normalized_shannon_entropy,
)


RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# normalized_shannon_entropy  (kept for backward-compat; not the auto metric)
# ---------------------------------------------------------------------------


def test_entropy_single_cluster_is_zero() -> None:
    labels = np.array([1, 1, 1, 1])
    assert_allclose(normalized_shannon_entropy(labels), 0.0, atol=1e-12)


def test_entropy_uniform_is_one() -> None:
    """Equal-sized clusters → H_norm = 1."""
    labels = np.array([1, 2, 3, 4])  # 4 clusters, 1 sample each
    assert_allclose(normalized_shannon_entropy(labels), 1.0, atol=1e-12)


def test_entropy_between_zero_and_one() -> None:
    labels = np.array([1, 1, 1, 2, 2, 3])
    h = normalized_shannon_entropy(labels)
    assert 0.0 <= h <= 1.0


def test_entropy_decreases_as_partition_degenerates() -> None:
    equal = np.array([1, 2, 3, 4, 5, 6])
    unequal = np.array([1, 1, 1, 2, 3, 4])
    assert normalized_shannon_entropy(equal) >= normalized_shannon_entropy(unequal)


# ---------------------------------------------------------------------------
# cohesion_score
# ---------------------------------------------------------------------------


def test_cohesion_score_all_singletons() -> None:
    """All singletons → no multi-member clusters → score = 0."""
    p = 4
    abs_R = np.eye(p, dtype=np.float64)
    labels = np.array([1, 2, 3, 4], dtype=np.int32)
    assert cohesion_score(abs_R, labels) == 0.0


def test_cohesion_score_single_cluster_is_zero() -> None:
    """Single cluster → H_norm = 0 → score = 0 (degenerate case penalised)."""
    p = 4
    abs_R = np.ones((p, p), dtype=np.float64)
    labels = np.ones(p, dtype=np.int32)
    assert cohesion_score(abs_R, labels) == 0.0


def test_cohesion_score_two_equal_clusters() -> None:
    """Two equal-sized clusters with cohesion 0.8 → H_norm=1 → score = 0.8."""
    abs_R = np.array(
        [
            [1.0, 0.8, 0.1, 0.1],
            [0.8, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.8],
            [0.1, 0.1, 0.8, 1.0],
        ],
        dtype=np.float64,
    )
    labels = np.array([1, 1, 2, 2], dtype=np.int32)
    # Equal sizes [2, 2] → H_norm = 1.0; mean_cohesion = 0.8 → score = 0.8
    assert_allclose(cohesion_score(abs_R, labels), 0.8, atol=1e-12)


def test_cohesion_score_penalizes_unequal_sizes() -> None:
    """Unequal cluster sizes → lower H_norm → lower score than equal sizes."""
    abs_R = np.array(
        [
            [1.0, 0.8, 0.1, 0.1],
            [0.8, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.8],
            [0.1, 0.1, 0.8, 1.0],
        ],
        dtype=np.float64,
    )
    equal_labels   = np.array([1, 1, 2, 2], dtype=np.int32)  # sizes [2, 2] → H_norm = 1
    unequal_labels = np.array([1, 1, 1, 2], dtype=np.int32)  # sizes [3, 1] → H_norm < 1
    assert cohesion_score(abs_R, equal_labels) > cohesion_score(abs_R, unequal_labels)


def test_cohesion_score_penalizes_degenerate_cases() -> None:
    """Both k=1 and all-singletons score 0; a good partition scores > 0."""
    abs_R = np.array(
        [
            [1.0, 0.8, 0.8, 0.8],
            [0.8, 1.0, 0.8, 0.8],
            [0.8, 0.8, 1.0, 0.8],
            [0.8, 0.8, 0.8, 1.0],
        ],
        dtype=np.float64,
    )
    all_one        = np.ones(4, dtype=np.int32)               # k=1  → H_norm = 0
    two_pairs      = np.array([1, 1, 2, 2], dtype=np.int32)  # good partition
    all_singletons = np.array([1, 2, 3, 4], dtype=np.int32)  # k=4, all size-1 → mean_cohesion = 0

    assert cohesion_score(abs_R, all_one) == 0.0
    assert cohesion_score(abs_R, all_singletons) == 0.0
    assert cohesion_score(abs_R, two_pairs) > 0.0


def test_cohesion_score_in_range() -> None:
    """Score is always in [0, 1]."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((30, 6))
    abs_R = np.abs(np.corrcoef(X, rowvar=False))
    labels = np.array([1, 1, 2, 2, 3, 3], dtype=np.int32)
    s = cohesion_score(abs_R, labels)
    assert 0.0 <= s <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# cophenetic_cluster — basic structural tests
# ---------------------------------------------------------------------------


def _independent_X(n: int = 60, p: int = 8, seed: int = 0) -> np.ndarray:
    """p independent standard-normal descriptors."""
    return np.random.default_rng(seed).standard_normal((n, p))


def _correlated_X(n: int = 60, p: int = 4, seed: int = 1) -> np.ndarray:
    """All p descriptors are highly correlated."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(n)
    noise = rng.standard_normal((n, p)) * 0.01
    return base[:, None] + noise


def test_returns_cluster_result() -> None:
    X = _independent_X()
    result = cophenetic_cluster(X, cut_d=0.5)
    assert isinstance(result, ClusterResult)


def test_cluster_map_covers_all_descriptors() -> None:
    p = 8
    X = _independent_X(p=p)
    result = cophenetic_cluster(X, cut_d=0.5)
    all_desc = set()
    for descs in result.cluster_map.values():
        all_desc.update(descs)
    assert all_desc == set(range(p))


def test_cluster_labels_length_matches_p() -> None:
    p = 8
    X = _independent_X(p=p)
    result = cophenetic_cluster(X, cut_d=0.5)
    assert len(result.cluster_labels) == p


def test_n_clusters_matches_cluster_map() -> None:
    X = _independent_X()
    result = cophenetic_cluster(X, cut_d=0.3)
    assert result.n_clusters == len(result.cluster_map)


def test_highly_correlated_form_fewer_clusters() -> None:
    """Nearly identical columns should collapse to a small number of clusters."""
    X = _correlated_X(p=6)
    result_tight = cophenetic_cluster(X, cut_d=0.05)
    result_loose = cophenetic_cluster(X, cut_d=0.5)
    assert result_loose.n_clusters <= result_tight.n_clusters


def test_large_cut_d_gives_single_cluster() -> None:
    X = _independent_X()
    result = cophenetic_cluster(X, cut_d=2.0)
    assert result.n_clusters == 1


def test_small_cut_d_gives_many_clusters() -> None:
    p = 8
    X = _independent_X(p=p)
    result = cophenetic_cluster(X, cut_d=1e-6)
    assert result.n_clusters == p


def test_single_descriptor_edge_case() -> None:
    X = np.random.default_rng(5).standard_normal((20, 1))
    result = cophenetic_cluster(X, cut_d=0.5)
    assert result.n_clusters == 1
    assert 0 in result.cluster_map[list(result.cluster_map.keys())[0]]


def test_cut_d_stored_in_result() -> None:
    X = _independent_X()
    result = cophenetic_cluster(X, cut_d=0.4)
    assert_allclose(result.cut_d, 0.4)


# ---------------------------------------------------------------------------
# ClusterResult cohesion and score fields
# ---------------------------------------------------------------------------


def test_cluster_result_has_cohesion_field() -> None:
    X = _correlated_X(p=6)
    result = cophenetic_cluster(X, cut_d=0.3)
    assert hasattr(result, "cohesion")
    assert isinstance(result.cohesion, dict)


def test_cluster_result_has_score_field() -> None:
    X = _correlated_X(p=6)
    result = cophenetic_cluster(X, cut_d=0.3)
    assert hasattr(result, "score")
    assert isinstance(result.score, float)


def test_cohesion_only_for_multi_member_clusters() -> None:
    """cohesion dict must only contain clusters with ≥ 2 members."""
    p = 8
    X = _independent_X(p=p)
    result = cophenetic_cluster(X, cut_d=1e-6)  # all singletons
    assert len(result.cohesion) == 0


def test_cohesion_values_in_unit_range() -> None:
    X = _correlated_X(p=6)
    result = cophenetic_cluster(X, cut_d=0.3)
    for v in result.cohesion.values():
        assert 0.0 <= v <= 1.0 + 1e-9


def test_score_in_unit_range() -> None:
    X = _independent_X(p=8)
    result = cophenetic_cluster(X)
    assert 0.0 <= result.score <= 1.0 + 1e-9


def test_single_descriptor_cohesion_empty() -> None:
    X = np.random.default_rng(5).standard_normal((20, 1))
    result = cophenetic_cluster(X, cut_d=0.5)
    assert result.cohesion == {}
    assert result.score == 0.0


# ---------------------------------------------------------------------------
# Automatic cut_d selection
# ---------------------------------------------------------------------------


def test_auto_cut_d_returns_cluster_result() -> None:
    X = _independent_X()
    result = cophenetic_cluster(X)
    assert isinstance(result, ClusterResult)
    assert result.n_clusters >= 1


def test_auto_cut_d_respects_min_clusters() -> None:
    X = _independent_X(p=10)
    min_k = 3
    result = cophenetic_cluster(X, min_clusters=min_k)
    assert result.n_clusters >= min_k


def test_auto_cut_d_value_is_positive() -> None:
    X = _independent_X()
    result = cophenetic_cluster(X)
    assert result.cut_d > 0.0


def test_auto_cut_d_score_beats_all_singletons() -> None:
    """Auto-selected cut should score no worse than all-singletons (score=0)."""
    X = _correlated_X(n=60, p=6)
    result_auto = cophenetic_cluster(X)
    result_singletons = cophenetic_cluster(X, cut_d=1e-6)  # all singletons → score=0
    assert result_auto.score >= result_singletons.score


def test_auto_cut_d_nonzero_score_on_correlated_data() -> None:
    """Correlated data should yield at least one multi-member cluster."""
    X = _correlated_X(n=60, p=6)
    result = cophenetic_cluster(X)
    assert result.score > 0.0
