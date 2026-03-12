"""Unit tests for qsarify.modeling.genetic_algorithm.

Tests the cluster-diversity invariant exhaustively for chromosome
initialization, crossover, and mutation.  Fitness evaluation, tournament
selection, and the full GA loop are verified with smoke tests.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qsarify.modeling.genetic_algorithm import (
    Chromosome,
    crossover,
    decode_chromosome,
    enumerate_subsets,
    evaluate_fitness,
    init_chromosome,
    init_population,
    mutate,
    run_ga_mlr,
    tournament_select,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(7)
N_TRAIN, P_ALL = 50, 12

# cluster_map: 4 clusters, 3 descriptors each
CLUSTER_MAP: dict[int, list[int]] = {
    1: [0, 1, 2],
    2: [3, 4, 5],
    3: [6, 7, 8],
    4: [9, 10, 11],
}

X_TRAIN = RNG.standard_normal((N_TRAIN, P_ALL))
TRUE_COEF = np.array([1.5, -1.0, 2.0])
Y_TRAIN = (
    X_TRAIN[:, 0] * TRUE_COEF[0]
    + X_TRAIN[:, 3] * TRUE_COEF[1]
    + X_TRAIN[:, 6] * TRUE_COEF[2]
    + RNG.standard_normal(N_TRAIN) * 0.3
)


# ---------------------------------------------------------------------------
# Cluster-diversity invariant helper
# ---------------------------------------------------------------------------


def has_unique_clusters(chrom: Chromosome) -> bool:
    cluster_ids = [g[0] for g in chrom]
    return len(cluster_ids) == len(set(cluster_ids))


# ---------------------------------------------------------------------------
# Chromosome initialization
# ---------------------------------------------------------------------------


def test_init_chromosome_length() -> None:
    rng = np.random.default_rng(0)
    chrom = init_chromosome(m=3, cluster_map=CLUSTER_MAP, rng=rng)
    assert len(chrom) == 3


def test_init_chromosome_cluster_diversity_invariant() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        chrom = init_chromosome(m=3, cluster_map=CLUSTER_MAP, rng=rng)
        assert has_unique_clusters(chrom), f"Diversity violated: {chrom}"


def test_init_chromosome_descriptor_in_cluster() -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        chrom = init_chromosome(m=2, cluster_map=CLUSTER_MAP, rng=rng)
        for cid, desc_idx in chrom:
            assert desc_idx in CLUSTER_MAP[cid]


def test_init_population_all_valid() -> None:
    rng = np.random.default_rng(2)
    pop = init_population(m=3, cluster_map=CLUSTER_MAP, pop_size=30, rng=rng)
    assert len(pop) == 30
    for chrom in pop:
        assert has_unique_clusters(chrom)


# ---------------------------------------------------------------------------
# decode_chromosome
# ---------------------------------------------------------------------------


def test_decode_chromosome_returns_descriptor_indices() -> None:
    chrom: Chromosome = [(1, 0), (2, 4), (3, 8)]
    result = decode_chromosome(chrom)
    assert result == [0, 4, 8]


def test_decode_chromosome_empty() -> None:
    assert decode_chromosome([]) == []


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------


def _make_parents(seed: int) -> tuple[Chromosome, Chromosome]:
    rng = np.random.default_rng(seed)
    p1 = init_chromosome(3, CLUSTER_MAP, rng)
    # Ensure different from p1 by using a different seed
    p2 = init_chromosome(3, CLUSTER_MAP, np.random.default_rng(seed + 100))
    return p1, p2


def test_crossover_offspring_length() -> None:
    rng = np.random.default_rng(3)
    p1, p2 = _make_parents(10)
    c1, c2 = crossover(p1, p2, 0.8, 0.6, CLUSTER_MAP, rng)
    assert len(c1) == 3
    assert len(c2) == 3


def test_crossover_diversity_invariant_100_trials() -> None:
    rng = np.random.default_rng(4)
    violations = 0
    for seed in range(100):
        p1, p2 = _make_parents(seed)
        c1, c2 = crossover(p1, p2, 0.8, 0.6, CLUSTER_MAP, rng)
        if not has_unique_clusters(c1) or not has_unique_clusters(c2):
            violations += 1
    assert violations == 0, f"{violations} diversity violations in crossover"


def test_crossover_genes_come_from_cluster_map() -> None:
    rng = np.random.default_rng(5)
    p1, p2 = _make_parents(20)
    c1, c2 = crossover(p1, p2, 0.9, 0.5, CLUSTER_MAP, rng)
    for chrom in (c1, c2):
        for cid, desc_idx in chrom:
            assert cid in CLUSTER_MAP, f"Unknown cluster {cid}"
            assert desc_idx in CLUSTER_MAP[cid], f"{desc_idx} not in cluster {cid}"


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------


def test_mutation_preserves_length() -> None:
    rng = np.random.default_rng(6)
    chrom = init_chromosome(3, CLUSTER_MAP, rng)
    mutated = mutate(chrom, CLUSTER_MAP, mutation_rate=1.0, inter_ratio=0.7, rng=rng)
    assert len(mutated) == 3


def test_mutation_diversity_invariant_100_trials() -> None:
    rng = np.random.default_rng(7)
    violations = 0
    for _ in range(100):
        chrom = init_chromosome(3, CLUSTER_MAP, rng)
        mutated = mutate(chrom, CLUSTER_MAP, mutation_rate=1.0, inter_ratio=0.7, rng=rng)
        if not has_unique_clusters(mutated):
            violations += 1
    assert violations == 0, f"{violations} diversity violations in mutation"


def test_intra_cluster_mutation_keeps_cluster() -> None:
    """Intra-cluster mutation: same cluster ID, possibly different descriptor."""
    rng = np.random.default_rng(8)
    # cluster 2 has 3 descriptors → intra mutation can change descriptor
    chrom: Chromosome = [(1, 0), (2, 3), (3, 6)]
    mutated = mutate(chrom, CLUSTER_MAP, mutation_rate=1.0, inter_ratio=0.0, rng=rng)
    assert has_unique_clusters(mutated)
    cluster_ids_before = {g[0] for g in chrom}
    cluster_ids_after = {g[0] for g in mutated}
    # With inter_ratio=0.0, all mutations are intra: cluster IDs unchanged
    assert cluster_ids_before == cluster_ids_after


def test_inter_cluster_mutation_changes_cluster() -> None:
    """Inter-cluster mutation: different cluster ID selected."""
    # Use a cluster map with more clusters to make inter mutation non-trivial
    big_map: dict[int, list[int]] = {i: [i] for i in range(8)}
    rng = np.random.default_rng(9)
    changes = 0
    for _ in range(50):
        chrom: Chromosome = [(0, 0), (1, 1), (2, 2)]
        mutated = mutate(chrom, big_map, mutation_rate=1.0, inter_ratio=1.0, rng=rng)
        if {g[0] for g in mutated} != {g[0] for g in chrom}:
            changes += 1
    assert changes > 0


# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn", ["q2_loo", "r2_adj", "lof", "rmse_cv"])
def test_fitness_returns_scalar(fn: str) -> None:
    chrom: Chromosome = [(1, 0), (2, 3), (3, 6)]
    f = evaluate_fitness(chrom, X_TRAIN, Y_TRAIN, fitness_fn=fn, quik_delta=None)
    assert isinstance(f, float)


def test_fitness_q2_loo_better_for_true_model() -> None:
    """True descriptor subset should have higher Q²_LOO than a random one."""
    true_chrom: Chromosome = [(1, 0), (2, 3), (3, 6)]
    noise_chrom: Chromosome = [(1, 2), (2, 5), (3, 8)]
    f_true = evaluate_fitness(true_chrom, X_TRAIN, Y_TRAIN, fitness_fn="q2_loo")
    f_noise = evaluate_fitness(noise_chrom, X_TRAIN, Y_TRAIN, fitness_fn="q2_loo")
    assert f_true > f_noise


def test_quik_penalty_applied() -> None:
    """A model that violates QUIK rule should receive penalty fitness.

    QUIK triggers when K_xy - K_xx < delta, i.e. when X descriptors are already
    so collinear that adding y doesn't change the K statistic.  This happens when
    X columns are nearly identical AND y ≈ x_1 (so augmenting with y adds nothing).
    """
    # All three columns nearly identical; y also nearly identical to them.
    # k_xx ≈ 1 (all X columns same), k_xy ≈ 1 (y same as X) → k_xy - k_xx ≈ 0 < delta
    base = np.linspace(0.0, 1.0, 30)
    X_quik = np.column_stack([base, base + 1e-8, base + 2e-8])
    y_quik = base + 1e-9 * np.arange(30)
    chrom: Chromosome = [(1, 0), (2, 1), (3, 2)]
    f = evaluate_fitness(chrom, X_quik, y_quik, fitness_fn="q2_loo", quik_delta=0.05)
    assert f == float("-inf"), f"Expected -inf penalty, got {f}"


# ---------------------------------------------------------------------------
# Tournament selection
# ---------------------------------------------------------------------------


def test_tournament_prefers_higher_fitness() -> None:
    rng = np.random.default_rng(10)
    pop = init_population(m=3, cluster_map=CLUSTER_MAP, pop_size=20, rng=rng)
    fitnesses = list(range(20))  # higher index = higher fitness
    # With large tournament, should almost always pick the best
    selected_indices = [
        tournament_select(pop, fitnesses, tournament_size=10, rng=rng)
        for _ in range(100)
    ]
    # Best chromosome (index 19) should be selected frequently
    best_selected = sum(1 for c in selected_indices if c == pop[19])
    assert best_selected > 30, f"Best selected only {best_selected}/100 times"


# ---------------------------------------------------------------------------
# Exhaustive enumeration
# ---------------------------------------------------------------------------


def test_exhaustive_enum_subset_sizes() -> None:
    """All returned models should have 1 to max_size descriptors."""
    models = enumerate_subsets(
        X_TRAIN, Y_TRAIN, CLUSTER_MAP, max_size=2, quik_delta=None
    )
    for m in models:
        r = m.get_results()
        assert r.n_features is not None
        assert 1 <= r.n_features <= 2


def test_exhaustive_enum_covers_all_pairs() -> None:
    """With max_size=2 and 4 clusters, should cover all 4C2=6 cluster pairs + 4 singles."""
    models = enumerate_subsets(
        X_TRAIN, Y_TRAIN, CLUSTER_MAP, max_size=2, quik_delta=None
    )
    sizes = [m.get_results().n_features for m in models]
    assert 1 in sizes and 2 in sizes
    # At minimum: C(4,1) + C(4,2) = 4 + 6 = 10 models
    # (1 descriptor per cluster × combinations, but clusters have 3 descriptors each
    #  so there can be more than 10 models due to selecting different descriptors)
    assert len(models) >= 10


def test_exhaustive_enum_diversity_invariant() -> None:
    """All enumerated models must satisfy cluster diversity invariant."""
    models = enumerate_subsets(
        X_TRAIN, Y_TRAIN, CLUSTER_MAP, max_size=2, quik_delta=None
    )
    for m in models:
        if m.get_results().cluster_assignments is not None:
            ca = m.get_results().cluster_assignments
            assert ca is not None
            cluster_ids = list(ca.values())
            assert len(cluster_ids) == len(set(cluster_ids)), "Diversity violated!"


# ---------------------------------------------------------------------------
# Full GA loop (smoke test)
# ---------------------------------------------------------------------------


def test_run_ga_mlr_returns_subset_models() -> None:
    from qsarify.modeling.base import SubsetModel

    models = run_ga_mlr(
        X_TRAIN, Y_TRAIN,
        CLUSTER_MAP,
        min_variables=1,
        max_variables=3,
        population_size=20,
        max_generations=5,
        mutation_rate=0.1,
        keep_best=3,
        fitness_function="q2_loo",
        quik_delta=None,
        random_seed=42,
        n_workers=0,  # sequential, no multiprocessing in tests
    )
    assert len(models) > 0
    for m in models:
        assert isinstance(m, SubsetModel)


def test_run_ga_mlr_results_not_none() -> None:
    models = run_ga_mlr(
        X_TRAIN, Y_TRAIN,
        CLUSTER_MAP,
        min_variables=2,
        max_variables=2,
        population_size=15,
        max_generations=4,
        mutation_rate=0.1,
        keep_best=2,
        fitness_function="r2_adj",
        quik_delta=None,
        random_seed=0,
        n_workers=0,
    )
    assert len(models) >= 1
    for m in models:
        r = m.get_results()
        assert r.r_squared is not None
        assert r.q_squared_loo is not None


def test_run_ga_mlr_finds_good_model() -> None:
    """GA should find a model with decent Q²_LOO on signal-heavy data."""
    models = run_ga_mlr(
        X_TRAIN, Y_TRAIN,
        CLUSTER_MAP,
        min_variables=3,
        max_variables=3,
        population_size=30,
        max_generations=20,
        mutation_rate=0.05,
        keep_best=1,
        fitness_function="q2_loo",
        quik_delta=None,
        random_seed=42,
        n_workers=0,
    )
    best = max(models, key=lambda m: m.get_results().q_squared_loo or -1.0)
    assert best.get_results().q_squared_loo is not None
    assert best.get_results().q_squared_loo > 0.5, (
        f"Expected Q²_LOO > 0.5, got {best.get_results().q_squared_loo}"
    )
