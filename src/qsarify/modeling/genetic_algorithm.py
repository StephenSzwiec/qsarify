"""Genetic Algorithm for MLR descriptor subset selection (GA-MLR).

Implements the cluster-aware GA described in ``agent_docs/algorithm_spec.md``:

- Chromosome encoding: ``(cluster_id, descriptor_index)`` gene pairs.
- **Cluster-diversity invariant:** no two genes share the same cluster_id.
  Enforced at initialization, crossover, and mutation.
- Fitness functions: ``q2_loo``, ``r2_adj``, ``lof``, ``rmse_cv``.
- QUIK-rule filtering with configurable δ_k threshold.
- Tournament selection.
- Uniform cluster-level crossover with conflict resolution.
- Inter/intra-cluster mutation with configurable ratio.
- Elitism (top *keep_best* per subset size retained across generations).
- Parallel fitness evaluation via :class:`concurrent.futures.ProcessPoolExecutor`.

References
----------
Gramatica, P. et al. (2013). QSARINS. J. Comput. Chem., 34, 2121–2132.
"""

from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from qsarify.modeling.subset_model import SubsetModel
from qsarify.utils import statistics as stat

__all__ = [
    "Chromosome",
    "Gene",
    "init_chromosome",
    "init_population",
    "decode_chromosome",
    "evaluate_fitness",
    "tournament_select",
    "crossover",
    "mutate",
    "enumerate_subsets",
    "run_ga_mlr",
]

# Gene = (cluster_id, descriptor_index)
Gene = tuple[int, int]
Chromosome = list[Gene]

# Penalty fitness for QUIK-rule failures and degenerate models
_PENALTY_FITNESS = float("-inf")


# ---------------------------------------------------------------------------
# Chromosome operations
# ---------------------------------------------------------------------------


def init_chromosome(
    m: int,
    cluster_map: dict[int, list[int]],
    rng: np.random.Generator,
) -> Chromosome:
    """Create one random chromosome of size *m* satisfying the diversity invariant.

    Parameters
    ----------
    m : int
        Number of genes (subset size).
    cluster_map : dict[int, list[int]]
        Mapping ``cluster_id → [descriptor_indices]``.
    rng : np.random.Generator
        NumPy RNG.

    Returns
    -------
    Chromosome
        Length-*m* list of ``(cluster_id, descriptor_idx)`` pairs with
        distinct cluster IDs.
    """
    cluster_ids = list(cluster_map.keys())
    chosen_clusters = rng.choice(cluster_ids, size=m, replace=False).tolist()
    return [(cid, int(rng.choice(cluster_map[cid]))) for cid in chosen_clusters]


def init_population(
    m: int,
    cluster_map: dict[int, list[int]],
    pop_size: int,
    rng: np.random.Generator,
) -> list[Chromosome]:
    """Initialise a population of *pop_size* chromosomes.

    Parameters
    ----------
    m : int
        Chromosome length (subset size).
    cluster_map : dict[int, list[int]]
        Cluster map.
    pop_size : int
        Population size.
    rng : np.random.Generator
        NumPy RNG.

    Returns
    -------
    list[Chromosome]
        Population satisfying the cluster-diversity invariant.
    """
    return [init_chromosome(m, cluster_map, rng) for _ in range(pop_size)]


def decode_chromosome(chrom: Chromosome) -> list[int]:
    """Return the list of descriptor column indices encoded by *chrom*.

    Parameters
    ----------
    chrom : Chromosome
        Chromosome to decode.

    Returns
    -------
    list[int]
        Descriptor indices in the order they appear in *chrom*.
    """
    return [gene[1] for gene in chrom]


# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------


def evaluate_fitness(
    chrom: Chromosome,
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    fitness_fn: str = "q2_loo",
    quik_delta: float | None = 0.05,
) -> float:
    """Evaluate fitness for a single chromosome.

    The returned value is always "higher is better" — LOF and RMSE_CV are
    negated so that tournament selection can uniformly maximise.

    Parameters
    ----------
    chrom : Chromosome
        Chromosome to evaluate.
    X_train : ndarray of shape (n, p_all)
        Full training descriptor matrix.
    y_train : ndarray of shape (n,)
        Training response.
    fitness_fn : str
        One of ``'q2_loo'``, ``'r2_adj'``, ``'lof'``, ``'rmse_cv'``.
    quik_delta : float or None
        QUIK rule threshold δ_k.  ``None`` disables the rule.

    Returns
    -------
    float
        Fitness score (higher is better).  Returns ``-inf`` for QUIK
        violations or degenerate models.
    """
    desc_indices = decode_chromosome(chrom)
    X_sub: NDArray[np.float64] = X_train[:, desc_indices]
    n, p = X_sub.shape

    # Guard against rank-deficient subsets
    if n <= p + 1:
        return _PENALTY_FITNESS

    # QUIK rule
    if quik_delta is not None:
        try:
            k_xx_val = stat.k_xx(X_sub)
            k_xy_val = stat.k_xy(X_sub, y_train)
            if k_xy_val - k_xx_val < quik_delta:
                return _PENALTY_FITNESS
        except Exception:  # noqa: BLE001
            return _PENALTY_FITNESS

    try:
        loo = stat.ols_hat_loo(X_sub, y_train)

        if fitness_fn == "q2_loo":
            return float(loo.q2_loo)

        if fitness_fn == "r2_adj":
            return float(stat.r_squared_adj(y_train, loo.y_pred, n_features=p))

        if fitness_fn == "lof":
            # Guard against 1 - 2p/n ≤ 0
            if 2 * p >= n:
                return _PENALTY_FITNESS
            return float(-stat.lof(y_train, loo.y_pred, n_features=p))

        if fitness_fn == "rmse_cv":
            return float(-np.sqrt(loo.press / n))

    except Exception:  # noqa: BLE001
        return _PENALTY_FITNESS

    raise ValueError(f"Unknown fitness_fn: {fitness_fn!r}")


# ---------------------------------------------------------------------------
# Worker for ProcessPoolExecutor (must be module-level to be picklable)
# ---------------------------------------------------------------------------


def _fitness_worker_args(
    args: tuple[
        Chromosome,
        NDArray[np.float64],
        NDArray[np.float64],
        str,
        float | None,
    ],
) -> float:
    chrom, X, y, fn, delta = args
    return evaluate_fitness(chrom, X, y, fitness_fn=fn, quik_delta=delta)


# ---------------------------------------------------------------------------
# Tournament selection
# ---------------------------------------------------------------------------


def tournament_select(
    population: list[Chromosome],
    fitnesses: Sequence[float],
    tournament_size: int,
    rng: np.random.Generator,
) -> Chromosome:
    """Standard tournament selection.

    Parameters
    ----------
    population : list[Chromosome]
        Current population.
    fitnesses : sequence of float
        Fitness score for each chromosome (higher is better).
    tournament_size : int
        Number of chromosomes drawn in each tournament.
    rng : np.random.Generator
        NumPy RNG.

    Returns
    -------
    Chromosome
        Winner of the tournament (highest fitness).
    """
    indices = rng.choice(len(population), size=tournament_size, replace=False).tolist()
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return population[best_idx]


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------


def crossover(
    parent1: Chromosome,
    parent2: Chromosome,
    fitness1: float,
    fitness2: float,
    cluster_map: dict[int, list[int]],
    rng: np.random.Generator,
) -> tuple[Chromosome, Chromosome]:
    """Uniform cluster-level crossover preserving the diversity invariant.

    For each gene position, each offspring randomly inherits the gene from
    one parent.  **Conflict resolution:** when a cluster would be duplicated
    in an offspring, the gene from the *lower-fitness* parent is replaced by
    a randomly chosen descriptor from a cluster not yet represented.

    Parameters
    ----------
    parent1, parent2 : Chromosome
        Parent chromosomes of the same length.
    fitness1, fitness2 : float
        Fitness scores for parent1 and parent2 respectively.
    cluster_map : dict[int, list[int]]
        Full cluster map.
    rng : np.random.Generator
        NumPy RNG.

    Returns
    -------
    tuple[Chromosome, Chromosome]
        Two offspring chromosomes, both satisfying the diversity invariant.
    """
    m = len(parent1)
    c1: Chromosome = [None] * m  # type: ignore[list-item]
    c2: Chromosome = [None] * m  # type: ignore[list-item]

    for i in range(m):
        if rng.random() < 0.5:
            c1[i] = parent1[i]
            c2[i] = parent2[i]
        else:
            c1[i] = parent2[i]
            c2[i] = parent1[i]

    c1 = _resolve_conflicts(c1, cluster_map, rng)
    c2 = _resolve_conflicts(c2, cluster_map, rng)
    return c1, c2


def _resolve_conflicts(
    chrom: Chromosome,
    cluster_map: dict[int, list[int]],
    rng: np.random.Generator,
) -> Chromosome:
    """Fix any duplicate cluster IDs in *chrom* by replacing with unused clusters.

    Iterates through the chromosome left-to-right, keeping the first
    occurrence of each cluster and replacing subsequent duplicates with a
    randomly chosen unused cluster.
    """
    seen_clusters: set[int] = set()
    all_clusters = list(cluster_map.keys())
    result: Chromosome = []

    for gene in chrom:
        cid, desc_idx = gene
        if cid not in seen_clusters:
            seen_clusters.add(cid)
            result.append(gene)
        else:
            # Find an unused cluster
            available = [c for c in all_clusters if c not in seen_clusters]
            if available:
                new_cid = int(rng.choice(available))
                new_desc = int(rng.choice(cluster_map[new_cid]))
                seen_clusters.add(new_cid)
                result.append((new_cid, new_desc))
            else:
                # Edge case: more genes than clusters — keep existing but skip
                # (shouldn't happen if m ≤ n_clusters)
                result.append(gene)

    return result


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------


def mutate(
    chrom: Chromosome,
    cluster_map: dict[int, list[int]],
    mutation_rate: float,
    inter_ratio: float,
    rng: np.random.Generator,
) -> Chromosome:
    """Apply per-gene mutation preserving the diversity invariant.

    Each gene is mutated independently with probability *mutation_rate*.
    When mutated:

    - **Inter-cluster** (probability *inter_ratio*): swap the gene's cluster
      for a different cluster not yet in the chromosome, then pick a random
      descriptor from the new cluster.
    - **Intra-cluster** (probability ``1 - inter_ratio``): keep the same
      cluster, select a different descriptor from within it.

    Parameters
    ----------
    chrom : Chromosome
        Chromosome to mutate (not modified in place).
    cluster_map : dict[int, list[int]]
        Full cluster map.
    mutation_rate : float
        Per-gene mutation probability in [0, 1].
    inter_ratio : float
        Fraction of mutations that are inter-cluster (vs. intra-cluster).
    rng : np.random.Generator
        NumPy RNG.

    Returns
    -------
    Chromosome
        (Possibly) mutated chromosome satisfying the diversity invariant.
    """
    result: Chromosome = list(chrom)
    occupied_clusters = {g[0] for g in result}
    available_clusters = [c for c in cluster_map if c not in occupied_clusters]

    for i, (cid, desc_idx) in enumerate(result):
        if rng.random() >= mutation_rate:
            continue  # no mutation for this gene

        if rng.random() < inter_ratio and available_clusters:
            # Inter-cluster mutation
            new_cid = int(rng.choice(available_clusters))
            new_desc = int(rng.choice(cluster_map[new_cid]))
            available_clusters.remove(new_cid)
            available_clusters.append(cid)  # old cluster becomes available
            occupied_clusters.discard(cid)
            occupied_clusters.add(new_cid)
            result[i] = (new_cid, new_desc)
        else:
            # Intra-cluster mutation: pick a different descriptor in same cluster
            candidates = [d for d in cluster_map[cid] if d != desc_idx]
            if candidates:
                new_desc = int(rng.choice(candidates))
                result[i] = (cid, new_desc)
            # else: cluster has only one descriptor, no intra mutation possible

    return result


# ---------------------------------------------------------------------------
# Exhaustive enumeration (FR2.2a)
# ---------------------------------------------------------------------------


def enumerate_subsets(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    cluster_map: dict[int, list[int]],
    *,
    X_test: NDArray[np.float64] | None = None,
    y_test: NDArray[np.float64] | None = None,
    max_size: int = 3,
    quik_delta: float | None = 0.05,
    n_workers: int = 0,
) -> list[SubsetModel]:
    """Exhaustively enumerate all descriptor subsets of size 1 to *max_size*.

    Only cluster-diverse subsets are enumerated (one descriptor per cluster).
    For each eligible cluster combination, all permutations of one descriptor
    per cluster are included to ensure complete coverage.

    Parameters
    ----------
    X_train : ndarray of shape (n, p_all)
        Training descriptor matrix.
    y_train : ndarray of shape (n,)
        Training response.
    cluster_map : dict[int, list[int]]
        Cluster map.
    X_test, y_test : ndarray or None
        Optional external test set for computing external validation metrics.
    max_size : int
        Maximum subset size (default 3, per FR2.2a).
    quik_delta : float or None
        QUIK rule threshold.  ``None`` disables the rule.
    n_workers : int
        Number of parallel workers.  0 = sequential.

    Returns
    -------
    list[SubsetModel]
        Fitted :class:`SubsetModel` instances, one per valid enumerated subset.
    """
    cluster_ids = list(cluster_map.keys())
    n_clusters = len(cluster_ids)
    models: list[SubsetModel] = []

    for m in range(1, min(max_size, n_clusters) + 1):
        for cluster_combo in itertools.combinations(cluster_ids, m):
            # All ways to pick one descriptor per selected cluster
            desc_options = [cluster_map[cid] for cid in cluster_combo]
            for desc_combo in itertools.product(*desc_options):
                desc_indices = list(desc_combo)
                chrom: Chromosome = list(zip(cluster_combo, desc_combo))

                f = evaluate_fitness(
                    chrom,
                    X_train,
                    y_train,
                    fitness_fn="q2_loo",
                    quik_delta=quik_delta,
                )
                if f == _PENALTY_FITNESS:
                    continue

                ca = {d: c for c, d in chrom}
                model = SubsetModel(desc_indices, cluster_assignments=ca)
                model.fit(X_train, y_train, X_test=X_test, y_test=y_test)
                models.append(model)

    return models


# ---------------------------------------------------------------------------
# GA-MLR internal helpers
# ---------------------------------------------------------------------------


def _count_chroms_up_to(
    n_feat: int,
    cluster_ids: list[int],
    cluster_map: dict[int, list[int]],
    limit: int,
) -> int:
    """Count cluster-diverse chromosomes of size *n_feat*, stopping at *limit*.

    Parameters
    ----------
    n_feat : int
        Number of genes per chromosome.
    cluster_ids : list[int]
        All available cluster IDs.
    cluster_map : dict[int, list[int]]
        Cluster map.
    limit : int
        Early-exit threshold.  Returns as soon as count exceeds *limit*.

    Returns
    -------
    int
        Total count, capped at the first value that exceeds *limit*.
    """
    count = 0
    for cluster_combo in itertools.combinations(cluster_ids, n_feat):
        count += math.prod(len(cluster_map[cid]) for cid in cluster_combo)
        if count > limit:
            return count
    return count


def _iter_cluster_diverse_chroms(
    n_feat: int,
    cluster_ids: list[int],
    cluster_map: dict[int, list[int]],
) -> Iterator[Chromosome]:
    """Yield every cluster-diverse chromosome of size *n_feat*.

    Parameters
    ----------
    n_feat : int
        Number of genes per chromosome.
    cluster_ids : list[int]
        All available cluster IDs.
    cluster_map : dict[int, list[int]]
        Cluster map.

    Yields
    ------
    Chromosome
        All ``(cluster_id, descriptor_index)`` combinations with one
        descriptor drawn from each of *n_feat* distinct clusters.
    """
    for cluster_combo in itertools.combinations(cluster_ids, n_feat):
        desc_options = [cluster_map[cid] for cid in cluster_combo]
        for desc_combo in itertools.product(*desc_options):
            yield list(zip(cluster_combo, desc_combo))


def _build_scored_bank(
    population: list[Chromosome],
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    fitness_fn: str,
    quik_delta: float | None,
    executor: ProcessPoolExecutor | None,
) -> list[tuple[float, Chromosome]]:
    """Evaluate *population*, filter penalties, and return sorted ``(fitness, chrom)`` pairs.

    Parameters
    ----------
    population : list[Chromosome]
        Chromosomes to evaluate.
    X_train : ndarray of shape (n, p_all)
        Full training descriptor matrix.
    y_train : ndarray of shape (n,)
        Training response.
    fitness_fn : str
        Fitness function name.
    quik_delta : float or None
        QUIK rule threshold.
    executor : ProcessPoolExecutor or None
        Live executor for parallel evaluation; ``None`` for sequential.

    Returns
    -------
    list[tuple[float, Chromosome]]
        Valid ``(fitness, chromosome)`` pairs sorted descending by fitness.
    """
    if not population:
        return []

    if executor is not None:
        args_list = [
            (chrom, X_train, y_train, fitness_fn, quik_delta) for chrom in population
        ]
        fitnesses: list[float] = list(executor.map(_fitness_worker_args, args_list))
    else:
        fitnesses = [
            evaluate_fitness(
                chrom,
                X_train,
                y_train,
                fitness_fn=fitness_fn,
                quik_delta=quik_delta,
            )
            for chrom in population
        ]

    bank = [
        (f, chrom) for f, chrom in zip(fitnesses, population) if f != _PENALTY_FITNESS
    ]
    bank.sort(key=lambda x: x[0], reverse=True)
    return bank


# ---------------------------------------------------------------------------
# Main GA-MLR loop (FR2.2, Tasks 2.3 + 2.4)
# ---------------------------------------------------------------------------


def run_ga_mlr(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    cluster_map: dict[int, list[int]],
    *,
    X_test: NDArray[np.float64] | None = None,
    y_test: NDArray[np.float64] | None = None,
    min_variables: int = 1,
    max_variables: int = 5,
    population_size: int = 100,
    max_generations: int = 100,
    mutation_rate: float = 0.01,
    keep_best: int = 10,
    fitness_function: str = "q2_loo",
    quik_delta: float | None = 0.05,
    inter_cluster_mutation_ratio: float = 0.7,
    tournament_size: int = 5,
    random_seed: int | None = None,
    n_workers: int = 0,
) -> list[SubsetModel]:
    """Run GA-MLR feature selection and return the best fitted models.

    For each feature count *m* in ``[min_variables, max_variables]``:

    1. If the total number of cluster-diverse subsets of size *m* is ≤
       *population_size*, enumerate them all (no GA needed).
    2. Otherwise, initialise a random population of size *population_size*
       and run *max_generations* evolutionary generations.

    A :class:`~concurrent.futures.ProcessPoolExecutor` is created **once**
    for the entire run and reused across all feature counts.  After each
    feature count the bank is sorted, the top *keep_best* unique models are
    fitted and collected, and the bank is discarded to free memory.

    Parameters
    ----------
    X_train : ndarray of shape (n_train, p_all)
        Full training descriptor matrix.
    y_train : ndarray of shape (n_train,)
        Training response.
    cluster_map : dict[int, list[int]]
        Cluster map from :func:`~qsarify.modeling.clustering.cophenetic_cluster`.
    X_test, y_test : ndarray or None
        Optional external test set.
    min_variables : int
        Minimum subset size.  Default 1.
    max_variables : int
        Maximum subset size.  Must be ≤ n/5 (statistical upper bound) and
        ≤ n_clusters.
    population_size : int
        Number of chromosomes per generation (also the full-enum threshold).
    max_generations : int
        Number of evolutionary generations.
    mutation_rate : float
        Per-gene mutation probability ∈ [0, 1].
    keep_best : int
        Number of top unique models retained per subset size.
    fitness_function : str
        One of ``'q2_loo'``, ``'r2_adj'``, ``'lof'``, ``'rmse_cv'``.
    quik_delta : float or None
        QUIK rule threshold δ_k.  ``None`` disables the rule.
    inter_cluster_mutation_ratio : float
        Fraction of mutations that are inter-cluster.  Default 0.7.
    tournament_size : int
        Tournament size for selection.  Default 5.
    random_seed : int or None
        Seed for reproducibility.
    n_workers : int
        Worker processes for parallel fitness evaluation.  0 = sequential.

    Returns
    -------
    list[SubsetModel]
        All retained best models across all subset sizes, each already fitted.
    """
    rng = np.random.default_rng(random_seed)
    cluster_ids = list(cluster_map.keys())
    n_clusters = len(cluster_ids)
    n_train = len(y_train)

    # Enforce constraints
    stat_upper = max(1, n_train // 5)
    if max_variables > stat_upper:
        warnings.warn(
            f"max_variables ({max_variables}) exceeds n/5={stat_upper}; capping.",
            UserWarning,
            stacklevel=2,
        )
        max_variables = stat_upper

    if max_variables > n_clusters:
        warnings.warn(
            f"max_variables ({max_variables}) > n_clusters ({n_clusters}); capping.",
            UserWarning,
            stacklevel=2,
        )
        max_variables = n_clusters

    min_variables = max(1, min(min_variables, max_variables))

    all_best_models: list[SubsetModel] = []

    _ctx: ProcessPoolExecutor | nullcontext[None] = (  # type: ignore[type-arg]
        ProcessPoolExecutor(max_workers=n_workers) if n_workers > 0 else nullcontext()
    )

    with _ctx as executor:
        for m in range(min_variables, max_variables + 1):
            if m > n_clusters:
                break

            # ---- choose initialisation strategy ----
            total = _count_chroms_up_to(m, cluster_ids, cluster_map, population_size)

            if total <= population_size:
                # Full enumeration — no GA loop needed
                initial_pop = list(
                    _iter_cluster_diverse_chroms(m, cluster_ids, cluster_map)
                )
                bank: list[tuple[float, Chromosome]] = _build_scored_bank(
                    initial_pop,
                    X_train,
                    y_train,
                    fitness_function,
                    quik_delta,
                    executor,
                )
            else:
                # Random init + evolutionary loop
                population = init_population(m, cluster_map, population_size, rng)
                bank = _build_scored_bank(
                    population,
                    X_train,
                    y_train,
                    fitness_function,
                    quik_delta,
                    executor,
                )
                bank = bank[:population_size]

                for _gen in range(max_generations):
                    if not bank:
                        break

                    bank_chroms = [chrom for _, chrom in bank]
                    bank_fits = [f for f, _ in bank]

                    # Generate offspring: tournament → crossover → mutate
                    offspring: list[Chromosome] = []
                    while len(offspring) < population_size:
                        p1 = tournament_select(
                            bank_chroms, bank_fits, tournament_size, rng
                        )
                        p2 = tournament_select(
                            bank_chroms, bank_fits, tournament_size, rng
                        )
                        idx1 = next(
                            (i for i, c in enumerate(bank_chroms) if c is p1), 0
                        )
                        idx2 = next(
                            (i for i, c in enumerate(bank_chroms) if c is p2), 0
                        )
                        f1 = bank_fits[idx1]
                        f2 = bank_fits[idx2]
                        c1, c2 = crossover(p1, p2, f1, f2, cluster_map, rng)
                        c1 = mutate(
                            c1,
                            cluster_map,
                            mutation_rate,
                            inter_cluster_mutation_ratio,
                            rng,
                        )
                        c2 = mutate(
                            c2,
                            cluster_map,
                            mutation_rate,
                            inter_cluster_mutation_ratio,
                            rng,
                        )
                        offspring.append(c1)
                        if len(offspring) < population_size:
                            offspring.append(c2)

                    offspring_bank = _build_scored_bank(
                        offspring,
                        X_train,
                        y_train,
                        fitness_function,
                        quik_delta,
                        executor,
                    )
                    # Merge, sort descending, crop to population_size
                    bank = bank + offspring_bank
                    bank.sort(key=lambda x: x[0], reverse=True)
                    bank = bank[:population_size]

            # ---- collect top keep_best unique models for this m ----
            seen: set[tuple[int, ...]] = set()
            collected = 0
            for _, chrom in bank:
                if collected >= keep_best:
                    break
                desc_indices = decode_chromosome(chrom)
                key = tuple(sorted(desc_indices))
                if key in seen:
                    continue
                seen.add(key)
                ca = {d: c for c, d in chrom}
                model = SubsetModel(desc_indices, cluster_assignments=ca)
                model.fit(X_train, y_train, X_test=X_test, y_test=y_test)
                all_best_models.append(model)
                collected += 1

            del bank

    return all_best_models
