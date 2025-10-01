import random
import warnings
import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed

from ..utils.statistics import (
    calculate_kxx,
    calculate_kxy,
    calculate_lof,
    calculate_q_squared_loo,
    calculate_r_squared_adj,
)
from .selection import FeatureCluster


@dataclass
class GeneticConfig:
    """Configuration for the hybrid feature selection process."""

    max_vars: int = 10
    population_size: int = 50
    max_generations: int = 100
    mutation_rate: float = 0.1
    keep_best: int = 5
    fitness_functions: List[str] = field(
        default_factory=lambda: ["Q2loo", "R2Adj", "LOF", "RMSE-CV"]
    )
    use_quik_rule: bool = True
    delta_k: float = 0.05
    significance_levels: List[float] = field(
        default_factory=lambda: [0.0001, 0.001, 0.01, 0.05, 0.10, 0.15, 0.20]
    )
    clustering_distance: float = 3.0
    clustering_method: str = "average"
    epsilon: float = 1e-10
    n_jobs: int = -1
    random_state: int = 42
    verbose: bool = True


class GeneticFeatureSelector:
    """
    Selects optimal feature subsets using a hybrid strategy.

    - For 1-2 features: A full subset search is performed.
    - For 3+ features: A genetic algorithm with a fixed feature count is used for each size.
    """

    def __init__(self, config: GeneticConfig):
        self.config = config
        self.clusterer: Optional[FeatureCluster] = None
        self.best_individuals: Dict[int, Dict[str, Any]] = {}
        self.random_state = random.Random(config.random_state)
        np.random.seed(config.random_state)

    def _calculate_fitness(
        self,
        individual: List[str],
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, float]:
        """Calculates fitness scores for a single individual."""
        if not individual:
            return {metric: -np.inf for metric in self.config.fitness_functions}

        try:
            X_subset = X[individual]
            if X_subset.shape[1] == 0:
                return {metric: -np.inf for metric in self.config.fitness_functions}

            if self.config.use_quik_rule:
                k_xx = calculate_kxx(X_subset.values)
                k_xy = calculate_kxy(X_subset.values, y.values)
                if (k_xy - k_xx) < self.config.delta_k:
                    return {metric: -np.inf for metric in self.config.fitness_functions}

            model = LinearRegression(n_jobs=1)
            fitness_scores = {}

            if "Q2loo" in self.config.fitness_functions:
                q2_loo = calculate_q_squared_loo(model, X_subset.values, y.values)
                fitness_scores["Q2loo"] = q2_loo if not np.isnan(q2_loo) else -np.inf

            if "RMSE-CV" in self.config.fitness_functions:
                cv_scores = cross_val_score(
                    model, X_subset, y, cv=5, scoring="neg_root_mean_squared_error"
                )
                fitness_scores["RMSE-CV"] = -np.mean(cv_scores)

            model.fit(X_subset, y)
            y_pred = model.predict(X_subset)

            if "R2Adj" in self.config.fitness_functions:
                r2_adj = calculate_r_squared_adj(y.values, y_pred, X_subset.shape[1])
                fitness_scores["R2Adj"] = r2_adj if not np.isnan(r2_adj) else -np.inf

            if "LOF" in self.config.fitness_functions:
                lof = calculate_lof(y.values, y_pred, X_subset.shape[1])
                fitness_scores["LOF"] = -lof if not np.isnan(lof) else -np.inf

            return fitness_scores

        except Exception:
            return {metric: -np.inf for metric in self.config.fitness_functions}

    def _full_subset_search(
        self,
        n_features: int,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Optional[Dict[str, Any]]:
        """Performs an exhaustive search for the best feature subset of a given size."""
        if n_features > len(X.columns):
            return None
            
        all_combinations = list(itertools.combinations(X.columns, n_features))
        if self.config.verbose:
            print(f"  Testing {len(all_combinations)} combinations for {n_features} feature(s)પૂર્ણ...")

        results = Parallel(n_jobs=self.config.n_jobs)(
            delayed(self._calculate_fitness)(list(individual), X, y)
            for individual in all_combinations
        )

        best_fitness = -np.inf
        best_result = None
        primary_metric = self.config.fitness_functions[0]

        for i, fitness_scores in enumerate(results):
            score = fitness_scores.get(primary_metric, -np.inf)
            if score > best_fitness:
                best_fitness = score
                best_result = {
                    "features": list(all_combinations[i]),
                    "fitness": fitness_scores,
                }
        return best_result

    def _mutate_swap_only(self, individual: List[str]) -> List[str]:
        """Performs a cluster-aware swap mutation, preserving feature count."""
        if not self.clusterer or self.random_state.random() > self.config.mutation_rate:
            return individual

        mutated = individual.copy()
        swap_idx = self.random_state.randrange(len(mutated))
        current_feature = mutated[swap_idx]
        current_cluster_id = self.clusterer.cludict.get(current_feature)

        # Find a feature from a different cluster
        available_clusters = [
            i for i, features in enumerate(self.clusterer.cluster_info)
            if features and (i + 1) != current_cluster_id
        ]
        if available_clusters:
            new_cluster_idx = self.random_state.choice(available_clusters)
            new_feature = self.random_state.choice(self.clusterer.cluster_info[new_cluster_idx])
            mutated[swap_idx] = new_feature

        return sorted(list(set(mutated)))

    def _crossover_fixed_n(
        self,
        parent1: List[str],
        parent2: List[str],
        n_features: int,
    ) -> Tuple[List[str], List[str]]:
        """Performs crossover that preserves the feature count."""
        combined_pool = sorted(list(set(parent1) | set(parent2)))
        
        child1 = self.random_state.sample(combined_pool, min(n_features, len(combined_pool)))
        child2 = self.random_state.sample(combined_pool, min(n_features, len(combined_pool)))

        return sorted(child1), sorted(child2)

    def _run_ga_for_n_features(
        self,
        n_features: int,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Optional[Dict[str, Any]]:
        """Runs a dedicated GA to find the best model for a fixed number of features."""
        if not self.clusterer:
            raise RuntimeError("Clusterer not initialized.")

        # 1. Create initial population
        population: List[Tuple[List[str], Dict[str, float]]] = []
        attempts = 0
        while len(population) < self.config.population_size and attempts < self.config.population_size * 5:
            individual = sorted(list(set(self.random_state.sample(list(X.columns), n_features))))
            if individual not in [p[0] for p in population]:
                fitness = self._calculate_fitness(individual, X, y)
                population.append((individual, fitness))
            attempts += 1
        
        if not population:
            warnings.warn(f"Could not create initial population for {n_features} features.")
            return None

        # 2. Evolution loop
        for generation in range(self.config.max_generations):
            sorted_pop = sorted(
                population,
                key=lambda ind: ind[1].get(self.config.fitness_functions[0], -np.inf),
                reverse=True,
            )
            new_population = sorted_pop[: self.config.keep_best]

            while len(new_population) < self.config.population_size:
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                child1, child2 = self._crossover_fixed_n(parent1, parent2, n_features)
                child1 = self._mutate_swap_only(child1)
                child2 = self._mutate_swap_only(child2)

                for child in [child1, child2]:
                    if child and len(new_population) < self.config.population_size:
                        fitness = self._calculate_fitness(child, X, y)
                        new_population.append((child, fitness))
            population = new_population

        # 3. Return best individual from the final population
        best_individual = max(
            population, key=lambda ind: ind[1].get(self.config.fitness_functions[0], -np.inf)
        )
        return {"features": best_individual[0], "fitness": best_individual[1]}

    def _tournament_selection(
        self,
        population: List[Tuple[List[str], Dict[str, float]]],
        tournament_size: int = 3,
    ) -> List[str]:
        """Selects an individual using tournament selection."""
        tournament = self.random_state.sample(population, min(tournament_size, len(population)))
        primary_metric = self.config.fitness_functions[0]
        best_individual = max(tournament, key=lambda ind: ind[1].get(primary_metric, -np.inf))
        return best_individual[0]

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        clustering_config: Optional[Dict] = None,
    ) -> Dict[int, Any]:
        """
        Runs the hybrid feature selection process.
        """
        if self.config.verbose:
            print("Starting hybrid feature selection...")

        self.clusterer = FeatureCluster(
            X,
            cut_d=self.config.clustering_distance,
            link=self.config.clustering_method,
            epsilon=self.config.epsilon,
            **(clustering_config or {}),
        )
        self.clusterer.set_cluster(verbose=False) # Verbosity handled here

        # --- Phase 1: Full Subset Search (1-2 features) ---
        for n_vars in range(1, 3):
            if n_vars > self.config.max_vars:
                break
            best_for_n = self._full_subset_search(n_vars, X, y)
            if best_for_n:
                self.best_individuals[n_vars] = best_for_n

        # --- Phase 2: Genetic Algorithm (3+ features) ---
        for n_vars in range(3, self.config.max_vars + 1):
            if self.config.verbose:
                print(f"Running GA for {n_vars} features...")
            best_for_n = self._run_ga_for_n_features(n_vars, X, y)
            if best_for_n:
                self.best_individuals[n_vars] = best_for_n

        if self.config.verbose:
            print("\nFeature selection completed.")
            for n_vars, result in sorted(self.best_individuals.items()):
                print(f"  Best model with {n_vars} vars: {result['fitness']}")

        return self.best_individuals
