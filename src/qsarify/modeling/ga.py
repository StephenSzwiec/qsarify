import random
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from ..utils.statistics import (
    calculate_lof, calculate_q_squared_loo, calculate_r_squared_adj
)
from .selection import FeatureCluster


@dataclass
class GeneticConfig:
    """Configuration for Genetic Algorithm feature selection"""
    max_vars: int = 10
    population_size: int = 50
    max_generations: int = 100
    mutation_rate: float = 0.1
    keep_best: int = 5
    fitness_functions: List[str] = field(default_factory=lambda: ['Q2loo', 'R2Adj', 'LOF', 'RMSE-CV'])
    significance_levels: List[float] = field(
        default_factory=lambda: [0.0001, 0.001, 0.01, 0.05, 0.10, 0.15, 0.20]
    )
    clustering_distance: float = 3.0
    clustering_method: str = 'average'
    epsilon: float = 1e-10
    n_jobs: int = -1
    random_state: int = 42
    verbose: bool = True
    quik_rule: bool = True


class GeneticFeatureSelector:
    """
    Genetic Algorithm with Tournament Selection for feature selection.
    Enhanced version with proper statistical validation and QUIK rule.
    """

    def __init__(self, config: GeneticConfig):
        """
        Initialize genetic feature selector.

        Args:
            config: Configuration for genetic algorithm
        """
        self.config = config
        self.clusterer = None
        self.population = []
        self.fitness_history = []
        self.best_individuals = {}
        self.random_state = random.Random(config.random_state)
        np.random.seed(config.random_state)

    def _create_individual(self, max_features: int) -> List[str]:
        """Create a random individual (feature subset)."""
        if not self.clusterer or not self.clusterer.cluster_info:
            return []

        n_features = self.random_state.randint(1, min(max_features, len(self.clusterer.cluster_info)))
        selected_clusters = self.random_state.sample(
            range(len(self.clusterer.cluster_info)),
            n_features
        )

        individual = []
        for cluster_idx in selected_clusters:
            cluster_features = self.clusterer.cluster_info[cluster_idx]
            if cluster_features:
                feature = self.random_state.choice(cluster_features)
                individual.append(feature)

        return sorted(list(set(individual)))  # Remove duplicates and sort

    def _mutate_individual(self, individual: List[str]) -> List[str]:
        """Mutate an individual by swapping features."""
        if (not individual or not self.clusterer or
            self.random_state.random() > self.config.mutation_rate):
            return individual

        mutated = individual.copy()

        # Choose mutation type
        mutation_type = self.random_state.choice(['swap', 'add', 'remove'])

        if mutation_type == 'swap' and len(mutated) > 0:
            # Swap one feature with another from a different cluster
            swap_idx = self.random_state.randint(0, len(mutated) - 1)
            current_feature = mutated[swap_idx]
            current_cluster = self.clusterer.cludict.get(current_feature, -1)

            # Find a different cluster
            available_clusters = [
                i for i, cluster in enumerate(self.clusterer.cluster_info)
                if i != (current_cluster - 1) and cluster  # cluster IDs are 1-indexed
            ]

            if available_clusters:
                new_cluster_idx = self.random_state.choice(available_clusters)
                new_feature = self.random_state.choice(self.clusterer.cluster_info[new_cluster_idx])
                mutated[swap_idx] = new_feature

        elif mutation_type == 'add' and len(mutated) < self.config.max_vars:
            # Add a feature from an unused cluster
            used_clusters = set()
            for feature in mutated:
                cluster_id = self.clusterer.cludict.get(feature, -1)
                if cluster_id > 0:
                    used_clusters.add(cluster_id - 1)  # Convert to 0-indexed

            available_clusters = [
                i for i in range(len(self.clusterer.cluster_info))
                if i not in used_clusters and self.clusterer.cluster_info[i]
            ]

            if available_clusters:
                new_cluster_idx = self.random_state.choice(available_clusters)
                new_feature = self.random_state.choice(self.clusterer.cluster_info[new_cluster_idx])
                mutated.append(new_feature)

        elif mutation_type == 'remove' and len(mutated) > 1:
            # Remove a random feature
            remove_idx = self.random_state.randint(0, len(mutated) - 1)
            mutated.pop(remove_idx)

        return sorted(list(set(mutated)))

    def _crossover(self, parent1: List[str], parent2: List[str]) -> Tuple[List[str], List[str]]:
        """Perform crossover between two parents."""
        if len(parent1) <= 1 or len(parent2) <= 1:
            return parent1.copy(), parent2.copy()

        # Find common features
        common = list(set(parent1) & set(parent2))
        unique1 = list(set(parent1) - set(parent2))
        unique2 = list(set(parent2) - set(parent1))

        # Create children by mixing unique features
        child1 = common.copy()
        child2 = common.copy()

        all_unique = unique1 + unique2
        self.random_state.shuffle(all_unique)

        # Split unique features between children
        mid = len(all_unique) // 2
        child1.extend(all_unique[:mid])
        child2.extend(all_unique[mid:])

        # Ensure children don't exceed max_vars
        child1 = sorted(child1[:self.config.max_vars])
        child2 = sorted(child2[:self.config.max_vars])

        return child1, child2

    def _calculate_fitness(self, individual: List[str], X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate fitness scores for an individual."""
        if not individual or len(individual) == 0:
            return {metric: -np.inf for metric in self.config.fitness_functions}

        try:
            X_subset = X[individual]

            # Check for sufficient data
            if len(X_subset) < 2 or X_subset.shape[1] == 0:
                return {metric: -np.inf for metric in self.config.fitness_functions}

            # Create and fit model
            model = LinearRegression()

            fitness_scores = {}

            # Calculate different fitness metrics
            if 'Q2loo' in self.config.fitness_functions:
                try:
                    q2_loo = calculate_q_squared_loo(model, X_subset.values, y.values)
                    fitness_scores['Q2loo'] = q2_loo if not np.isnan(q2_loo) else -np.inf
                except:
                    fitness_scores['Q2loo'] = -np.inf

            # Fit model for other metrics
            model.fit(X_subset, y)
            y_pred = model.predict(X_subset)

            if 'R2Adj' in self.config.fitness_functions:
                try:
                    r2_adj = calculate_r_squared_adj(y.values, y_pred, X_subset.shape[1])
                    fitness_scores['R2Adj'] = r2_adj if not np.isnan(r2_adj) else -np.inf
                except:
                    fitness_scores['R2Adj'] = -np.inf

            if 'LOF' in self.config.fitness_functions:
                try:
                    lof = calculate_lof(y.values, y_pred, X_subset.shape[1])
                    # LOF should be minimized, so we negate it
                    fitness_scores['LOF'] = -lof if not np.isnan(lof) else -np.inf
                except:
                    fitness_scores['LOF'] = -np.inf

            if 'RMSE-CV' in self.config.fitness_functions:
                try:
                    cv_scores = cross_val_score(
                        model, X_subset, y,
                        cv=self.config.cv_folds if hasattr(self.config, 'cv_folds') else 5,
                        scoring='neg_root_mean_squared_error'
                    )
                    rmse_cv = -np.mean(cv_scores)
                    # RMSE should be minimized, so we negate it
                    fitness_scores['RMSE-CV'] = -rmse_cv if not np.isnan(rmse_cv) else -np.inf
                except:
                    fitness_scores['RMSE-CV'] = -np.inf

            # Apply QUIK rule if enabled
            if self.config.quik_rule:
                n_samples, n_features = X_subset.shape
                if n_features > 0:
                    ratio = n_samples / n_features
                    if ratio < 5:  # QUIK rule threshold
                        # Penalize models that don't meet QUIK rule
                        penalty_factor = 0.1
                        fitness_scores = {k: v * penalty_factor for k, v in fitness_scores.items()}

            return fitness_scores

        except Exception as e:
            warnings.warn(f"Fitness calculation failed for {individual}: {e}")
            return {metric: -np.inf for metric in self.config.fitness_functions}

    def _tournament_selection(self, population: List[Tuple[List[str], Dict[str, float]]],
                            tournament_size: int = 3) -> List[str]:
        """Select individual using tournament selection."""
        tournament = self.random_state.sample(population, min(tournament_size, len(population)))

        # Select based on primary fitness function (first in list)
        primary_metric = self.config.fitness_functions[0]
        best_individual = max(tournament, key=lambda x: x[1].get(primary_metric, -np.inf))

        return best_individual[0]

    def fit(self, X: pd.DataFrame, y: pd.Series, clustering_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run genetic algorithm for feature selection.

        Args:
            X: Feature matrix
            y: Target variable
            clustering_config: Optional configuration for clustering

        Returns:
            Dictionary containing best individuals and selection results
        """
        if self.config.verbose:
            print(f"Starting genetic algorithm feature selection...")
            print(f"Data shape: {X.shape}")

        # Perform hierarchical clustering
        clustering_params = clustering_config or {}
        self.clusterer = FeatureCluster(
            X,
            cut_d=self.config.clustering_distance,
            link=self.config.clustering_method,
            epsilon=self.config.epsilon,
            **clustering_params
        )

        cluster_dict = self.clusterer.set_cluster(verbose=self.config.verbose)

        if self.config.verbose:
            print(f"Created {len(self.clusterer.cluster_info)} clusters")
            cophenetic_corrs = self.clusterer.cophenetic_correlation()
            print(f"Cophenetic correlations: {cophenetic_corrs}")

        # Initialize population
        self.population = []
        for _ in range(self.config.population_size):
            individual = self._create_individual(self.config.max_vars)
            if individual:  # Only add non-empty individuals
                fitness = self._calculate_fitness(individual, X, y)
                self.population.append((individual, fitness))

        if not self.population:
            raise ValueError("Failed to create valid population")

        # Evolution loop
        for generation in range(self.config.max_generations):
            new_population = []

            # Keep best individuals
            sorted_pop = sorted(
                self.population,
                key=lambda x: x[1].get(self.config.fitness_functions[0], -np.inf),
                reverse=True
            )

            new_population.extend(sorted_pop[:self.config.keep_best])

            # Generate new individuals
            while len(new_population) < self.config.population_size:
                parent1 = self._tournament_selection(self.population)
                parent2 = self._tournament_selection(self.population)

                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate_individual(child1)
                child2 = self._mutate_individual(child2)

                for child in [child1, child2]:
                    if child and len(new_population) < self.config.population_size:
                        fitness = self._calculate_fitness(child, X, y)
                        new_population.append((child, fitness))

            self.population = new_population[:self.config.population_size]

            # Log progress
            if self.config.verbose and generation % 10 == 0:
                best_fitness = max(
                    ind[1].get(self.config.fitness_functions[0], -np.inf)
                    for ind in self.population
                )
                print(f"Generation {generation}: Best {self.config.fitness_functions[0]} = {best_fitness:.4f}")

        # Collect best individuals by number of variables
        self.best_individuals = {}
        for n_vars in range(1, self.config.max_vars + 1):
            candidates = [
                ind for ind in self.population
                if len(ind[0]) == n_vars
            ]

            if candidates:
                best = max(
                    candidates,
                    key=lambda x: x[1].get(self.config.fitness_functions[0], -np.inf)
                )
                self.best_individuals[n_vars] = {
                    'features': best[0],
                    'fitness': best[1],
                    'n_features': len(best[0])
                }

        if self.config.verbose:
            print("Genetic algorithm completed!")
            for n_vars, result in self.best_individuals.items():
                print(f"{n_vars} vars: {result['fitness']} - {result['features']}")

        return self.best_individuals
