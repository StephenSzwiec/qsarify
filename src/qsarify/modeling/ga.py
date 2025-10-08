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
    """Configuration for the GA-MLR feature selection process."""

    max_vars: int = 10
    population_size: int = 50
    max_generations: int = 100
    mutation_rate: float = 0.1
    keep_best: int = 5  # Keep best individuals per generation
    top_k_models: int = 3  # Keep top K models per M features
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
    Selects optimal feature subsets using a hybrid cluster-based strategy.
    
    Key innovation: For 3+ features, tournament selection operates over feature clusters
    rather than individual features, ensuring diversity and avoiding highly correlated features.

    - For 1-2 features: Full exhaustive subset search over all individual features.
    - For 3+ features: Genetic algorithm with cluster-based tournament selection.
    """

    def __init__(self, config: GeneticConfig):
        self.config = config
        self.clusterer: Optional[FeatureCluster] = None
        self.best_individuals: Dict[int, List[Dict[str, Any]]] = {}  # Store top K models per M features
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
    ) -> List[Dict[str, Any]]:
        """Performs an exhaustive search and returns top K feature subsets of given size."""
        if n_features > len(X.columns):
            return []
            
        all_combinations = list(itertools.combinations(X.columns, n_features))
        if self.config.verbose:
            print(f"  Testing {len(all_combinations)} combinations for {n_features} feature(s)...")

        results = Parallel(n_jobs=self.config.n_jobs)(
            delayed(self._calculate_fitness)(list(individual), X, y)
            for individual in all_combinations
        )

        # Collect all valid results (those that pass QUIK rule if enabled)
        valid_results = []
        primary_metric = self.config.fitness_functions[0]

        for i, fitness_scores in enumerate(results):
            score = fitness_scores.get(primary_metric, -np.inf)
            if score > -np.inf:  # Valid fitness (passed QUIK rule)
                valid_results.append({
                    "features": list(all_combinations[i]),
                    "fitness": fitness_scores,
                    "score": score
                })

        # Sort by primary fitness function and return top K
        valid_results.sort(key=lambda x: x["score"], reverse=True)
        top_k_results = valid_results[:self.config.top_k_models]
        
        # Remove the temporary score field
        for result in top_k_results:
            del result["score"]
            
        if self.config.verbose:
            print(f"  Found {len(valid_results)} valid models, keeping top {len(top_k_results)}")
            
        return top_k_results


    def _cluster_based_population_init(self, n_features: int, X_features: List[str]) -> List[str]:
        """Generate individual using cluster-based selection to ensure diversity."""
        if not self.clusterer:
            return self.random_state.sample(X_features, min(n_features, len(X_features)))
        
        selected_features = []
        available_clusters = list(range(len(self.clusterer.cluster_info)))
        
        # First, select one feature from each cluster (if possible)
        for _ in range(min(n_features, len(available_clusters))):
            if not available_clusters:
                break
                
            cluster_idx = self.random_state.choice(available_clusters)
            cluster_features = [f for f in self.clusterer.cluster_info[cluster_idx] 
                             if f in X_features and f not in selected_features]
            
            if cluster_features:
                selected_feature = self.random_state.choice(cluster_features)
                selected_features.append(selected_feature)
                available_clusters.remove(cluster_idx)
        
        # Fill remaining slots with random features from any cluster
        while len(selected_features) < n_features:
            remaining_features = [f for f in X_features if f not in selected_features]
            if not remaining_features:
                break
            selected_features.append(self.random_state.choice(remaining_features))
        
        return sorted(selected_features[:n_features])

    def _run_cluster_based_ga(
        self,
        n_features: int,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> List[Dict[str, Any]]:
        """Runs cluster-based GA to find top K models for a fixed number of features."""
        if not self.clusterer:
            raise RuntimeError("Clusterer not initialized.")

        X_features = list(X.columns)
        
        # 1. Create initial population using cluster-based initialization
        population: List[Tuple[List[str], Dict[str, float]]] = []
        attempts = 0
        max_attempts = self.config.population_size * 10
        
        while len(population) < self.config.population_size and attempts < max_attempts:
            individual = self._cluster_based_population_init(n_features, X_features)
            
            # Avoid duplicates
            if individual not in [p[0] for p in population]:
                fitness = self._calculate_fitness(individual, X, y)
                # Only add if passes QUIK rule (fitness not -inf)
                if fitness.get(self.config.fitness_functions[0], -np.inf) > -np.inf:
                    population.append((individual, fitness))
            attempts += 1
        
        if not population:
            warnings.warn(f"Could not create valid initial population for {n_features} features.")
            return []

        if self.config.verbose and len(population) < self.config.population_size:
            print(f"    Generated {len(population)} valid individuals (target: {self.config.population_size})")

        # 2. Evolution loop with cluster-aware breeding
        for generation in range(self.config.max_generations):
            # Sort by primary fitness function
            population.sort(
                key=lambda ind: ind[1].get(self.config.fitness_functions[0], -np.inf),
                reverse=True
            )
            
            # Keep best individuals
            new_population = population[:self.config.keep_best]

            # Generate offspring using cluster-aware crossover and mutation
            while len(new_population) < self.config.population_size:
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                
                child1, child2 = self._cluster_aware_crossover(parent1, parent2, n_features, X_features)
                child1 = self._cluster_aware_mutation(child1, X_features)
                child2 = self._cluster_aware_mutation(child2, X_features)

                for child in [child1, child2]:
                    if child and len(new_population) < self.config.population_size:
                        fitness = self._calculate_fitness(child, X, y)
                        # Only add if passes QUIK rule
                        if fitness.get(self.config.fitness_functions[0], -np.inf) > -np.inf:
                            new_population.append((child, fitness))
                
                # Break if we can't generate more valid offspring
                if len(new_population) == self.config.population_size or attempts > max_attempts:
                    break
                    
            population = new_population

        # 3. Return top K individuals from final population
        population.sort(
            key=lambda ind: ind[1].get(self.config.fitness_functions[0], -np.inf),
            reverse=True
        )
        
        top_k_individuals = population[:self.config.top_k_models]
        top_k_results = [
            {"features": ind[0], "fitness": ind[1]} 
            for ind in top_k_individuals
        ]
        
        if self.config.verbose:
            print(f"    Keeping top {len(top_k_results)} models from GA")
        
        return top_k_results

    def _cluster_aware_crossover(self, parent1: List[str], parent2: List[str], 
                               n_features: int, X_features: List[str]) -> Tuple[List[str], List[str]]:
        """Crossover that respects cluster diversity."""
        combined_features = list(set(parent1 + parent2))
        
        child1 = self._cluster_based_population_init(n_features, combined_features)
        child2 = self._cluster_based_population_init(n_features, combined_features)
        
        return child1, child2

    def _cluster_aware_mutation(self, individual: List[str], X_features: List[str]) -> List[str]:
        """Mutation that maintains cluster diversity."""
        if self.random_state.random() > self.config.mutation_rate:
            return individual

        if not individual or not self.clusterer:
            return individual

        mutated = individual.copy()
        
        # Choose a random feature to replace
        replace_idx = self.random_state.randrange(len(mutated))
        current_feature = mutated[replace_idx]
        current_cluster_id = self.clusterer.cludict.get(current_feature)

        # Find features from different clusters
        different_cluster_features = []
        for cluster_id, cluster_features in enumerate(self.clusterer.cluster_info):
            if (cluster_id + 1) != current_cluster_id:
                different_cluster_features.extend([
                    f for f in cluster_features 
                    if f in X_features and f not in mutated
                ])

        if different_cluster_features:
            new_feature = self.random_state.choice(different_cluster_features)
            mutated[replace_idx] = new_feature

        return sorted(list(set(mutated)))

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
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Runs the GA-MLR feature selection process.
        
        Process:
        1. Always cluster X columns first using cophenetic distance on correlation matrix
        2. Automated cluster quality assessment for optimal cutoff
        3. Exhaustive search for 1-2 features (manageable combinatorial load)
        4. GA with cluster-based tournament selection for 3+ features
        5. QUIK rule filtering during fitness evaluation
        6. Returns top K models per M features
        """
        n_samples = len(X)
        max_vars_limit = max(1, n_samples // 5)  # Statistical bound: n/5
        
        if self.config.max_vars > max_vars_limit:
            warnings.warn(
                f"max_vars ({self.config.max_vars}) exceeds statistical bound (n/5 = {max_vars_limit}). "
                f"Consider reducing max_vars for statistical validity."
            )
        
        effective_max_vars = min(self.config.max_vars, max_vars_limit)
        
        if self.config.verbose:
            print("Starting GA-MLR feature selection...")
            print(f"Dataset: {n_samples} samples, {len(X.columns)} features")
            print(f"Max variables limit: {effective_max_vars} (statistical bound: n/5)")

        # Always cluster X columns first using automated quality assessment
        self.clusterer = FeatureCluster(
            X,
            cut_d=self.config.clustering_distance,
            link=self.config.clustering_method,
            epsilon=self.config.epsilon,
            **(clustering_config or {}),
        )
        
        # Use automated cutoff selection for optimal clustering
        self.clusterer.set_cluster(verbose=self.config.verbose, auto_cutoff=True)
        
        if self.config.verbose:
            n_clusters = len(self.clusterer.cluster_info)
            print(f"Clustered {len(X.columns)} features into {n_clusters} clusters")

        # Phase 1: Full exhaustive subset search (1-2 features)
        for n_vars in range(1, min(3, effective_max_vars + 1)):
            if self.config.verbose:
                print(f"\nExhaustive search for {n_vars} features...")
            top_models = self._full_subset_search(n_vars, X, y)
            if top_models:
                self.best_individuals[n_vars] = top_models

        # Phase 2: GA with cluster-based tournament selection (3+ features)
        for n_vars in range(3, effective_max_vars + 1):
            if self.config.verbose:
                print(f"\nRunning cluster-based GA for {n_vars} features...")
            top_models = self._run_cluster_based_ga(n_vars, X, y)
            if top_models:
                self.best_individuals[n_vars] = top_models

        if self.config.verbose:
            print("\nGA-MLR feature selection completed.")
            for n_vars, models in sorted(self.best_individuals.items()):
                print(f"  {n_vars} features: {len(models)} top models found")
                for i, model in enumerate(models[:3]):  # Show top 3
                    fitness_summary = {k: f"{v:.4f}" for k, v in model['fitness'].items()}
                    print(f"    #{i+1}: {fitness_summary}")

        return self.best_individuals
