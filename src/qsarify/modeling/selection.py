import warnings
from typing import Dict

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import pdist


class FeatureCluster:
    """
    Feature clustering using hierarchical clustering based on correlation matrix.
    Enhanced version of the original clustering approach with proper error handling.
    """

    def __init__(self, X_data: pd.DataFrame, method: str = 'corr',
                 link: str = 'average', cut_d: float = 3.0, epsilon: float = 1e-10):
        """
        Initialize feature clustering.

        Args:
            X_data: Input features DataFrame
            method: Method for distance calculation ('corr' for correlation)
            link: Linkage method for hierarchical clustering
            cut_d: Distance threshold for clustering
            epsilon: Small value to ensure positive definite matrix
        """
        self.method = method
        self.X_data = X_data
        self.link = link
        self.cut_d = cut_d
        self.epsilon = epsilon
        self.cluster_info = []
        self.assignments = np.array([])
        self.cluster_output = pd.DataFrame()
        self.cludict = {}
        self.xcorr = None
        self._compute_correlation_matrix()

    def _compute_correlation_matrix(self):
        """Compute positive definite correlation matrix."""
        # Compute correlation matrix
        corr_matrix = np.corrcoef(self.X_data.T)

        # Handle NaN values
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Ensure positive definite by adding epsilon to diagonal
        corr_matrix += np.eye(corr_matrix.shape[0]) * self.epsilon

        # Take absolute values for distance calculation
        self.xcorr = pd.DataFrame(
            np.abs(corr_matrix),
            columns=self.X_data.columns,
            index=self.X_data.columns
        )

    def cophenetic_correlation(self) -> Dict[str, float]:
        """Calculate cophenetic correlation coefficients for different linkage methods."""
        distance_matrix = 1 - self.xcorr.values  # Convert correlation to distance

        methods = ['average', 'complete', 'single']
        cophenetic_coeffs = {}

        for method in methods:
            try:
                Z = linkage(pdist(distance_matrix), method=method)
                c, _ = cophenet(Z, pdist(distance_matrix))
                cophenetic_coeffs[method] = c
            except Exception as e:
                warnings.warn(f"Failed to calculate cophenetic correlation for {method}: {e}")
                cophenetic_coeffs[method] = 0.0

        return cophenetic_coeffs
    
    def find_optimal_cutoff(self, cutoff_range: tuple = (1.0, 5.0), n_points: int = 20) -> float:
        """
        Find optimal clustering cutoff using automated cluster quality assessment.
        
        Uses cophenetic correlation to evaluate cluster quality across different cutoffs
        and selects the one that maximizes clustering goodness while preserving diversity.
        """
        cutoffs = np.linspace(cutoff_range[0], cutoff_range[1], n_points)
        best_cutoff = self.cut_d
        best_score = -1.0
        
        # Get cophenetic correlation for current linkage method
        distance_matrix = 1 - self.xcorr.values
        Z = linkage(pdist(distance_matrix), method=self.link)
        coph_corr, _ = cophenet(Z, pdist(distance_matrix))
        
        for cutoff in cutoffs:
            try:
                # Test clustering with this cutoff
                assignments = fcluster(Z, cutoff, criterion='distance')
                n_clusters = len(np.unique(assignments))
                
                # Skip if too few or too many clusters
                if n_clusters < 2 or n_clusters > len(self.X_data.columns) * 0.8:
                    continue
                
                # Calculate cluster quality metrics
                cluster_sizes = np.bincount(assignments)
                mean_cluster_size = np.mean(cluster_sizes)
                cluster_size_std = np.std(cluster_sizes)
                
                # Penalize very unbalanced clustering
                balance_penalty = cluster_size_std / (mean_cluster_size + 1e-10)
                
                # Score combines cophenetic correlation with balance
                diversity_score = n_clusters / len(self.X_data.columns)  # Favor more diverse clustering
                score = coph_corr * diversity_score / (1 + balance_penalty)
                
                if score > best_score:
                    best_score = score
                    best_cutoff = cutoff
                    
            except Exception:
                continue
                
        return best_cutoff

    def set_cluster(self, verbose: bool = False, auto_cutoff: bool = True) -> Dict[str, int]:
        """
        Perform hierarchical clustering of features with optional automated cutoff selection.

        Args:
            verbose: Whether to print cluster information
            auto_cutoff: Whether to automatically find optimal cutoff

        Returns:
            Dictionary mapping feature names to cluster numbers
        """
        try:
            # Use automated cutoff selection if requested
            if auto_cutoff:
                optimal_cutoff = self.find_optimal_cutoff()
                if verbose:
                    print(f"Automated cutoff selection: {optimal_cutoff:.3f} (original: {self.cut_d:.3f})")
                cutoff_to_use = optimal_cutoff
            else:
                cutoff_to_use = self.cut_d
            
            distance_matrix = 1 - self.xcorr.values
            Z = linkage(pdist(distance_matrix), method=self.link)
            self.assignments = fcluster(Z, cutoff_to_use, criterion='distance')

            self.cluster_output = pd.DataFrame({
                'Feature': list(self.X_data.columns),
                'cluster': self.assignments
            })

            # Create feature -> cluster mapping
            self.cludict = dict(zip(self.cluster_output.Feature, self.cluster_output.cluster))

            # Create cluster information for feature selection
            max_cluster = max(self.assignments)
            self.cluster_info = []

            for cluster_id in range(1, max_cluster + 1):
                cluster_features = [
                    k for k, v in self.cludict.items() if v == cluster_id
                ]
                self.cluster_info.append(cluster_features)

                if verbose:
                    print(f"Cluster {cluster_id}: {cluster_features}")

            return self.cludict

        except Exception as e:
            warnings.warn(f"Clustering failed: {e}")
            # Fallback: each feature in its own cluster
            self.cludict = {feat: i+1 for i, feat in enumerate(self.X_data.columns)}
            self.cluster_info = [[feat] for feat in self.X_data.columns]
            return self.cludict
