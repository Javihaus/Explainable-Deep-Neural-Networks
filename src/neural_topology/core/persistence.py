"""
Persistent homology computation for neural network layer analysis.

This module implements Vietoris-Rips persistent homology computation
for analyzing the topological structure of neural network activations.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import pairwise_distances


class PersistenceComputer:
    """
    Computes persistent homology using Vietoris-Rips complex.
    
    This class provides methods for computing persistent homology of point clouds
    derived from neural network layer activations.
    """
    
    def __init__(self,
                 homology_dimensions: List[int] = [0, 1, 2],
                 metric: str = 'euclidean',
                 n_jobs: int = -1,
                 coeff: int = 2):
        """
        Initialize persistence computer.
        
        Args:
            homology_dimensions: Dimensions for homology computation
            metric: Distance metric for Vietoris-Rips complex
            n_jobs: Number of parallel jobs
            coeff: Coefficient field for homology computation
        """
        self.homology_dimensions = homology_dimensions
        self.metric = metric
        self.n_jobs = n_jobs
        self.coeff = coeff
        
        # Initialize giotto-tda components
        self._init_persistence_pipeline()
    
    def _init_persistence_pipeline(self):
        """Initialize the persistent homology computation pipeline."""
        try:
            from gtda.homology import VietorisRipsPersistence
            
            self.vr_persistence = VietorisRipsPersistence(
                homology_dimensions=self.homology_dimensions,
                coeff=self.coeff,
                n_jobs=self.n_jobs
            )
        except ImportError:
            print("Warning: giotto-tda not available. Install with: pip install giotto-tda")
            self.vr_persistence = None
    
    def compute_persistence(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        Compute persistence diagram for a point cloud.
        
        Args:
            point_cloud: Array of shape (n_samples, n_features)
            
        Returns:
            Persistence diagram as array of (birth, death, dimension) triplets
        """
        if self.vr_persistence is None:
            return self._fallback_persistence(point_cloud)
        
        # Reshape for giotto-tda (expects 3D array)
        point_cloud_3d = point_cloud.reshape(1, *point_cloud.shape)
        
        # Compute persistence diagram
        persistence_diagram = self.vr_persistence.fit_transform(point_cloud_3d)
        
        # Return as 2D array
        return persistence_diagram[0]
    
    def _fallback_persistence(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        Fallback persistence computation without giotto-tda.
        
        This provides a simplified version that computes basic connectivity information.
        """
        print("Using fallback persistence computation (simplified version)")
        
        # Compute pairwise distances
        distances = pairwise_distances(point_cloud, metric=self.metric)
        
        # Simple connected components analysis
        n_points = len(point_cloud)
        sorted_distances = np.sort(distances.flatten())
        
        # Create simplified persistence diagram
        persistence_pairs = []
        
        # H0 (connected components) - simplified version
        for i in range(min(10, n_points)):  # Limit to avoid memory issues
            birth = 0.0
            death = sorted_distances[min(i * n_points + 10, len(sorted_distances) - 1)]
            if death > birth:
                persistence_pairs.append([birth, death, 0])
        
        return np.array(persistence_pairs) if persistence_pairs else np.empty((0, 3))
    
    def compute_betti_numbers(self, persistence_diagram: np.ndarray) -> Dict[int, int]:
        """
        Compute Betti numbers from persistence diagram.
        
        Args:
            persistence_diagram: Array of (birth, death, dimension) triplets
            
        Returns:
            Dictionary mapping homology dimension to Betti number
        """
        betti_numbers = {dim: 0 for dim in self.homology_dimensions}
        
        if len(persistence_diagram) == 0:
            return betti_numbers
        
        for birth, death, dimension in persistence_diagram:
            dim = int(dimension)
            if dim in betti_numbers:
                # Count infinite features (death = inf) or long-lived features
                if np.isinf(death) or (death - birth) > self._get_persistence_threshold():
                    betti_numbers[dim] += 1
        
        return betti_numbers
    
    def _get_persistence_threshold(self) -> float:
        """Get threshold for considering a feature as significant."""
        return 0.1  # This could be made configurable
    
    def compute_persistence_landscape(self, persistence_diagram: np.ndarray,
                                    resolution: int = 100) -> np.ndarray:
        """
        Compute persistence landscape from persistence diagram.
        
        Args:
            persistence_diagram: Array of (birth, death, dimension) triplets
            resolution: Resolution for landscape computation
            
        Returns:
            Persistence landscape array
        """
        try:
            from gtda.diagrams import PersistenceLandscape
            
            landscape_computer = PersistenceLandscape(
                n_landscapes=5,
                n_bins=resolution
            )
            
            # Reshape for giotto-tda
            diagram_3d = persistence_diagram.reshape(1, *persistence_diagram.shape)
            landscape = landscape_computer.fit_transform(diagram_3d)
            
            return landscape[0]
            
        except ImportError:
            print("Persistence landscape requires giotto-tda")
            return np.empty((resolution,))
    
    def compute_persistence_entropy(self, persistence_diagram: np.ndarray) -> float:
        """
        Compute persistence entropy from persistence diagram.
        
        Args:
            persistence_diagram: Array of (birth, death, dimension) triplets
            
        Returns:
            Persistence entropy value
        """
        try:
            from gtda.diagrams import PersistenceEntropy
            
            entropy_computer = PersistenceEntropy()
            
            # Reshape for giotto-tda
            diagram_3d = persistence_diagram.reshape(1, *persistence_diagram.shape)
            entropy = entropy_computer.fit_transform(diagram_3d)
            
            return entropy[0, 0]
            
        except ImportError:
            # Fallback entropy computation
            return self._fallback_entropy(persistence_diagram)
    
    def _fallback_entropy(self, persistence_diagram: np.ndarray) -> float:
        """Fallback entropy computation without giotto-tda."""
        if len(persistence_diagram) == 0:
            return 0.0
        
        # Extract finite lifespans
        finite_features = persistence_diagram[persistence_diagram[:, 1] != np.inf]
        if len(finite_features) == 0:
            return 0.0
        
        lifespans = finite_features[:, 1] - finite_features[:, 0]
        total_persistence = np.sum(lifespans)
        
        if total_persistence == 0:
            return 0.0
        
        # Compute probabilities and entropy
        probabilities = lifespans / total_persistence
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        return entropy
    
    def plot_persistence_diagram(self, persistence_diagram: np.ndarray,
                                save_path: Optional[str] = None):
        """
        Plot persistence diagram.
        
        Args:
            persistence_diagram: Array of (birth, death, dimension) triplets
            save_path: Optional path to save the plot
        """
        try:
            from gtda.plotting import plot_diagram
            
            # Reshape for giotto-tda
            diagram_3d = persistence_diagram.reshape(1, *persistence_diagram.shape)
            
            fig = plot_diagram(diagram_3d[0])
            
            if save_path:
                fig.write_image(save_path)
            
            fig.show()
            
        except ImportError:
            print("Plotting requires giotto-tda and plotly")
    
    def compare_persistence_diagrams(self, 
                                   diagram1: np.ndarray,
                                   diagram2: np.ndarray,
                                   metric: str = 'wasserstein') -> float:
        """
        Compare two persistence diagrams using specified metric.
        
        Args:
            diagram1: First persistence diagram
            diagram2: Second persistence diagram
            metric: Distance metric ('wasserstein', 'bottleneck')
            
        Returns:
            Distance between diagrams
        """
        try:
            from gtda.diagrams import PairwiseDistance
            
            distance_computer = PairwiseDistance(metric=metric, n_jobs=self.n_jobs)
            
            # Reshape for giotto-tda
            diagrams = np.array([diagram1, diagram2]).reshape(2, -1, 3)
            
            distance_matrix = distance_computer.fit_transform(diagrams)
            
            return distance_matrix[0, 1]
            
        except ImportError:
            print(f"{metric} distance requires giotto-tda")
            return 0.0