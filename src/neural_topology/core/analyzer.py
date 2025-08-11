"""
Main topological analyzer for neural networks.

This module provides the primary interface for analyzing the topological structure
of neural network layers using various TDA techniques.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass

from .persistence import PersistenceComputer
from .mapper import MapperPipeline
from ..visualization.persistence_plots import PersistencePlotter
from ..visualization.mapper_plots import MapperPlotter


@dataclass
class TopologyResults:
    """Results container for topological analysis."""
    persistence_diagrams: Dict[str, np.ndarray]
    betti_numbers: Dict[str, Dict[int, int]]
    mapper_graphs: Dict[str, Any]
    umap_projections: Dict[str, np.ndarray]
    metadata: Dict[str, Any]


class TopologicalAnalyzer:
    """
    Main class for analyzing the topology of neural network layers.
    
    This class orchestrates the various TDA techniques to provide comprehensive
    topological analysis of neural network layer activations.
    """
    
    def __init__(self, 
                 homology_dimensions: Optional[List[int]] = None,
                 n_jobs: int = -1):
        """
        Initialize the topological analyzer.
        
        Args:
            homology_dimensions: Dimensions for homology computation [0, 1, 2]
            n_jobs: Number of parallel jobs for computation
        """
        self.homology_dimensions = homology_dimensions or [0, 1, 2]
        self.n_jobs = n_jobs
        
        # Initialize components
        self.persistence_computer = PersistenceComputer(
            homology_dimensions=self.homology_dimensions,
            n_jobs=self.n_jobs
        )
        self.mapper = MapperPipeline()
        self.persistence_plotter = PersistencePlotter()
        self.mapper_plotter = MapperPlotter()
    
    def analyze_network_topology(self,
                                activations: Dict[str, np.ndarray],
                                labels: Optional[np.ndarray] = None,
                                compute_mapper: bool = True,
                                compute_umap: bool = True) -> TopologyResults:
        """
        Perform comprehensive topological analysis on network activations.
        
        Args:
            activations: Dictionary mapping layer names to activation arrays
            labels: Optional target labels for visualization
            compute_mapper: Whether to compute Mapper graphs
            compute_umap: Whether to compute UMAP projections
            
        Returns:
            TopologyResults containing all analysis results
        """
        results = TopologyResults(
            persistence_diagrams={},
            betti_numbers={},
            mapper_graphs={},
            umap_projections={},
            metadata={}
        )
        
        for layer_name, activation_data in activations.items():
            print(f"Analyzing topology of {layer_name}...")
            
            # Compute persistence diagrams
            persistence_diagram = self.persistence_computer.compute_persistence(
                activation_data
            )
            results.persistence_diagrams[layer_name] = persistence_diagram
            
            # Compute Betti numbers
            betti_numbers = self.persistence_computer.compute_betti_numbers(
                persistence_diagram
            )
            results.betti_numbers[layer_name] = betti_numbers
            
            # Compute Mapper graph if requested
            if compute_mapper:
                mapper_graph = self.mapper.fit_transform(activation_data)
                results.mapper_graphs[layer_name] = mapper_graph
            
            # Compute UMAP projection if requested
            if compute_umap:
                umap_projection = self._compute_umap_projection(activation_data)
                results.umap_projections[layer_name] = umap_projection
        
        # Store metadata
        results.metadata = {
            'layer_count': len(activations),
            'homology_dimensions': self.homology_dimensions,
            'has_labels': labels is not None,
            'sample_count': next(iter(activations.values())).shape[0]
        }
        
        return results
    
    def _compute_umap_projection(self, data: np.ndarray,
                               n_components: int = 2) -> np.ndarray:
        """
        Compute UMAP projection of high-dimensional data.
        
        Args:
            data: Input data array
            n_components: Number of UMAP components
            
        Returns:
            UMAP projection array
        """
        try:
            import umap.umap_ as umap
            
            reducer = umap.UMAP(
                n_neighbors=10,
                min_dist=0.5,
                n_components=n_components,
                random_state=42
            )
            
            return reducer.fit_transform(data)
        except ImportError:
            print("UMAP not available. Install with: pip install umap-learn")
            return np.empty((data.shape[0], n_components))
    
    def plot_persistence_diagrams(self, results: TopologyResults,
                                save_path: Optional[str] = None):
        """Plot persistence diagrams for all layers."""
        self.persistence_plotter.plot_multi_layer_persistence(
            results.persistence_diagrams,
            save_path=save_path
        )
    
    def plot_mapper_graphs(self, activations: Dict[str, np.ndarray],
                          labels: Optional[np.ndarray] = None,
                          save_path: Optional[str] = None):
        """Plot Mapper graphs for all layers."""
        for layer_name, activation_data in activations.items():
            self.mapper_plotter.plot_static_graph(
                self.mapper.fit_transform(activation_data),
                color_by=labels,
                title=f"Mapper Graph: {layer_name}",
                save_path=save_path
            )
    
    def plot_layer_evolution(self, betti_numbers: Dict[str, Dict[int, int]],
                           save_path: Optional[str] = None):
        """Plot evolution of topological features across layers."""
        self.persistence_plotter.plot_betti_evolution(
            betti_numbers,
            save_path=save_path
        )
    
    def compute_topology_metrics(self, results: TopologyResults) -> Dict[str, float]:
        """
        Compute summary metrics for topological analysis.
        
        Args:
            results: TopologyResults from analysis
            
        Returns:
            Dictionary of topology metrics
        """
        metrics = {}
        
        for layer_name, betti_nums in results.betti_numbers.items():
            # Total topological complexity
            total_features = sum(betti_nums.values())
            metrics[f"{layer_name}_total_features"] = total_features
            
            # Feature diversity (non-zero Betti numbers)
            feature_diversity = len([b for b in betti_nums.values() if b > 0])
            metrics[f"{layer_name}_feature_diversity"] = feature_diversity
            
            # Persistence entropy (if available)
            if layer_name in results.persistence_diagrams:
                entropy = self._compute_persistence_entropy(
                    results.persistence_diagrams[layer_name]
                )
                metrics[f"{layer_name}_persistence_entropy"] = entropy
        
        return metrics
    
    def _compute_persistence_entropy(self, persistence_diagram: np.ndarray) -> float:
        """
        Compute persistence entropy from persistence diagram.
        
        Args:
            persistence_diagram: Array of (birth, death, dimension) triplets
            
        Returns:
            Persistence entropy value
        """
        # Extract lifespans (death - birth) for finite features
        finite_features = persistence_diagram[persistence_diagram[:, 1] != np.inf]
        if len(finite_features) == 0:
            return 0.0
            
        lifespans = finite_features[:, 1] - finite_features[:, 0]
        
        # Normalize to get probabilities
        total_persistence = np.sum(lifespans)
        if total_persistence == 0:
            return 0.0
            
        probabilities = lifespans / total_persistence
        
        # Compute entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        return entropy