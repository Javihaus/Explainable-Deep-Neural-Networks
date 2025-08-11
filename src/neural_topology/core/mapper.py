"""
Mapper algorithm implementation for neural network topology analysis.

This module implements the Mapper algorithm for creating topological summaries
of high-dimensional neural network activation data.
"""

from typing import Dict, List, Optional, Any, Union, Callable
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler


class MapperPipeline:
    """
    Mapper algorithm implementation for topological data visualization.
    
    The Mapper algorithm creates a topological summary of high-dimensional data
    by applying a filter function, creating a cover, clustering, and building
    a simplicial complex.
    """
    
    def __init__(self,
                 filter_func: Union[str, Callable] = 'eccentricity',
                 cover_intervals: int = 20,
                 overlap_fraction: float = 0.5,
                 clusterer: Any = None,
                 verbose: bool = False,
                 n_jobs: int = -1):
        """
        Initialize Mapper pipeline.
        
        Args:
            filter_func: Filter function ('eccentricity', 'projection', 'entropy')
            cover_intervals: Number of intervals in the cover
            overlap_fraction: Overlap between adjacent intervals
            clusterer: Clustering algorithm (default: DBSCAN)
            verbose: Whether to print progress information
            n_jobs: Number of parallel jobs
        """
        self.filter_func = filter_func
        self.cover_intervals = cover_intervals
        self.overlap_fraction = overlap_fraction
        self.clusterer = clusterer or DBSCAN(eps=0.5, min_samples=2)
        self.verbose = verbose
        self.n_jobs = n_jobs
        
        # Initialize components
        self._init_components()
    
    def _init_components(self):
        """Initialize Mapper pipeline components."""
        try:
            from gtda.mapper import (
                CubicalCover, make_mapper_pipeline, 
                Eccentricity, Entropy, Projection
            )
            
            # Set up filter function
            if self.filter_func == 'eccentricity':
                self.filter = Eccentricity(metric='euclidean')
            elif self.filter_func == 'entropy':
                self.filter = Entropy()
            elif self.filter_func == 'projection':
                self.filter = Projection()
            else:
                self.filter = Eccentricity(metric='euclidean')
            
            # Set up cover
            self.cover = CubicalCover(
                n_intervals=self.cover_intervals,
                overlap_frac=self.overlap_fraction
            )
            
            # Create pipeline
            self.mapper_pipeline = make_mapper_pipeline(
                filter_func=self.filter,
                cover=self.cover,
                clusterer=self.clusterer,
                verbose=self.verbose,
                n_jobs=self.n_jobs
            )
            
            self.available = True
            
        except ImportError:
            print("Warning: giotto-tda not available for Mapper algorithm")
            self.available = False
    
    def fit_transform(self, data: np.ndarray) -> Any:
        """
        Apply Mapper algorithm to data.
        
        Args:
            data: Input data array of shape (n_samples, n_features)
            
        Returns:
            Mapper graph object
        """
        if not self.available:
            return self._fallback_mapper(data)
        
        # Apply Mapper pipeline
        mapper_graph = self.mapper_pipeline.fit_transform(data)
        
        return mapper_graph
    
    def _fallback_mapper(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Fallback Mapper implementation without giotto-tda.
        
        This provides a simplified version of the Mapper algorithm.
        """
        if self.verbose:
            print("Using fallback Mapper implementation")
        
        # Compute filter values (using distance from centroid as proxy)
        centroid = np.mean(data, axis=0)
        filter_values = np.linalg.norm(data - centroid, axis=1)
        
        # Create cover
        filter_min, filter_max = np.min(filter_values), np.max(filter_values)
        interval_length = (filter_max - filter_min) / self.cover_intervals
        overlap_length = interval_length * self.overlap_fraction
        
        # Build graph
        nodes = {}
        edges = []
        node_id = 0
        
        for i in range(self.cover_intervals):
            # Define interval
            start = filter_min + i * interval_length - (overlap_length if i > 0 else 0)
            end = start + interval_length + overlap_length
            
            # Get points in interval
            mask = (filter_values >= start) & (filter_values <= end)
            if not np.any(mask):
                continue
            
            interval_data = data[mask]
            
            # Cluster points in interval
            if len(interval_data) > 1:
                cluster_labels = self.clusterer.fit_predict(interval_data)
                
                for cluster_id in np.unique(cluster_labels):
                    if cluster_id >= 0:  # Skip noise points
                        cluster_mask = cluster_labels == cluster_id
                        cluster_points = np.where(mask)[0][cluster_mask]
                        
                        nodes[node_id] = {
                            'points': cluster_points,
                            'size': len(cluster_points),
                            'interval': i,
                            'cluster': cluster_id
                        }
                        node_id += 1
        
        # Create simplified graph structure
        graph = {
            'nodes': nodes,
            'edges': edges,
            'n_nodes': len(nodes),
            'n_edges': len(edges),
            'filter_values': filter_values
        }
        
        return graph
    
    def plot_static_graph(self, graph: Any,
                         color_by: Optional[np.ndarray] = None,
                         layout: str = 'fruchterman_reingold',
                         title: str = "Mapper Graph",
                         save_path: Optional[str] = None) -> Any:
        """
        Plot static Mapper graph.
        
        Args:
            graph: Mapper graph object
            color_by: Array for coloring nodes
            layout: Graph layout algorithm
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        try:
            from gtda.mapper import plot_static_mapper_graph
            
            if self.available and hasattr(graph, 'fit_transform'):
                plotly_params = {"node_trace": {"marker_colorscale": "RdBu"}}
                
                fig = plot_static_mapper_graph(
                    graph,
                    data=None,  # Will be passed separately
                    layout=layout,
                    color_variable=color_by,
                    node_scale=20,
                    plotly_params=plotly_params
                )
                
                fig.update_layout(title=title)
                
                if save_path:
                    fig.write_html(save_path)
                
                return fig
            else:
                return self._fallback_plot(graph, color_by, title)
                
        except ImportError:
            return self._fallback_plot(graph, color_by, title)
    
    def _fallback_plot(self, graph: Dict[str, Any], 
                      color_by: Optional[np.ndarray] = None,
                      title: str = "Mapper Graph") -> None:
        """Fallback plotting without giotto-tda."""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            
            # Create NetworkX graph
            G = nx.Graph()
            
            # Add nodes
            for node_id, node_data in graph['nodes'].items():
                G.add_node(node_id, size=node_data['size'])
            
            # Add edges (simplified connectivity)
            node_list = list(graph['nodes'].keys())
            for i, node1 in enumerate(node_list):
                for node2 in node_list[i+1:]:
                    # Connect nodes from adjacent intervals
                    if abs(graph['nodes'][node1]['interval'] - 
                          graph['nodes'][node2]['interval']) <= 1:
                        G.add_edge(node1, node2)
            
            # Plot
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G)
            
            # Color nodes
            if color_by is not None:
                node_colors = []
                for node_id in G.nodes():
                    points = graph['nodes'][node_id]['points']
                    avg_color = np.mean(color_by[points]) if len(points) > 0 else 0
                    node_colors.append(avg_color)
            else:
                node_colors = 'lightblue'
            
            # Draw graph
            nx.draw(G, pos, 
                   node_color=node_colors,
                   node_size=[graph['nodes'][node]['size'] * 50 
                             for node in G.nodes()],
                   with_labels=True,
                   font_size=8,
                   cmap=plt.cm.RdBu if color_by is not None else None)
            
            plt.title(title)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Visualization requires matplotlib and networkx")
    
    def plot_interactive_graph(self, graph: Any,
                              color_by: Optional[np.ndarray] = None,
                              title: str = "Interactive Mapper Graph") -> Any:
        """
        Plot interactive Mapper graph.
        
        Args:
            graph: Mapper graph object
            color_by: Array for coloring nodes
            title: Plot title
            
        Returns:
            Interactive plot object
        """
        try:
            from gtda.mapper import plot_interactive_mapper_graph
            
            if self.available:
                fig = plot_interactive_mapper_graph(
                    graph,
                    color_variable=color_by,
                    title=title
                )
                return fig
            else:
                print("Interactive plotting requires giotto-tda")
                return None
                
        except ImportError:
            print("Interactive plotting requires giotto-tda")
            return None
    
    def get_graph_metrics(self, graph: Any) -> Dict[str, float]:
        """
        Compute metrics for Mapper graph.
        
        Args:
            graph: Mapper graph object
            
        Returns:
            Dictionary of graph metrics
        """
        if isinstance(graph, dict):
            # Fallback graph format
            return {
                'n_nodes': graph['n_nodes'],
                'n_edges': graph['n_edges'],
                'connectivity': graph['n_edges'] / max(1, graph['n_nodes']),
                'avg_node_size': np.mean([node['size'] for node in graph['nodes'].values()])
            }
        else:
            # giotto-tda graph format
            try:
                # Extract basic metrics (implementation depends on giotto-tda internals)
                return {
                    'n_nodes': len(graph.nodes()) if hasattr(graph, 'nodes') else 0,
                    'n_edges': len(graph.edges()) if hasattr(graph, 'edges') else 0
                }
            except:
                return {'n_nodes': 0, 'n_edges': 0}