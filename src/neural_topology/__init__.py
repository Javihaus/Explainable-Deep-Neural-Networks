"""
Explainable Deep Neural Networks: Topological Analysis Framework

A professional framework for understanding the inner topology of neural network layers
through Topological Data Analysis (TDA), featuring Vietoris-Rips persistence homology
and Mapper algorithms for explainable AI.
"""

__version__ = "1.0.0"
__author__ = "Javier Marin"
__email__ = "javier@jmarin.info"

from .core.analyzer import TopologicalAnalyzer
from .core.persistence import PersistenceComputer
from .core.mapper import MapperPipeline
from .neural.extractors import NeuralExtractor
from .visualization.persistence_plots import PersistencePlotter
from .visualization.mapper_plots import MapperPlotter
from .utils.data_processing import DataProcessor

__all__ = [
    'TopologicalAnalyzer',
    'PersistenceComputer', 
    'MapperPipeline',
    'NeuralExtractor',
    'PersistencePlotter',
    'MapperPlotter',
    'DataProcessor'
]