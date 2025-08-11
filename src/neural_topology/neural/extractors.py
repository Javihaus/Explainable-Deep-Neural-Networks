"""
Neural network layer activation extractors.

This module provides utilities for extracting intermediate layer activations
from various deep learning frameworks for topological analysis.
"""

from typing import Dict, List, Optional, Any, Union
import numpy as np
from abc import ABC, abstractmethod


class BaseExtractor(ABC):
    """Base class for neural network activation extractors."""
    
    @abstractmethod
    def extract_layer_activations(self, model: Any, data: np.ndarray, 
                                layer_name: str) -> np.ndarray:
        """Extract activations from a specific layer."""
        pass
    
    @abstractmethod
    def extract_all_layers(self, model: Any, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract activations from all layers."""
        pass


class TensorFlowExtractor(BaseExtractor):
    """Extractor for TensorFlow/Keras models."""
    
    def __init__(self):
        """Initialize TensorFlow extractor."""
        try:
            import tensorflow as tf
            self.tf = tf
            self.available = True
        except ImportError:
            print("TensorFlow not available")
            self.available = False
    
    def extract_layer_activations(self, model: Any, data: np.ndarray, 
                                layer_name: str) -> np.ndarray:
        """
        Extract activations from a specific Keras layer.
        
        Args:
            model: Keras model
            data: Input data
            layer_name: Name of the layer to extract from
            
        Returns:
            Layer activations array
        """
        if not self.available:
            raise ImportError("TensorFlow not available")
        
        # Create intermediate model
        intermediate_model = self.tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer(layer_name).output
        )
        
        # Get activations
        activations = intermediate_model.predict(data, verbose=0)
        
        return activations
    
    def extract_all_layers(self, model: Any, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract activations from all layers in a Keras model.
        
        Args:
            model: Keras model
            data: Input data
            
        Returns:
            Dictionary mapping layer names to activation arrays
        """
        if not self.available:
            raise ImportError("TensorFlow not available")
        
        layer_outputs = {}
        
        # Get all layer names (excluding input layer)
        layer_names = [layer.name for layer in model.layers if len(layer.output_shape) > 1]
        
        for layer_name in layer_names:
            try:
                activations = self.extract_layer_activations(model, data, layer_name)
                
                # Flatten if needed (keep batch dimension)
                if len(activations.shape) > 2:
                    activations = activations.reshape(activations.shape[0], -1)
                
                layer_outputs[layer_name] = activations
                
            except Exception as e:
                print(f"Warning: Could not extract from layer {layer_name}: {e}")
        
        return layer_outputs


class PyTorchExtractor(BaseExtractor):
    """Extractor for PyTorch models."""
    
    def __init__(self):
        """Initialize PyTorch extractor."""
        try:
            import torch
            self.torch = torch
            self.available = True
        except ImportError:
            print("PyTorch not available")
            self.available = False
        
        self.hooks = []
        self.activations = {}
    
    def _hook_fn(self, name: str):
        """Create a hook function for layer activation extraction."""
        def hook(model, input, output):
            # Convert to numpy and flatten if needed
            activation = output.detach().cpu().numpy()
            if len(activation.shape) > 2:
                activation = activation.reshape(activation.shape[0], -1)
            self.activations[name] = activation
        return hook
    
    def extract_layer_activations(self, model: Any, data: np.ndarray, 
                                layer_name: str) -> np.ndarray:
        """
        Extract activations from a specific PyTorch layer.
        
        Args:
            model: PyTorch model
            data: Input data
            layer_name: Name of the layer to extract from
            
        Returns:
            Layer activations array
        """
        if not self.available:
            raise ImportError("PyTorch not available")
        
        model.eval()
        self.activations = {}
        
        # Register hook
        layer = dict(model.named_modules())[layer_name]
        hook = layer.register_forward_hook(self._hook_fn(layer_name))
        
        # Forward pass
        with self.torch.no_grad():
            tensor_data = self.torch.tensor(data, dtype=self.torch.float32)
            _ = model(tensor_data)
        
        # Remove hook
        hook.remove()
        
        return self.activations.get(layer_name, np.array([]))
    
    def extract_all_layers(self, model: Any, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract activations from all layers in a PyTorch model.
        
        Args:
            model: PyTorch model
            data: Input data
            
        Returns:
            Dictionary mapping layer names to activation arrays
        """
        if not self.available:
            raise ImportError("PyTorch not available")
        
        model.eval()
        self.activations = {}
        
        # Register hooks for all layers
        for name, layer in model.named_modules():
            if len(list(layer.children())) == 0:  # Leaf layers only
                hook = layer.register_forward_hook(self._hook_fn(name))
                self.hooks.append(hook)
        
        # Forward pass
        with self.torch.no_grad():
            tensor_data = self.torch.tensor(data, dtype=self.torch.float32)
            _ = model(tensor_data)
        
        # Remove all hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        return self.activations.copy()


class NeuralExtractor:
    """
    Unified neural network activation extractor.
    
    This class automatically detects the framework and uses the appropriate extractor.
    """
    
    def __init__(self):
        """Initialize the unified extractor."""
        self.tensorflow_extractor = TensorFlowExtractor()
        self.pytorch_extractor = PyTorchExtractor()
    
    def _detect_framework(self, model: Any) -> str:
        """
        Detect the deep learning framework of the model.
        
        Args:
            model: Neural network model
            
        Returns:
            Framework name ('tensorflow' or 'pytorch')
        """
        model_type = str(type(model))
        
        if 'keras' in model_type.lower() or 'tensorflow' in model_type.lower():
            return 'tensorflow'
        elif 'torch' in model_type.lower():
            return 'pytorch'
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")
    
    def extract_layer_activations(self, model: Any, data: np.ndarray, 
                                layer_name: str) -> np.ndarray:
        """
        Extract activations from a specific layer.
        
        Args:
            model: Neural network model
            data: Input data
            layer_name: Name of the layer to extract from
            
        Returns:
            Layer activations array
        """
        framework = self._detect_framework(model)
        
        if framework == 'tensorflow':
            return self.tensorflow_extractor.extract_layer_activations(
                model, data, layer_name
            )
        elif framework == 'pytorch':
            return self.pytorch_extractor.extract_layer_activations(
                model, data, layer_name
            )
        else:
            raise ValueError(f"Unsupported framework: {framework}")
    
    def extract_all_layers(self, model: Any, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract activations from all layers.
        
        Args:
            model: Neural network model
            data: Input data
            
        Returns:
            Dictionary mapping layer names to activation arrays
        """
        framework = self._detect_framework(model)
        
        if framework == 'tensorflow':
            return self.tensorflow_extractor.extract_all_layers(model, data)
        elif framework == 'pytorch':
            return self.pytorch_extractor.extract_all_layers(model, data)
        else:
            raise ValueError(f"Unsupported framework: {framework}")
    
    def get_layer_names(self, model: Any) -> List[str]:
        """
        Get names of all layers in the model.
        
        Args:
            model: Neural network model
            
        Returns:
            List of layer names
        """
        framework = self._detect_framework(model)
        
        if framework == 'tensorflow':
            return [layer.name for layer in model.layers]
        elif framework == 'pytorch':
            return [name for name, _ in model.named_modules() 
                   if len(list(_.children())) == 0]
        else:
            raise ValueError(f"Unsupported framework: {framework}")
    
    def get_layer_info(self, model: Any) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed information about all layers.
        
        Args:
            model: Neural network model
            
        Returns:
            Dictionary with layer information
        """
        framework = self._detect_framework(model)
        layer_info = {}
        
        if framework == 'tensorflow':
            for layer in model.layers:
                layer_info[layer.name] = {
                    'type': layer.__class__.__name__,
                    'output_shape': layer.output_shape,
                    'params': layer.count_params() if hasattr(layer, 'count_params') else 0
                }
        elif framework == 'pytorch':
            for name, module in model.named_modules():
                if len(list(module.children())) == 0:  # Leaf layers only
                    layer_info[name] = {
                        'type': module.__class__.__name__,
                        'params': sum(p.numel() for p in module.parameters())
                    }
        
        return layer_info