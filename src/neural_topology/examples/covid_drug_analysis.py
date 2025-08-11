"""
COVID-19 Drug Discovery Topological Analysis Example

This example demonstrates how to use the neural topology framework
to analyze the topological structure of neural networks trained on
COVID-19 drug discovery data.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt

from neural_topology import (
    TopologicalAnalyzer, 
    NeuralExtractor,
    DataProcessor
)


class CovidDrugAnalyzer:
    """
    Specialized analyzer for COVID-19 drug discovery data.
    
    This class provides methods for loading, preprocessing, and analyzing
    molecular data using topological methods.
    """
    
    def __init__(self):
        """Initialize the COVID drug analyzer."""
        self.topology_analyzer = TopologicalAnalyzer()
        self.neural_extractor = NeuralExtractor()
        self.data_processor = DataProcessor()
        
    def load_covid_dataset(self, filepath: str) -> pd.DataFrame:
        """
        Load and preprocess COVID-19 drug discovery dataset.
        
        Args:
            filepath: Path to CSV file containing molecular data
            
        Returns:
            Preprocessed DataFrame
        """
        print("Loading COVID-19 drug discovery dataset...")
        
        # Load data
        data = pd.read_csv(filepath)
        
        # Remove rows with missing pIC50 values or 'BLINDED' entries
        data = data.dropna()
        clean_data = data[data['pIC50'] != 'BLINDED'].copy()
        
        # Convert pIC50 to numeric
        clean_data['pIC50'] = pd.to_numeric(clean_data['pIC50'], errors='coerce')
        clean_data = clean_data.dropna()
        
        # Remove non-numeric columns for feature extraction
        feature_columns = clean_data.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in feature_columns if col != 'pIC50']
        
        print(f"Loaded {len(clean_data)} samples with {len(feature_columns)} features")
        
        return clean_data
    
    def build_drug_classifier(self, 
                             input_dim: int,
                             architecture: list = [50, 40, 20, 1],
                             target_column: str = 'pIC50') -> Any:
        """
        Build neural network for drug classification.
        
        Args:
            input_dim: Number of input features
            architecture: Network architecture (list of layer sizes)
            target_column: Name of target column
            
        Returns:
            Trained neural network model
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout
            from tensorflow.keras import regularizers
            
            print("Building neural network classifier...")
            
            model = Sequential()
            
            # Input layer
            model.add(Dense(
                architecture[0],
                activation='relu',
                kernel_initializer='random_normal',
                kernel_regularizer=regularizers.l2(0.05),
                input_dim=input_dim
            ))
            
            # Hidden layers
            for units in architecture[1:-1]:
                model.add(Dense(
                    units,
                    activation='relu',
                    kernel_initializer='random_normal',
                    kernel_regularizer=regularizers.l2(0.05)
                ))
            
            # Output layer
            model.add(Dense(
                architecture[-1],
                activation='sigmoid',
                kernel_initializer='random_normal',
                kernel_regularizer=regularizers.l2(0.05)
            ))
            
            # Compile model
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            print(f"Model architecture: {architecture}")
            print(f"Total parameters: {model.count_params():,}")
            
            return model
            
        except ImportError:
            raise ImportError("TensorFlow is required for building neural networks")
    
    def prepare_training_data(self, 
                            data: pd.DataFrame,
                            target_column: str = 'pIC50',
                            test_size: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data for neural network.
        
        Args:
            data: Input DataFrame
            target_column: Name of target column
            test_size: Fraction of data for testing
            
        Returns:
            X_train, X_test, y_train, y_test arrays
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler
        
        print("Preparing training data...")
        
        # Extract features and target
        feature_columns = [col for col in data.select_dtypes(include=[np.number]).columns 
                          if col != target_column]
        
        X = data[feature_columns].values
        y = data[target_column].values
        
        # Create binary classification target (above/below median)
        y_binary = (y > np.median(y)).astype(int)
        
        # Scale features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_binary, test_size=test_size, random_state=42
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Features: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test
    
    def analyze_molecular_topology(self,
                                 model: Any,
                                 molecular_data: np.ndarray,
                                 molecular_labels: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the topological structure of molecular representations.
        
        Args:
            model: Trained neural network
            molecular_data: Input molecular feature data
            molecular_labels: Molecular activity labels
            
        Returns:
            Dictionary containing topological analysis results
        """
        print("Extracting layer activations...")
        
        # Extract activations from all layers
        activations = self.neural_extractor.extract_all_layers(model, molecular_data)
        
        print("Performing topological analysis...")
        
        # Analyze topology
        topology_results = self.topology_analyzer.analyze_network_topology(
            activations=activations,
            labels=molecular_labels,
            compute_mapper=True,
            compute_umap=True
        )
        
        # Compute additional metrics
        topology_metrics = self.topology_analyzer.compute_topology_metrics(topology_results)
        
        results = {
            'topology_results': topology_results,
            'topology_metrics': topology_metrics,
            'activations': activations,
            'layer_info': self.neural_extractor.get_layer_info(model)
        }
        
        return results
    
    def plot_molecular_evolution(self, results: Dict[str, Any],
                               save_path: str = None):
        """
        Visualize how molecular features evolve through network layers.
        
        Args:
            results: Results from analyze_molecular_topology
            save_path: Optional path to save plots
        """
        topology_results = results['topology_results']
        
        print("Creating visualization plots...")
        
        # Plot persistence diagrams
        self.topology_analyzer.plot_persistence_diagrams(
            topology_results,
            save_path=f"{save_path}_persistence.html" if save_path else None
        )
        
        # Plot Betti number evolution
        self.topology_analyzer.plot_layer_evolution(
            topology_results.betti_numbers,
            save_path=f"{save_path}_evolution.html" if save_path else None
        )
        
        # Plot UMAP projections
        self._plot_umap_evolution(
            topology_results.umap_projections,
            results['activations'].keys(),
            save_path=f"{save_path}_umap.png" if save_path else None
        )
        
        # Print summary metrics
        self._print_topology_summary(results['topology_metrics'])
    
    def _plot_umap_evolution(self, 
                           umap_projections: Dict[str, np.ndarray],
                           layer_names: list,
                           save_path: str = None):
        """Plot UMAP projections for all layers."""
        n_layers = len(umap_projections)
        if n_layers == 0:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (layer_name, projection) in enumerate(umap_projections.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            ax.scatter(projection[:, 0], projection[:, 1], alpha=0.6, s=20)
            ax.set_title(f'UMAP: {layer_name}')
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(umap_projections), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Molecular Feature Evolution Through Network Layers', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _print_topology_summary(self, metrics: Dict[str, float]):
        """Print summary of topological metrics."""
        print("\n" + "="*60)
        print("TOPOLOGICAL ANALYSIS SUMMARY")
        print("="*60)
        
        # Group metrics by layer
        layer_metrics = {}
        for metric_name, value in metrics.items():
            layer_name = metric_name.rsplit('_', 2)[0]  # Extract layer name
            metric_type = metric_name.rsplit('_', 2)[1] + '_' + metric_name.rsplit('_', 2)[2]
            
            if layer_name not in layer_metrics:
                layer_metrics[layer_name] = {}
            layer_metrics[layer_name][metric_type] = value
        
        for layer_name, layer_data in layer_metrics.items():
            print(f"\n{layer_name.upper()}:")
            print(f"  Total topological features: {layer_data.get('total_features', 0):.0f}")
            print(f"  Feature diversity: {layer_data.get('feature_diversity', 0):.0f}")
            print(f"  Persistence entropy: {layer_data.get('persistence_entropy', 0):.3f}")


def main():
    """Main function demonstrating COVID drug analysis."""
    print("COVID-19 Drug Discovery Topological Analysis")
    print("="*50)
    
    # Initialize analyzer
    analyzer = CovidDrugAnalyzer()
    
    # Load and preprocess data
    data = analyzer.load_covid_dataset('DDH_Data_with_Properties.csv')
    
    # Prepare training data
    X_train, X_test, y_train, y_test = analyzer.prepare_training_data(data)
    
    # Build and train model
    model = analyzer.build_drug_classifier(input_dim=X_train.shape[1])
    
    print("Training neural network...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=10,
        validation_data=(X_test, y_test),
        verbose=0
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.3f}")
    
    # Perform topological analysis
    results = analyzer.analyze_molecular_topology(
        model=model,
        molecular_data=X_test,
        molecular_labels=y_test
    )
    
    # Visualize results
    analyzer.plot_molecular_evolution(results, save_path="covid_analysis")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()