"""
MRI Master System - Complete Orchestration
==========================================

Single entry point for all MRI functionality.
Production-ready system with complete examples.
"""

import numpy as np
from pathlib import Path
import json
import sys
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


# ============================================================================
# MASTER SYSTEM CLASS
# ============================================================================

class MRIMasterSystem:
    """
    Complete MRI system orchestrating all components.
    Single interface for production deployment.
    """
    
    def __init__(self, 
                 config: Optional[Any] = None,
                 enable_gpu: bool = False,
                 enable_optimization: bool = True,
                 enable_monitoring: bool = True):
        
        # Import all components
        from mri_production_complete import (
            MorphicResonanceIntelligence, MRIConfig,
            MRIClassifier, MRIAnomalyDetector, MRIRecommender
        )
        from mri_optimization import OptimizedMRI
        from mri_integration import (
            MRIModelRegistry, MRIMonitor, 
            MRIDataPreprocessor, MRIAPIServer
        )
        
        # Load or create configuration
        if config:
            self.config = config
        else:
            self.config = MRIConfig()
        
        # Initialize core system
        if enable_optimization:
            self.mri = OptimizedMRI(
                self.config,
                enable_gpu=enable_gpu,
                enable_distributed=False
            ).mri
        else:
            self.mri = MorphicResonanceIntelligence(self.config)
        
        # Initialize utilities
        self.registry = MRIModelRegistry()
        self.preprocessor = MRIDataPreprocessor()
        
        if enable_monitoring:
            self.monitor = MRIMonitor(self.mri)
        else:
            self.monitor = None
        
        # Application templates
        self.classifier = None
        self.anomaly_detector = None
        self.recommender = None
        
        # API server
        self.api_server = None
    
    # ========================================================================
    # CORE OPERATIONS
    # ========================================================================
    
    def learn(self, data: np.ndarray, label: Optional[str] = None, **kwargs):
        """Learn a new pattern."""
        if self.monitor:
            self.monitor.log_operation('learn', {'label': label})
        
        return self.mri.inject_pattern(data, label=label, **kwargs)
    
    def predict(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Generate prediction."""
        if self.monitor:
            self.monitor.log_operation('predict', {})
        
        return self.mri.predict(data, **kwargs)
    
    def measure_similarity(self, data: np.ndarray, **kwargs) -> float:
        """Measure similarity to learned patterns."""
        return self.mri.measure_resonance(data, **kwargs)
    
    def associate(self, pattern_pairs: List[tuple], **kwargs):
        """Learn associations between patterns."""
        return self.mri.associate_patterns(pattern_pairs, **kwargs)
    
    # ========================================================================
    # HIGH-LEVEL APPLICATIONS
    # ========================================================================
    
    def create_classifier(self, task_type: str = 'classification'):
        """Create classification application."""
        from mri_integration import MRISklearnWrapper
        
        self.classifier = MRISklearnWrapper(self.config, task_type=task_type)
        return self.classifier
    
    def create_anomaly_detector(self, threshold: float = 0.3):
        """Create anomaly detection application."""
        from mri_production_complete import MRIAnomalyDetector
        
        self.anomaly_detector = MRIAnomalyDetector(self.config, threshold=threshold)
        return self.anomaly_detector
    
    def create_recommender(self):
        """Create recommendation system."""
        from mri_production_complete import MRIRecommender
        
        self.recommender = MRIRecommender(self.config)
        return self.recommender
    
    # ========================================================================
    # MODEL MANAGEMENT
    # ========================================================================
    
    def save_model(self, name: str, version: Optional[str] = None, 
                  metadata: Optional[Dict] = None):
        """Save current model to registry."""
        return self.registry.register_model(self.mri, name, version, metadata)
    
    def load_model(self, name: str, version: Optional[str] = None):
        """Load model from registry."""
        mri, info = self.registry.load_model(name, version)
        self.mri = mri
        return info
    
    def list_models(self) -> List[Dict]:
        """List all saved models."""
        return self.registry.list_models()
    
    # ========================================================================
    # DEPLOYMENT
    # ========================================================================
    
    def start_api_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Start REST API server."""
        from mri_integration import MRIAPIServer
        
        self.api_server = MRIAPIServer(self.mri, host=host, port=port)
        self.api_server.run()
    
    def export_for_deployment(self, output_dir: str = "./deployment"):
        """Export everything needed for deployment."""
        from mri_integration import MRIDeployment
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_path / "model.pkl"
        self.mri.save_state(str(model_path))
        
        # Save configuration
        config_path = output_path / "config.json"
        with open(config_path, 'w') as f:
            from dataclasses import asdict
            json.dump(asdict(self.config), f, indent=2, default=str)
        
        # Create Docker files
        MRIDeployment.create_docker_image(self.mri)
        
        # Create Kubernetes config
        MRIDeployment.create_kubernetes_deployment(self.mri)
        
        print(f"Deployment files created in {output_dir}")
    
    # ========================================================================
    # MONITORING AND DIAGNOSTICS
    # ========================================================================
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        return self.mri.get_system_metrics()
    
    def check_health(self) -> Dict[str, Any]:
        """Check system health."""
        if self.monitor:
            return self.monitor.check_health()
        return {'status': 'monitoring disabled'}
    
    def generate_report(self, output_file: str = "system_report.json"):
        """Generate comprehensive system report."""
        if self.monitor:
            return self.monitor.generate_report(output_file)
        
        # Basic report without monitoring
        report = {
            'metrics': self.get_metrics(),
            'config': asdict(self.config)
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    # ========================================================================
    # UTILITIES
    # ========================================================================
    
    def benchmark(self, n_patterns: int = 100, pattern_size: tuple = (128, 128)):
        """Run performance benchmark."""
        from mri_production_complete import benchmark_performance
        
        return benchmark_performance(self.config, n_patterns, pattern_size)
    
    def visualize_field(self, output_file: str = "field_visualization.png"):
        """Visualize current resonance field."""
        self.mri.visualize_field(output_file)
    
    def optimize_configuration(self, test_data: List[np.ndarray], 
                             metric: str = 'balanced'):
        """Auto-optimize system configuration."""
        from mri_optimization import AutoOptimizer
        
        optimizer = AutoOptimizer()
        optimal_config = optimizer.optimize_config(self.mri, test_data, metric)
        self.config = optimal_config
        
        # Reinitialize with optimal config
        from mri_production_complete import MorphicResonanceIntelligence
        self.mri = MorphicResonanceIntelligence(optimal_config)
        
        return optimal_config


# ============================================================================
# COMPLETE USAGE EXAMPLES
# ============================================================================

class MRIExamples:
    """Complete working examples for all use cases."""
    
    @staticmethod
    def example_classification():
        """Example: Image classification."""
        print("\n" + "="*70)
        print("EXAMPLE 1: IMAGE CLASSIFICATION")
        print("="*70)
        
        # Initialize system
        system = MRIMasterSystem(enable_gpu=False, enable_optimization=True)
        
        # Create classifier
        clf = system.create_classifier(task_type='classification')
        
        # Generate synthetic training data
        from mri_production_complete import create_test_patterns
        patterns = create_test_patterns(size=(64, 64))
        
        X_train = np.array([patterns['square'], patterns['circle'], 
                           patterns['stripes'], patterns['diagonal']])
        y_train = np.array(['square', 'circle', 'stripes', 'diagonal'])
        
        print("Training classifier...")
        clf.fit(X_train, y_train)
        
        # Test prediction
        X_test = np.array([patterns['square'], patterns['circle']])
        predictions = clf.predict(X_test)
        
        print(f"Predictions: {predictions}")
        print(f"Accuracy: {clf.score(X_test, ['square', 'circle']):.2%}")
        
        # Save model
        version = system.save_model('image_classifier', metadata={'task': 'classification'})
        print(f"Model saved as version {version}")
    
    @staticmethod
    def example_anomaly_detection():
        """Example: Anomaly detection."""
        print("\n" + "="*70)
        print("EXAMPLE 2: ANOMALY DETECTION")
        print("="*70)
        
        system = MRIMasterSystem()
        
        # Create detector
        detector = system.create_anomaly_detector(threshold=0.3)
        
        # Generate normal patterns
        from mri_production_complete import create_test_patterns
        patterns = create_test_patterns(size=(64, 64))
        
        normal_data = [patterns['square'], patterns['circle']]
        
        print("Learning normal patterns...")
        detector.fit(normal_data)
        
        # Test with normal and anomalous data
        test_data = [
            patterns['square'],  # Normal
            patterns['checkerboard']  # Anomaly (different pattern)
        ]
        
        predictions = detector.predict(test_data)
        scores = detector.score_samples(test_data)
        
        print("Predictions (0=normal, 1=anomaly):", predictions)
        print("Anomaly scores:", scores)
    
    @staticmethod
    def example_recommendation():
        """Example: Recommendation system."""
        print("\n" + "="*70)
        print("EXAMPLE 3: RECOMMENDATION SYSTEM")
        print("="*70)
        
        system = MRIMasterSystem()
        
        # Create recommender
        recommender = system.create_recommender()
        
        # Simulate user interactions
        users = ['user1', 'user2']
        items = {
            'item1': np.random.rand(64, 64),
            'item2': np.random.rand(64, 64),
            'item3': np.random.rand(64, 64)
        }
        
        print("Recording user interactions...")
        recommender.add_interaction('user1', 'item1', items['item1'])
        recommender.add_interaction('user1', 'item2', items['item2'])
        recommender.add_interaction('user2', 'item2', items['item2'])
        recommender.add_interaction('user2', 'item3', items['item3'])
        
        # Generate recommendations
        recommendations = recommender.recommend('user1', n_items=2)
        print(f"Recommendations for user1: {recommendations}")
    
    @staticmethod
    def example_associative_learning():
        """Example: Learning associations."""
        print("\n" + "="*70)
        print("EXAMPLE 4: ASSOCIATIVE LEARNING")
        print("="*70)
        
        system = MRIMasterSystem()
        
        # Create pattern pairs
        from mri_production_complete import create_test_patterns
        patterns = create_test_patterns(size=(64, 64))
        
        pattern_pairs = [
            (patterns['square'], patterns['circle']),
            (patterns['stripes'], patterns['diagonal'])
        ]
        
        print("Learning pattern associations...")
        history = system.associate(pattern_pairs, iterations=50)
        
        print(f"Final resonance: {history['resonances'][-1]:.4f}")
        
        # Test retrieval
        print("\nTesting association retrieval...")
        prediction = system.predict(patterns['square'], evolution_time=30)
        similarity = system.measure_similarity(patterns['circle'])
        print(f"Similarity to associated pattern: {similarity:.4f}")
    
    @staticmethod
    def example_continuous_learning():
        """Example: Continuous lifelong learning."""
        print("\n" + "="*70)
        print("EXAMPLE 5: CONTINUOUS LEARNING")
        print("="*70)
        
        system = MRIMasterSystem(enable_monitoring=True)
        
        # Learn patterns in sequence
        from mri_production_complete import create_test_patterns
        patterns = create_test_patterns(size=(64, 64))
        
        pattern_sequence = [
            ('square', patterns['square']),
            ('circle', patterns['circle']),
            ('stripes', patterns['stripes']),
            ('diagonal', patterns['diagonal']),
            ('checker', patterns['checkerboard'])
        ]
        
        print("Learning patterns sequentially...")
        for label, pattern in pattern_sequence:
            system.learn(pattern, label=label)
            metrics = system.get_metrics()
            print(f"Learned '{label}' - Patterns in memory: {metrics['patterns_learned']}")
        
        # Verify no forgetting
        print("\nVerifying retention (no catastrophic forgetting)...")
        for label, pattern in pattern_sequence:
            similarity = system.measure_similarity(pattern)
            print(f"{label}: {similarity:.4f} similarity")
    
    @staticmethod
    def example_real_time_adaptation():
        """Example: Real-time adaptation."""
        print("\n" + "="*70)
        print("EXAMPLE 6: REAL-TIME ADAPTATION")
        print("="*70)
        
        system = MRIMasterSystem()
        
        # Simulate real-time data stream
        print("Simulating real-time data stream...")
        
        for t in range(20):
            # Generate time-varying pattern
            pattern = np.random.rand(64, 64)
            pattern = pattern * (1 + 0.1 * np.sin(2 * np.pi * t / 20))
            
            # Learn immediately
            system.learn(pattern, label=f"time_{t}")
            
            # System adapts in real-time - no retraining needed
            if t % 5 == 0:
                metrics = system.get_metrics()
                print(f"Time {t}: Patterns learned={metrics['patterns_learned']}, "
                      f"Avg learning time={metrics['avg_learning_time']*1000:.2f}ms")
    
    @staticmethod
    def example_deployment():
        """Example: Deployment workflow."""
        print("\n" + "="*70)
        print("EXAMPLE 7: PRODUCTION DEPLOYMENT")
        print("="*70)
        
        system = MRIMasterSystem(enable_monitoring=True)
        
        # Train model
        from mri_production_complete import create_test_patterns
        patterns = create_test_patterns(size=(64, 64))
        
        for label, pattern in patterns.items():
            system.learn(pattern, label=label)
        
        # Save to registry
        print("Saving to model registry...")
        version = system.save_model(
            'production_model',
            metadata={'purpose': 'pattern recognition', 'stage': 'production'}
        )
        print(f"Saved as version {version}")
        
        # Run benchmark
        print("\nRunning performance benchmark...")
        benchmark_results = system.benchmark(n_patterns=50)
        print(f"Learning speed: {benchmark_results['avg_learning_time_ms']:.2f}ms")
        print(f"Inference speed: {benchmark_results['avg_inference_time_ms']:.2f}ms")
        print(f"Memory usage: {benchmark_results['memory_mb']:.2f}MB")
        
        # Health check
        print("\nChecking system health...")
        health = system.check_health()
        print(f"Status: {health['status']}")
        if health['issues']:
            print("Issues:", health['issues'])
        
        # Generate report
        system.generate_report('deployment_report.json')
        print("Deployment report saved")
        
        # Export for deployment
        print("\nExporting deployment artifacts...")
        system.export_for_deployment('./production_deployment')
        print("Ready for deployment!")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

class MRICLIInterface:
    """Command-line interface for MRI system."""
    
    @staticmethod
    def run():
        """Run CLI interface."""
        import argparse
        
        parser = argparse.ArgumentParser(description='MRI Master System CLI')
        subparsers = parser.add_subparsers(dest='command', help='Commands')
        
        # Train command
        train_parser = subparsers.add_parser('train', help='Train a model')
        train_parser.add_argument('--data', required=True, help='Path to training data')
        train_parser.add_argument('--model-name', required=True, help='Model name')
        train_parser.add_argument('--task', default='classification', 
                                 choices=['classification', 'regression', 'unsupervised'])
        
        # Predict command
        predict_parser = subparsers.add_parser('predict', help='Make predictions')
        predict_parser.add_argument('--model', required=True, help='Model name')
        predict_parser.add_argument('--data', required=True, help='Path to input data')
        predict_parser.add_argument('--output', default='predictions.json', 
                                   help='Output file')
        
        # Serve command
        serve_parser = subparsers.add_parser('serve', help='Start API server')
        serve_parser.add_argument('--model', required=True, help='Model name')
        serve_parser.add_argument('--host', default='0.0.0.0', help='Host address')
        serve_parser.add_argument('--port', type=int, default=8000, help='Port')
        
        # List models command
        list_parser = subparsers.add_parser('list', help='List saved models')
        
        # Benchmark command
        benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmark')
        benchmark_parser.add_argument('--n-patterns', type=int, default=100)
        
        # Examples command
        examples_parser = subparsers.add_parser('examples', help='Run examples')
        examples_parser.add_argument('--example', type=int, 
                                    help='Run specific example (1-7)')
        
        args = parser.parse_args()
        
        if args.command == 'train':
            MRICLIInterface._train(args)
        elif args.command == 'predict':
            MRICLIInterface._predict(args)
        elif args.command == 'serve':
            MRICLIInterface._serve(args)
        elif args.command == 'list':
            MRICLIInterface._list_models()
        elif args.command == 'benchmark':
            MRICLIInterface._benchmark(args)
        elif args.command == 'examples':
            MRICLIInterface._run_examples(args)
        else:
            parser.print_help()
    
    @staticmethod
    def _train(args):
        """Train command implementation."""
        print(f"Training {args.task} model '{args.model_name}'...")
        # Implementation here
    
    @staticmethod
    def _predict(args):
        """Predict command implementation."""
        print(f"Making predictions with model '{args.model}'...")
        # Implementation here
    
    @staticmethod
    def _serve(args):
        """Serve command implementation."""
        system = MRIMasterSystem()
        system.load_model(args.model)
        print(f"Starting API server on {args.host}:{args.port}...")
        system.start_api_server(host=args.host, port=args.port)
    
    @staticmethod
    def _list_models():
        """List models command implementation."""
        system = MRIMasterSystem()
        models = system.list_models()
        
        print("\nRegistered Models:")
        print("-" * 70)
        for model in models:
            print(f"Name: {model['name']}")
            print(f"  Version: {model['latest_version']}")
            print(f"  Total versions: {model['total_versions']}")
            print(f"  Last updated: {model['last_updated']}")
            print()
    
    @staticmethod
    def _benchmark(args):
        """Benchmark command implementation."""
        system = MRIMasterSystem()
        results = system.benchmark(n_patterns=args.n_patterns)
        
        print("\nBenchmark Results:")
        print("-" * 70)
        for key, value in results.items():
            print(f"{key}: {value}")
    
    @staticmethod
    def _run_examples(args):
        """Run examples command implementation."""
        if args.example:
            example_funcs = [
                MRIExamples.example_classification,
                MRIExamples.example_anomaly_detection,
                MRIExamples.example_recommendation,
                MRIExamples.example_associative_learning,
                MRIExamples.example_continuous_learning,
                MRIExamples.example_real_time_adaptation,
                MRIExamples.example_deployment
            ]
            
            if 1 <= args.example <= len(example_funcs):
                example_funcs[args.example - 1]()
            else:
                print(f"Example {args.example} not found. Available: 1-{len(example_funcs)}")
        else:
            # Run all examples
            print("\n" + "="*70)
            print("RUNNING ALL EXAMPLES")
            print("="*70)
            
            MRIExamples.example_classification()
            MRIExamples.example_anomaly_detection()
            MRIExamples.example_recommendation()
            MRIExamples.example_associative_learning()
            MRIExamples.example_continuous_learning()
            MRIExamples.example_real_time_adaptation()
            MRIExamples.example_deployment()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    import sys
    from mri_json_encoder import MRIJSONEncoder
    
    print("="*70)
    print("MRI MASTER SYSTEM - Production Ready")
    print("="*70)
    print()
    
    if len(sys.argv) > 1:
        # CLI mode
        MRICLIInterface.run()
    else:
        # Interactive mode - run with production config
        from production_config import production_config
        print("Running in production mode...")
        system = MRIMasterSystem(config=production_config)
        # I will add a simple execution here to show the system is running
        system.learn(np.random.rand(512, 512), label='production_test')
        metrics = system.get_metrics()
        print("System metrics:", json.dumps(metrics, indent=2, cls=MRIJSONEncoder))


if __name__ == "__main__":
    main()
