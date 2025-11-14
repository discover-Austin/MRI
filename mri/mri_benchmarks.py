"""
MRI Performance Benchmarking Suite
===================================

Comprehensive benchmarking against traditional ML approaches:
- Neural Networks (PyTorch/TensorFlow)
- Classical ML (scikit-learn)
- Performance metrics comparison
- Detailed analysis and reporting
"""

import numpy as np
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path


# ============================================================================
# BENCHMARK CONFIGURATION
# ============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking experiments."""
    # Dataset parameters
    n_samples: int = 1000
    n_features: int = 784  # 28x28 images
    n_classes: int = 10
    test_split: float = 0.2

    # Training parameters
    epochs: int = 10
    batch_size: int = 32

    # What to benchmark
    include_neural_networks: bool = True
    include_classical_ml: bool = True
    include_memory_tests: bool = True
    include_speed_tests: bool = True
    include_accuracy_tests: bool = True

    # Repeat runs for statistical significance
    n_runs: int = 3


# ============================================================================
# BENCHMARK RESULTS
# ============================================================================

@dataclass
class BenchmarkResults:
    """Results from a single benchmark run."""
    method_name: str
    training_time: float
    inference_time: float
    accuracy: float
    memory_usage_mb: float
    energy_consumption: Optional[float] = None
    additional_metrics: Dict[str, Any] = None


# ============================================================================
# MRI BENCHMARK ADAPTER
# ============================================================================

class MRIBenchmarkAdapter:
    """Adapter for benchmarking MRI system."""

    def __init__(self, config: BenchmarkConfig):
        from mri_production_complete import MorphicResonanceIntelligence, MRIConfig

        self.config = config

        # Initialize MRI
        field_size = int(np.sqrt(config.n_features))
        if field_size * field_size != config.n_features:
            field_size = 32  # Fallback
            self.needs_reshape = True
        else:
            self.needs_reshape = False

        self.mri_config = MRIConfig(
            field_size=(field_size, field_size),
            learning_rate=0.1,
            evolution_steps=10
        )
        self.mri = MorphicResonanceIntelligence(self.mri_config)
        self.class_phases = {}

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """Train MRI system."""
        start_time = time.time()

        # Assign phase to each class
        unique_classes = np.unique(y_train)
        for i, cls in enumerate(unique_classes):
            self.class_phases[cls] = 2 * np.pi * i / len(unique_classes)

        # Train
        for i, (sample, label) in enumerate(zip(X_train, y_train)):
            if self.needs_reshape:
                sample = sample.reshape(self.mri_config.field_size)

            self.mri.inject_pattern(
                sample,
                label=str(label),
                phase=self.class_phases[label]
            )

            if i % 100 == 0:
                self.mri.evolve_system(steps=5)

        training_time = time.time() - start_time
        return training_time

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, float]:
        """Make predictions."""
        start_time = time.time()

        predictions = []
        for sample in X_test:
            if self.needs_reshape:
                sample = sample.reshape(self.mri_config.field_size)

            # Measure resonance with each class
            resonances = {}
            for cls, phase in self.class_phases.items():
                res = self.mri.measure_resonance(sample, evolve=False)
                resonances[cls] = res

            predicted_class = max(resonances, key=resonances.get)
            predictions.append(predicted_class)

        inference_time = time.time() - start_time
        return np.array(predictions), inference_time

    def get_memory_usage(self) -> float:
        """Get memory usage in MB."""
        metrics = self.mri.get_system_metrics()
        return metrics['memory_usage_mb']


# ============================================================================
# NEURAL NETWORK BENCHMARKS
# ============================================================================

class NeuralNetworkBenchmarks:
    """Benchmark against neural networks."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def benchmark_pytorch(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray) -> BenchmarkResults:
        """Benchmark against PyTorch neural network."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import TensorDataset, DataLoader

            # Simple MLP
            class SimpleMLP(nn.Module):
                def __init__(self, input_size, hidden_size, num_classes):
                    super(SimpleMLP, self).__init__()
                    self.fc1 = nn.Linear(input_size, hidden_size)
                    self.relu = nn.ReLU()
                    self.fc2 = nn.Linear(hidden_size, num_classes)

                def forward(self, x):
                    out = self.fc1(x)
                    out = self.relu(out)
                    out = self.fc2(out)
                    return out

            # Prepare data
            X_train_tensor = torch.FloatTensor(X_train.reshape(len(X_train), -1))
            y_train_tensor = torch.LongTensor(y_train)
            X_test_tensor = torch.FloatTensor(X_test.reshape(len(X_test), -1))
            y_test_tensor = torch.LongTensor(y_test)

            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)

            # Model
            model = SimpleMLP(self.config.n_features, 128, self.config.n_classes)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Training
            start_time = time.time()
            model.train()
            for epoch in range(self.config.epochs):
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            training_time = time.time() - start_time

            # Inference
            model.eval()
            start_time = time.time()
            with torch.no_grad():
                predictions = model(X_test_tensor)
                _, predicted = torch.max(predictions.data, 1)

            inference_time = time.time() - start_time

            # Accuracy
            accuracy = (predicted == y_test_tensor).sum().item() / len(y_test)

            # Memory usage (approximate)
            memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)

            return BenchmarkResults(
                method_name="PyTorch MLP",
                training_time=training_time,
                inference_time=inference_time,
                accuracy=accuracy,
                memory_usage_mb=memory_mb,
                additional_metrics={'epochs': self.config.epochs}
            )

        except ImportError:
            print("PyTorch not installed, skipping PyTorch benchmark")
            return None

    def benchmark_simple_nn(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray) -> BenchmarkResults:
        """Benchmark against simple numpy-based neural network."""
        # Simple 2-layer neural network implementation
        def sigmoid(x):
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

        def sigmoid_derivative(x):
            return x * (1 - x)

        # Initialize weights
        np.random.seed(42)
        input_size = X_train.shape[1] if len(X_train.shape) == 2 else np.prod(X_train.shape[1:])
        hidden_size = 128
        output_size = self.config.n_classes

        weights1 = np.random.randn(input_size, hidden_size) * 0.01
        weights2 = np.random.randn(hidden_size, output_size) * 0.01

        X_train_flat = X_train.reshape(len(X_train), -1)
        X_test_flat = X_test.reshape(len(X_test), -1)

        # One-hot encode labels
        y_train_onehot = np.zeros((len(y_train), output_size))
        y_train_onehot[np.arange(len(y_train)), y_train] = 1

        # Training
        start_time = time.time()
        learning_rate = 0.01

        for epoch in range(self.config.epochs):
            # Forward pass
            hidden = sigmoid(np.dot(X_train_flat, weights1))
            output = sigmoid(np.dot(hidden, weights2))

            # Backward pass
            output_error = y_train_onehot - output
            output_delta = output_error * sigmoid_derivative(output)

            hidden_error = output_delta.dot(weights2.T)
            hidden_delta = hidden_error * sigmoid_derivative(hidden)

            # Update weights
            weights2 += hidden.T.dot(output_delta) * learning_rate
            weights1 += X_train_flat.T.dot(hidden_delta) * learning_rate

        training_time = time.time() - start_time

        # Inference
        start_time = time.time()
        hidden = sigmoid(np.dot(X_test_flat, weights1))
        output = sigmoid(np.dot(hidden, weights2))
        predictions = np.argmax(output, axis=1)
        inference_time = time.time() - start_time

        # Accuracy
        accuracy = np.mean(predictions == y_test)

        # Memory
        memory_mb = (weights1.nbytes + weights2.nbytes) / (1024**2)

        return BenchmarkResults(
            method_name="Simple Neural Network (NumPy)",
            training_time=training_time,
            inference_time=inference_time,
            accuracy=accuracy,
            memory_usage_mb=memory_mb,
            additional_metrics={'epochs': self.config.epochs}
        )


# ============================================================================
# CLASSICAL ML BENCHMARKS
# ============================================================================

class ClassicalMLBenchmarks:
    """Benchmark against classical ML algorithms."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def benchmark_svm(self, X_train, y_train, X_test, y_test) -> BenchmarkResults:
        """Benchmark against SVM."""
        try:
            from sklearn.svm import SVC

            X_train_flat = X_train.reshape(len(X_train), -1)
            X_test_flat = X_test.reshape(len(X_test), -1)

            # Training
            start_time = time.time()
            svm = SVC(kernel='rbf', gamma='scale')
            svm.fit(X_train_flat, y_train)
            training_time = time.time() - start_time

            # Inference
            start_time = time.time()
            predictions = svm.predict(X_test_flat)
            inference_time = time.time() - start_time

            # Accuracy
            accuracy = np.mean(predictions == y_test)

            # Memory (approximate)
            n_support = len(svm.support_vectors_)
            memory_mb = (n_support * X_train_flat.shape[1] * 8) / (1024**2)

            return BenchmarkResults(
                method_name="Support Vector Machine",
                training_time=training_time,
                inference_time=inference_time,
                accuracy=accuracy,
                memory_usage_mb=memory_mb,
                additional_metrics={'n_support_vectors': n_support}
            )

        except ImportError:
            print("scikit-learn not installed, skipping SVM benchmark")
            return None

    def benchmark_random_forest(self, X_train, y_train, X_test, y_test) -> BenchmarkResults:
        """Benchmark against Random Forest."""
        try:
            from sklearn.ensemble import RandomForestClassifier

            X_train_flat = X_train.reshape(len(X_train), -1)
            X_test_flat = X_test.reshape(len(X_test), -1)

            # Training
            start_time = time.time()
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train_flat, y_train)
            training_time = time.time() - start_time

            # Inference
            start_time = time.time()
            predictions = rf.predict(X_test_flat)
            inference_time = time.time() - start_time

            # Accuracy
            accuracy = np.mean(predictions == y_test)

            # Memory (rough estimate)
            n_trees = len(rf.estimators_)
            memory_mb = n_trees * 0.5  # Rough estimate

            return BenchmarkResults(
                method_name="Random Forest",
                training_time=training_time,
                inference_time=inference_time,
                accuracy=accuracy,
                memory_usage_mb=memory_mb,
                additional_metrics={'n_trees': n_trees}
            )

        except ImportError:
            print("scikit-learn not installed, skipping Random Forest benchmark")
            return None


# ============================================================================
# MAIN BENCHMARK SUITE
# ============================================================================

class MRIBenchmarkSuite:
    """Complete benchmark suite for MRI system."""

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.results: List[BenchmarkResults] = []

    def generate_synthetic_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic classification dataset."""
        np.random.seed(42)

        # Generate samples
        n_train = int(self.config.n_samples * (1 - self.config.test_split))
        n_test = self.config.n_samples - n_train

        # Create synthetic images (simplified)
        img_size = int(np.sqrt(self.config.n_features))
        if img_size * img_size != self.config.n_features:
            img_size = 28
            actual_features = img_size * img_size
        else:
            actual_features = self.config.n_features

        X_train = np.random.rand(n_train, img_size, img_size)
        X_test = np.random.rand(n_test, img_size, img_size)

        # Generate labels
        y_train = np.random.randint(0, self.config.n_classes, n_train)
        y_test = np.random.randint(0, self.config.n_classes, n_test)

        # Add some structure to make it learnable
        for i in range(len(X_train)):
            X_train[i] += y_train[i] * 0.1
        for i in range(len(X_test)):
            X_test[i] += y_test[i] * 0.1

        return X_train, y_train, X_test, y_test

    def run_complete_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        print("="*70)
        print("MRI SYSTEM COMPREHENSIVE BENCHMARK")
        print("="*70)

        # Generate dataset
        print("\nGenerating dataset...")
        X_train, y_train, X_test, y_test = self.generate_synthetic_dataset()
        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

        results_summary = {
            'config': asdict(self.config),
            'results': [],
            'comparison': {}
        }

        # Benchmark MRI
        print("\n1. Benchmarking MRI System...")
        mri_adapter = MRIBenchmarkAdapter(self.config)

        training_time = mri_adapter.train(X_train, y_train)
        predictions, inference_time = mri_adapter.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        memory_mb = mri_adapter.get_memory_usage()

        mri_results = BenchmarkResults(
            method_name="MRI (Morphic Resonance Intelligence)",
            training_time=training_time,
            inference_time=inference_time,
            accuracy=accuracy,
            memory_usage_mb=memory_mb
        )
        self.results.append(mri_results)
        results_summary['results'].append(asdict(mri_results))

        print(f"   Training time: {training_time:.3f}s")
        print(f"   Inference time: {inference_time:.3f}s")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Memory: {memory_mb:.2f} MB")

        # Benchmark Neural Networks
        if self.config.include_neural_networks:
            print("\n2. Benchmarking Neural Networks...")
            nn_benchmarks = NeuralNetworkBenchmarks(self.config)

            # Simple NN
            print("   a) Simple Neural Network...")
            nn_results = nn_benchmarks.benchmark_simple_nn(X_train, y_train, X_test, y_test)
            if nn_results:
                self.results.append(nn_results)
                results_summary['results'].append(asdict(nn_results))
                print(f"      Training: {nn_results.training_time:.3f}s, Accuracy: {nn_results.accuracy:.3f}")

            # PyTorch
            print("   b) PyTorch MLP...")
            pytorch_results = nn_benchmarks.benchmark_pytorch(X_train, y_train, X_test, y_test)
            if pytorch_results:
                self.results.append(pytorch_results)
                results_summary['results'].append(asdict(pytorch_results))
                print(f"      Training: {pytorch_results.training_time:.3f}s, Accuracy: {pytorch_results.accuracy:.3f}")

        # Benchmark Classical ML
        if self.config.include_classical_ml:
            print("\n3. Benchmarking Classical ML...")
            classical_benchmarks = ClassicalMLBenchmarks(self.config)

            # SVM
            print("   a) Support Vector Machine...")
            svm_results = classical_benchmarks.benchmark_svm(X_train, y_train, X_test, y_test)
            if svm_results:
                self.results.append(svm_results)
                results_summary['results'].append(asdict(svm_results))
                print(f"      Training: {svm_results.training_time:.3f}s, Accuracy: {svm_results.accuracy:.3f}")

            # Random Forest
            print("   b) Random Forest...")
            rf_results = classical_benchmarks.benchmark_random_forest(X_train, y_train, X_test, y_test)
            if rf_results:
                self.results.append(rf_results)
                results_summary['results'].append(asdict(rf_results))
                print(f"      Training: {rf_results.training_time:.3f}s, Accuracy: {rf_results.accuracy:.3f}")

        # Generate comparison
        results_summary['comparison'] = self._generate_comparison()

        return results_summary

    def _generate_comparison(self) -> Dict[str, Any]:
        """Generate comparative analysis."""
        if not self.results:
            return {}

        # Find best in each category
        best_training_time = min(self.results, key=lambda x: x.training_time)
        best_inference_time = min(self.results, key=lambda x: x.inference_time)
        best_accuracy = max(self.results, key=lambda x: x.accuracy)
        best_memory = min(self.results, key=lambda x: x.memory_usage_mb)

        # Calculate speedup/improvements relative to MRI
        mri_result = next((r for r in self.results if 'MRI' in r.method_name), None)

        comparison = {
            'best_training_speed': {
                'method': best_training_time.method_name,
                'time': best_training_time.training_time
            },
            'best_inference_speed': {
                'method': best_inference_time.method_name,
                'time': best_inference_time.inference_time
            },
            'best_accuracy': {
                'method': best_accuracy.method_name,
                'accuracy': best_accuracy.accuracy
            },
            'best_memory_efficiency': {
                'method': best_memory.method_name,
                'memory_mb': best_memory.memory_usage_mb
            }
        }

        if mri_result:
            comparison['mri_vs_best'] = {
                'training_speedup': best_training_time.training_time / mri_result.training_time,
                'inference_speedup': best_inference_time.inference_time / mri_result.inference_time,
                'accuracy_ratio': best_accuracy.accuracy / (mri_result.accuracy + 1e-10),
                'memory_ratio': best_memory.memory_usage_mb / (mri_result.memory_usage_mb + 1e-10)
            }

        return comparison

    def save_results(self, output_file: str = "benchmark_results.json"):
        """Save benchmark results to file."""
        results_summary = {
            'config': asdict(self.config),
            'results': [asdict(r) for r in self.results],
            'comparison': self._generate_comparison()
        }

        with open(output_file, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)

        print(f"\nResults saved to {output_file}")

    def generate_report(self, output_file: str = "benchmark_report.txt"):
        """Generate human-readable report."""
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("MRI SYSTEM BENCHMARK REPORT")
        report_lines.append("="*70)
        report_lines.append("")

        # Results table
        report_lines.append("RESULTS SUMMARY")
        report_lines.append("-"*70)
        report_lines.append(f"{'Method':<35} {'Train(s)':<10} {'Infer(s)':<10} {'Acc':<8} {'Mem(MB)':<10}")
        report_lines.append("-"*70)

        for result in self.results:
            report_lines.append(
                f"{result.method_name:<35} "
                f"{result.training_time:<10.3f} "
                f"{result.inference_time:<10.3f} "
                f"{result.accuracy:<8.3f} "
                f"{result.memory_usage_mb:<10.2f}"
            )

        report_lines.append("")

        # Comparison
        comparison = self._generate_comparison()
        if comparison:
            report_lines.append("COMPARISON")
            report_lines.append("-"*70)
            report_lines.append(f"Best Training Speed: {comparison['best_training_speed']['method']}")
            report_lines.append(f"Best Inference Speed: {comparison['best_inference_speed']['method']}")
            report_lines.append(f"Best Accuracy: {comparison['best_accuracy']['method']}")
            report_lines.append(f"Best Memory Efficiency: {comparison['best_memory_efficiency']['method']}")

        report_text = "\n".join(report_lines)

        with open(output_file, 'w') as f:
            f.write(report_text)

        print(f"\nReport saved to {output_file}")
        print("\n" + report_text)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Configure benchmark
    config = BenchmarkConfig(
        n_samples=500,  # Smaller for quick test
        n_features=784,
        n_classes=10,
        epochs=5,
        include_neural_networks=True,
        include_classical_ml=True
    )

    # Run benchmark suite
    suite = MRIBenchmarkSuite(config)
    results = suite.run_complete_benchmark()

    # Save results
    suite.save_results("mri_benchmark_results.json")
    suite.generate_report("mri_benchmark_report.txt")
