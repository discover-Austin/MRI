"""
MRI Integration Framework and Deployment Tools
==============================================

Complete integration with standard ML frameworks,
deployment utilities, and production application templates.
"""

import numpy as np
from typing import Optional, List, Dict, Any, Tuple, Callable
from pathlib import Path
import json
import pickle
from dataclasses import dataclass, asdict
import logging
import time
from mri_json_encoder import MRIJSONEncoder


# ============================================================================
# STANDARD ML FRAMEWORK INTEGRATION
# ============================================================================

class MRISklearnWrapper:
    """Scikit-learn compatible wrapper for MRI."""
    
    def __init__(self, config=None, task_type: str = 'classification'):
        from mri_production_complete import MorphicResonanceIntelligence, MRIConfig
        
        self.config = config or MRIConfig()
        self.mri = MorphicResonanceIntelligence(self.config)
        self.task_type = task_type
        self.classes_ = None
        self.n_classes_ = 0
        self.is_fitted_ = False
    
    def fit(self, X, y=None):
        """Scikit-learn compatible fit method."""
        X = np.array(X)
        
        if self.task_type == 'classification':
            if y is None:
                raise ValueError("y required for classification")
            
            y = np.array(y)
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
            
            # Learn patterns with class labels
            for i, (sample, label) in enumerate(zip(X, y)):
                phase = 2 * np.pi * np.where(self.classes_ == label)[0][0] / self.n_classes_
                
                # Reshape if needed
                if sample.ndim == 1:
                    side = int(np.sqrt(len(sample)))
                    if side * side == len(sample):
                        sample = sample.reshape(side, side)
                    else:
                        sample = sample.reshape(1, -1)
                
                self.mri.inject_pattern(sample, label=str(label), phase=phase)
                
                if i % 10 == 0:
                    self.mri.evolve_system(steps=5)
        
        elif self.task_type == 'regression':
            # For regression, learn input-output associations
            for i, (sample, target) in enumerate(zip(X, y)):
                if sample.ndim == 1:
                    side = int(np.sqrt(len(sample)))
                    if side * side == len(sample):
                        sample = sample.reshape(side, side)
                    else:
                        sample = sample.reshape(1, -1)
                
                # Create target pattern
                target_pattern = np.full_like(sample, target)
                
                # Learn association
                phase = 2 * np.pi * i / len(X)
                self.mri.inject_pattern(sample, phase=phase)
                self.mri.inject_pattern(target_pattern, phase=phase)
                
                if i % 10 == 0:
                    self.mri.evolve_system(steps=5)
        
        else:  # unsupervised
            for i, sample in enumerate(X):
                if sample.ndim == 1:
                    side = int(np.sqrt(len(sample)))
                    if side * side == len(sample):
                        sample = sample.reshape(side, side)
                    else:
                        sample = sample.reshape(1, -1)
                
                self.mri.inject_pattern(sample)
                
                if i % 10 == 0:
                    self.mri.evolve_system(steps=5)
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """Scikit-learn compatible predict method."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted yet")
        
        X = np.array(X)
        predictions = []
        
        if self.task_type == 'classification':
            for sample in X:
                if sample.ndim == 1:
                    side = int(np.sqrt(len(sample)))
                    if side * side == len(sample):
                        sample = sample.reshape(side, side)
                    else:
                        sample = sample.reshape(1, -1)
                
                # Measure resonance with each class
                resonances = {}
                for cls in self.classes_:
                    res = self.mri.measure_resonance(sample, evolve=False)
                    resonances[cls] = res
                
                predictions.append(max(resonances, key=resonances.get))
        
        elif self.task_type == 'regression':
            for sample in X:
                if sample.ndim == 1:
                    side = int(np.sqrt(len(sample)))
                    if side * side == len(sample):
                        sample = sample.reshape(side, side)
                    else:
                        sample = sample.reshape(1, -1)
                
                prediction = self.mri.predict(sample, evolution_time=30)
                predictions.append(np.mean(prediction))
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        if not self.is_fitted_:
            raise ValueError("Model not fitted yet")
        
        X = np.array(X)
        probabilities = []
        
        for sample in X:
            if sample.ndim == 1:
                side = int(np.sqrt(len(sample)))
                if side * side == len(sample):
                    sample = sample.reshape(side, side)
                else:
                    sample = sample.reshape(1, -1)
            
            resonances = {}
            for cls in self.classes_:
                res = self.mri.measure_resonance(sample, evolve=False)
                resonances[cls] = res
            
            total = sum(resonances.values())
            probs = [resonances[cls] / total for cls in self.classes_]
            probabilities.append(probs)
        
        return np.array(probabilities)
    
    def score(self, X, y):
        """Return accuracy score."""
        predictions = self.predict(X)
        return np.mean(predictions == y)


class MRIPyTorchModule:
    """PyTorch compatible module wrapper for MRI."""
    
    def __init__(self, config=None):
        from mri_production_complete import MorphicResonanceIntelligence, MRIConfig
        
        self.config = config or MRIConfig()
        self.mri = MorphicResonanceIntelligence(self.config)
    
    def forward(self, x):
        """Forward pass compatible with PyTorch."""
        # Convert tensor to numpy
        if hasattr(x, 'detach'):
            x = x.detach().cpu().numpy()
        
        # Process batch
        batch_size = x.shape[0]
        outputs = []
        
        for i in range(batch_size):
            sample = x[i]
            
            # Reshape if needed
            if sample.ndim == 1:
                side = int(np.sqrt(len(sample)))
                if side * side == len(sample):
                    sample = sample.reshape(side, side)
            
            output = self.mri.predict(sample, evolution_time=20)
            outputs.append(output.flatten())
        
        return np.array(outputs)
    
    def train_step(self, x_batch, y_batch):
        """Training step compatible with PyTorch training loops."""
        if hasattr(x_batch, 'detach'):
            x_batch = x_batch.detach().cpu().numpy()
        if hasattr(y_batch, 'detach'):
            y_batch = y_batch.detach().cpu().numpy()
        
        losses = []
        for x, y in zip(x_batch, y_batch):
            if x.ndim == 1:
                side = int(np.sqrt(len(x)))
                if side * side == len(x):
                    x = x.reshape(side, side)
            
            # Learn association
            self.mri.inject_pattern(x)
            
            # Measure loss (inverse resonance)
            prediction = self.mri.predict(x, evolution_time=10)
            if y.ndim == x.ndim:
                loss = 1.0 - self.mri.measure_resonance(y, evolve=False)
            else:
                loss = np.mean((prediction.flatten() - y.flatten())**2)
            
            losses.append(loss)
        
        return np.mean(losses)


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

class MRIDataPreprocessor:
    """Preprocessing utilities for MRI system."""
    
    @staticmethod
    def normalize_images(images: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """Normalize image data."""
        if method == 'minmax':
            return (images - images.min()) / (images.max() - images.min() + 1e-10)
        elif method == 'zscore':
            return (images - images.mean()) / (images.std() + 1e-10)
        elif method == 'l2':
            norms = np.linalg.norm(images.reshape(images.shape[0], -1), axis=1, keepdims=True)
            return images / (norms.reshape(-1, 1, 1) + 1e-10)
        return images
    
    @staticmethod
    def augment_images(images: np.ndarray, 
                      rotation: bool = True,
                      flip: bool = True,
                      noise: bool = True) -> List[np.ndarray]:
        """Data augmentation for images."""
        from scipy.ndimage import rotate
        
        augmented = [images]
        
        if rotation:
            for angle in [90, 180, 270]:
                rotated = rotate(images, angle, axes=(1, 2), reshape=False)
                augmented.append(rotated)
        
        if flip:
            flipped_h = np.flip(images, axis=1)
            flipped_v = np.flip(images, axis=2)
            augmented.extend([flipped_h, flipped_v])
        
        if noise:
            noisy = images + np.random.randn(*images.shape) * 0.01
            augmented.append(noisy)
        
        return augmented
    
    @staticmethod
    def encode_text_to_field(text: str, vocab_size: int = 50000,
                           field_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
        """Encode text to resonance field pattern."""
        # Simple hash-based encoding
        # In production, use proper tokenization
        
        field = np.zeros(field_size, dtype=complex)
        
        words = text.split()
        for i, word in enumerate(words):
            # Hash word to frequency components
            hash_val = hash(word) % (field_size[0] * field_size[1])
            x = hash_val // field_size[1]
            y = hash_val % field_size[1]
            
            # Add with phase based on position
            phase = 2 * np.pi * i / len(words)
            field[x, y] += np.exp(1j * phase)
        
        return field
    
    @staticmethod
    def encode_time_series(series: np.ndarray,
                          field_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
        """Encode time series to resonance field."""
        # Reshape time series to 2D pattern
        if len(series) < np.prod(field_size):
            # Pad with zeros
            padded = np.zeros(np.prod(field_size))
            padded[:len(series)] = series
            pattern = padded.reshape(field_size)
        else:
            # Downsample or crop
            pattern = series[:np.prod(field_size)].reshape(field_size)
        
        # Add frequency information
        from scipy.fft import fft
        freq_components = fft(series)
        
        # Encode as phase
        phase_pattern = np.angle(freq_components[:np.prod(field_size)]).reshape(field_size)
        
        return pattern * np.exp(1j * phase_pattern)


# ============================================================================
# MODEL PERSISTENCE AND VERSIONING
# ============================================================================

class MRIModelRegistry:
    """Model versioning and registry system."""
    
    def __init__(self, registry_path: str = "./mri_models"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_path / "registry.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Load registry metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {'models': {}, 'versions': {}}
    
    def _save_metadata(self):
        """Save registry metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, cls=MRIJSONEncoder)
    
    def register_model(self, mri, model_name: str, 
                      version: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None):
        """Register a trained MRI model."""
        import time
        from datetime import datetime
        
        if version is None:
            # Auto-generate version
            if model_name in self.metadata['versions']:
                last_version = self.metadata['versions'][model_name]
                major, minor = map(int, last_version.split('.'))
                version = f"{major}.{minor + 1}"
            else:
                version = "1.0"
        
        # Save model
        model_path = self.registry_path / f"{model_name}_v{version}.pkl"
        mri.save_state(str(model_path))
        
        # Update metadata
        model_info = {
            'name': model_name,
            'version': version,
            'path': str(model_path),
            'timestamp': datetime.now().isoformat(),
            'config': asdict(mri.config),
            'metrics': mri.get_system_metrics(),
            'custom_metadata': metadata or {}
        }
        
        if model_name not in self.metadata['models']:
            self.metadata['models'][model_name] = []
        
        self.metadata['models'][model_name].append(model_info)
        self.metadata['versions'][model_name] = version
        
        self._save_metadata()
        
        return version
    
    def load_model(self, model_name: str, version: Optional[str] = None):
        """Load a registered model."""
        from mri_production_complete import MorphicResonanceIntelligence, MRIConfig
        
        if model_name not in self.metadata['models']:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        # Get version
        if version is None:
            version = self.metadata['versions'][model_name]
        
        # Find model info
        model_info = None
        for info in self.metadata['models'][model_name]:
            if info['version'] == version:
                model_info = info
                break
        
        if model_info is None:
            raise ValueError(f"Version {version} of model '{model_name}' not found")
        
        # Load model
        config = MRIConfig(**model_info['config'])
        mri = MorphicResonanceIntelligence(config)
        mri.load_state(model_info['path'])
        
        return mri, model_info
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        models = []
        for name, versions in self.metadata['models'].items():
            latest = versions[-1]
            models.append({
                'name': name,
                'latest_version': latest['version'],
                'total_versions': len(versions),
                'last_updated': latest['timestamp']
            })
        return models


# ============================================================================
# REST API SERVER
# ============================================================================

class MRIAPIServer:
    """REST API server for MRI deployment."""
    
    def __init__(self, mri, host: str = "0.0.0.0", port: int = 8000):
        self.mri = mri
        self.host = host
        self.port = port
        self.app = None
    
    def create_app(self):
        """Create Flask/FastAPI application."""
        try:
            from flask import Flask, request, jsonify
            self.app = Flask(__name__)
            self._setup_flask_routes()
        except ImportError:
            try:
                from fastapi import FastAPI
                self.app = FastAPI()
                self._setup_fastapi_routes()
            except ImportError:
                raise ImportError("Install flask or fastapi for API server")
    
    def _setup_flask_routes(self):
        """Setup Flask routes."""
        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({'status': 'healthy', 'system': 'MRI'})
        
        @self.app.route('/inject', methods=['POST'])
        def inject():
            data = request.json
            pattern = np.array(data['pattern'])
            label = data.get('label')
            phase = data.get('phase', 0.0)
            
            result = self.mri.inject_pattern(pattern, label=label, phase=phase)
            
            return jsonify({
                'success': True,
                'result': {k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) 
                          else v for k, v in result.items()}
            })
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            data = request.json
            pattern = np.array(data['pattern'])
            evolution_time = data.get('evolution_time', 50)
            
            prediction = self.mri.predict(pattern, evolution_time=evolution_time)
            
            return jsonify({
                'success': True,
                'prediction': prediction.tolist()
            })
        
        @self.app.route('/measure', methods=['POST'])
        def measure():
            data = request.json
            pattern = np.array(data['pattern'])
            evolve = data.get('evolve', True)
            
            resonance = self.mri.measure_resonance(pattern, evolve=evolve)
            
            return jsonify({
                'success': True,
                'resonance': float(resonance)
            })
        
        @self.app.route('/metrics', methods=['GET'])
        def metrics():
            return jsonify(self.mri.get_system_metrics())
    
    def run(self):
        """Run the server."""
        if self.app is None:
            self.create_app()
        
        self.app.run(host=self.host, port=self.port)


# ============================================================================
# DEPLOYMENT UTILITIES
# ============================================================================

class MRIDeployment:
    """Deployment utilities for production."""
    
    @staticmethod
    def export_to_onnx(mri, output_path: str):
        """Export MRI model to ONNX format (approximation)."""
        # This is a template - actual ONNX export would require
        # converting the resonance field dynamics to neural network ops
        # or using custom ONNX operators
        
        logging.warning("ONNX export creates approximation - not exact MRI dynamics")
        
        # Could create a surrogate neural network that approximates MRI behavior
        # For specific learned patterns
        pass
    
    @staticmethod
    def create_docker_image(mri, image_name: str = "mri-system"):
        """Create Docker image for deployment."""
        dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy MRI system
COPY mri_production_complete.py .
COPY mri_optimization.py .
COPY mri_integration.py .
COPY model.pkl .

# Expose API port
EXPOSE 8000

# Run server
CMD ["python", "-c", "from mri_integration import MRIAPIServer; from mri_production_complete import MorphicResonanceIntelligence; mri = MorphicResonanceIntelligence(); mri.load_state('model.pkl'); server = MRIAPIServer(mri); server.run()"]
"""
        
        requirements_content = """
numpy>=1.21.0
scipy>=1.7.0
"""
        
        # Save files
        with open("Dockerfile", 'w') as f:
            f.write(dockerfile_content)
        
        with open("requirements.txt", 'w') as f:
            f.write(requirements_content)
        
        # Save model
        mri.save_state("model.pkl")
        
        print(f"Docker configuration created. Build with: docker build -t {image_name} .")
    
    @staticmethod
    def create_kubernetes_deployment(mri, deployment_name: str = "mri-deployment"):
        """Create Kubernetes deployment configuration."""
        k8s_config = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {deployment_name}
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mri-system
  template:
    metadata:
      labels:
        app: mri-system
    spec:
      containers:
      - name: mri-container
        image: mri-system:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: mri-service
spec:
  selector:
    app: mri-system
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
"""
        
        with open("mri-deployment.yaml", 'w') as f:
            f.write(k8s_config)
        
        print("Kubernetes configuration created: mri-deployment.yaml")


# ============================================================================
# MONITORING AND LOGGING
# ============================================================================

class MRIMonitor:
    """Production monitoring for MRI systems."""
    
    def __init__(self, mri, log_file: str = "mri_monitor.log"):
        self.mri = mri
        self.logger = self._setup_logger(log_file)
        self.metrics_history: List[Dict[str, Any]] = []
    
    def _setup_logger(self, log_file: str):
        """Setup logging."""
        logger = logging.getLogger('MRI_Monitor')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)
        
        return logger
    
    def log_operation(self, operation: str, details: Dict[str, Any]):
        """Log system operation."""
        self.logger.info(f"{operation}: {json.dumps(details)}")
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        metrics = self.mri.get_system_metrics()
        metrics['timestamp'] = time.time()
        self.metrics_history.append(metrics)
        
        return metrics
    
    def check_health(self) -> Dict[str, Any]:
        """Check system health."""
        metrics = self.mri.get_system_metrics()
        
        health_status = {
            'status': 'healthy',
            'issues': []
        }
        
        # Check for issues
        if metrics['field_statistics']['energy'] > 1e6:
            health_status['issues'].append('High field energy - may need normalization')
            health_status['status'] = 'warning'
        
        if metrics['avg_learning_time'] > 1.0:
            health_status['issues'].append('Slow learning - consider optimization')
            health_status['status'] = 'warning'
        
        if metrics['memory_usage_mb'] > 1000:
            health_status['issues'].append('High memory usage - consider compression')
            health_status['status'] = 'warning'
        
        return health_status
    
    def generate_report(self, output_file: str = "mri_report.json"):
        """Generate monitoring report."""
        report = {
            'system_metrics': self.mri.get_system_metrics(),
            'health_status': self.check_health(),
            'metrics_history': self.metrics_history[-100:],  # Last 100 samples
            'timestamp': time.time()
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report


# ============================================================================
# COMPLETE APPLICATION TEMPLATES
# ============================================================================

class ImageClassificationApp:
    """Complete image classification application."""
    
    def __init__(self, config=None):
        from mri_production_complete import MorphicResonanceIntelligence, MRIConfig
        
        self.config = config or MRIConfig(field_size=(128, 128))
        self.mri = MorphicResonanceIntelligence(self.config)
        self.preprocessor = MRIDataPreprocessor()
        self.classes = {}
    
    def train(self, images: np.ndarray, labels: np.ndarray, 
             augment: bool = True, epochs: int = 1):
        """Train on image dataset."""
        # Normalize
        images = self.preprocessor.normalize_images(images)
        
        # Augment if requested
        if augment:
            augmented_sets = self.preprocessor.augment_images(images)
            images = np.concatenate(augmented_sets, axis=0)
            labels = np.tile(labels, len(augmented_sets))
        
        # Store classes
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            self.classes[label] = 2 * np.pi * i / len(unique_labels)
        
        # Training loop
        for epoch in range(epochs):
            for img, label in zip(images, labels):
                self.mri.inject_pattern(
                    img,
                    label=str(label),
                    phase=self.classes[label]
                )
            
            self.mri.evolve_system(steps=20)
    
    def predict(self, images: np.ndarray) -> np.ndarray:
        """Predict classes for images."""
        images = self.preprocessor.normalize_images(images)
        
        predictions = []
        for img in images:
            resonances = {}
            for label, phase in self.classes.items():
                res = self.mri.measure_resonance(img, evolve=False)
                resonances[label] = res
            
            predictions.append(max(resonances, key=resonances.get))
        
        return np.array(predictions)
    
    def save(self, path: str):
        """Save trained model."""
        state = {
            'mri': self.mri,
            'classes': self.classes,
            'config': self.config
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, path: str):
        """Load trained model."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.mri = state['mri']
        self.classes = state['classes']
        self.config = state['config']


if __name__ == "__main__":
    print("MRI Integration Framework - Production Ready")
    print("Components:")
    print("  - Scikit-learn wrapper")
    print("  - PyTorch integration")
    print("  - REST API server")
    print("  - Model registry")
    print("  - Deployment tools")
    print("  - Monitoring system")
    print("  - Application templates")
