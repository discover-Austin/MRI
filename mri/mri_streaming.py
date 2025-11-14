"""
MRI Streaming Data Processing
==============================

Real-time and streaming data processing capabilities for MRI system.
Handles continuous data streams, time series, and online learning scenarios.
"""

import numpy as np
from typing import Optional, Callable, Dict, Any, List, Iterator
from collections import deque
import time
from threading import Thread, Lock
from queue import Queue, Empty
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# STREAMING DATA PROCESSOR
# ============================================================================

class StreamingDataProcessor:
    """Process streaming data in real-time with MRI."""

    def __init__(self, mri_system, buffer_size: int = 1000,
                 batch_size: int = 10, auto_evolve: bool = True):
        self.mri = mri_system
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.auto_evolve = auto_evolve

        # Streaming buffers
        self.data_buffer = deque(maxlen=buffer_size)
        self.label_buffer = deque(maxlen=buffer_size)

        # Statistics
        self.samples_processed = 0
        self.processing_times = deque(maxlen=1000)
        self.accuracy_history = deque(maxlen=1000)

        # Threading
        self.processing_queue = Queue()
        self.is_running = False
        self.lock = Lock()

    def process_stream(self, data: np.ndarray, label: Optional[str] = None,
                      online_validation: bool = False) -> Dict[str, Any]:
        """Process a single data point from stream."""
        start_time = time.time()

        with self.lock:
            # Add to buffer
            self.data_buffer.append(data)
            if label is not None:
                self.label_buffer.append(label)

            # Learn from data
            result = self.mri.inject_pattern(data, label=label)

            # Batch evolution
            if self.auto_evolve and len(self.data_buffer) % self.batch_size == 0:
                self.mri.evolve_system(steps=5)

            # Online validation
            accuracy = None
            if online_validation and len(self.data_buffer) >= 10:
                # Test on recent samples
                recent_data = list(self.data_buffer)[-10:]
                recent_labels = list(self.label_buffer)[-10:] if self.label_buffer else None

                if recent_labels:
                    correct = 0
                    for test_data, test_label in zip(recent_data, recent_labels):
                        resonance = self.mri.measure_resonance(test_data, evolve=False)
                        # Simple accuracy check
                        if resonance > 0.5:
                            correct += 1
                    accuracy = correct / len(recent_data)
                    self.accuracy_history.append(accuracy)

            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.samples_processed += 1

            return {
                'sample_id': self.samples_processed,
                'processing_time': processing_time,
                'accuracy': accuracy,
                'buffer_size': len(self.data_buffer)
            }

    def start_async_processing(self, callback: Optional[Callable] = None):
        """Start asynchronous stream processing."""
        self.is_running = True

        def worker():
            while self.is_running:
                try:
                    data, label = self.processing_queue.get(timeout=0.1)
                    result = self.process_stream(data, label, online_validation=True)

                    if callback:
                        callback(result)

                except Empty:
                    continue
                except Exception as e:
                    print(f"Error in stream processing: {e}")

        self.worker_thread = Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def enqueue_data(self, data: np.ndarray, label: Optional[str] = None):
        """Add data to processing queue for async processing."""
        self.processing_queue.put((data, label))

    def stop_async_processing(self):
        """Stop asynchronous processing."""
        self.is_running = False
        if hasattr(self, 'worker_thread'):
            self.worker_thread.join(timeout=1.0)

    def get_statistics(self) -> Dict[str, Any]:
        """Get streaming processing statistics."""
        return {
            'samples_processed': self.samples_processed,
            'buffer_utilization': len(self.data_buffer) / self.buffer_size,
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'throughput_samples_per_sec': 1.0 / (np.mean(self.processing_times) + 1e-10) if self.processing_times else 0,
            'recent_accuracy': np.mean(list(self.accuracy_history)[-100:]) if self.accuracy_history else None,
            'queue_size': self.processing_queue.qsize()
        }


# ============================================================================
# TIME SERIES PROCESSOR
# ============================================================================

class TimeSeriesProcessor:
    """Specialized processor for time series data."""

    def __init__(self, mri_system, window_size: int = 50,
                 stride: int = 1, aggregation: str = 'mean'):
        self.mri = mri_system
        self.window_size = window_size
        self.stride = stride
        self.aggregation = aggregation

        self.time_series_buffer = deque(maxlen=window_size * 10)
        self.timestamps = deque(maxlen=window_size * 10)

    def process_time_series(self, time_series: np.ndarray,
                           timestamps: Optional[np.ndarray] = None,
                           online: bool = True) -> Dict[str, Any]:
        """Process time series data."""
        # Add to buffer
        for i, value in enumerate(time_series):
            self.time_series_buffer.append(value)
            if timestamps is not None:
                self.timestamps.append(timestamps[i])
            else:
                self.timestamps.append(time.time())

        # Extract windows
        windows = self._extract_windows()

        if online:
            # Process most recent window
            if len(windows) > 0:
                recent_window = windows[-1]
                window_field = self._window_to_field(recent_window)

                # Learn pattern
                self.mri.inject_pattern(window_field)

                # Detect anomalies
                resonance = self.mri.measure_resonance(window_field, evolve=False)
                is_anomaly = resonance < 0.3

                return {
                    'window': recent_window,
                    'resonance': resonance,
                    'is_anomaly': is_anomaly,
                    'timestamp': self.timestamps[-1] if self.timestamps else None
                }

        return {'windows_extracted': len(windows)}

    def _extract_windows(self) -> List[np.ndarray]:
        """Extract sliding windows from time series."""
        if len(self.time_series_buffer) < self.window_size:
            return []

        windows = []
        data = np.array(self.time_series_buffer)

        for i in range(0, len(data) - self.window_size + 1, self.stride):
            window = data[i:i + self.window_size]
            windows.append(window)

        return windows

    def _window_to_field(self, window: np.ndarray) -> np.ndarray:
        """Convert time series window to field representation."""
        # Simple approach: reshape to 2D
        field_size = self.mri.config.field_size

        if len(window) < np.prod(field_size):
            # Pad
            padded = np.zeros(np.prod(field_size))
            padded[:len(window)] = window
            window = padded
        else:
            # Truncate
            window = window[:np.prod(field_size)]

        field = window.reshape(field_size)

        return field

    def forecast(self, n_steps: int = 10) -> np.ndarray:
        """Forecast future values."""
        if len(self.time_series_buffer) < self.window_size:
            raise ValueError("Insufficient data for forecasting")

        # Use recent window for prediction
        recent_data = np.array(list(self.time_series_buffer)[-self.window_size:])
        window_field = self._window_to_field(recent_data)

        # Predict future
        prediction_field = self.mri.predict(window_field, evolution_time=30)

        # Extract forecast values
        forecast = prediction_field.flatten()[:n_steps]

        return forecast


# ============================================================================
# STREAMING ANOMALY DETECTOR
# ============================================================================

class StreamingAnomalyDetector:
    """Real-time anomaly detection on streaming data."""

    def __init__(self, mri_system, threshold: float = 0.3,
                 adaptation_rate: float = 0.01):
        self.mri = mri_system
        self.threshold = threshold
        self.adaptation_rate = adaptation_rate

        # Adaptive threshold
        self.adaptive_threshold = threshold
        self.resonance_history = deque(maxlen=1000)

        # Anomaly tracking
        self.anomalies_detected = []
        self.false_positive_rate = 0.0

    def detect(self, data: np.ndarray, label: Optional[str] = None,
              update_model: bool = True) -> Dict[str, Any]:
        """Detect anomalies in streaming data."""
        # Measure resonance
        resonance = self.mri.measure_resonance(data, evolve=False)
        self.resonance_history.append(resonance)

        # Adapt threshold based on recent history
        if len(self.resonance_history) > 100:
            mean_resonance = np.mean(self.resonance_history)
            std_resonance = np.std(self.resonance_history)
            self.adaptive_threshold = mean_resonance - 2 * std_resonance

        # Detect anomaly
        is_anomaly = resonance < self.adaptive_threshold

        # Update model with normal data
        if update_model and not is_anomaly:
            self.mri.inject_pattern(data, label=label)

        # Track anomalies
        if is_anomaly:
            self.anomalies_detected.append({
                'timestamp': time.time(),
                'resonance': resonance,
                'label': label,
                'data_hash': hash(data.tobytes())
            })

        return {
            'is_anomaly': is_anomaly,
            'resonance': resonance,
            'threshold': self.adaptive_threshold,
            'anomaly_score': 1.0 - resonance,
            'total_anomalies': len(self.anomalies_detected)
        }

    def get_anomaly_report(self) -> Dict[str, Any]:
        """Generate anomaly detection report."""
        if not self.anomalies_detected:
            return {'total_anomalies': 0}

        # Calculate statistics
        recent_anomalies = [a for a in self.anomalies_detected
                           if time.time() - a['timestamp'] < 3600]  # Last hour

        return {
            'total_anomalies': len(self.anomalies_detected),
            'recent_anomalies_1h': len(recent_anomalies),
            'adaptive_threshold': self.adaptive_threshold,
            'mean_anomaly_score': np.mean([1.0 - a['resonance'] for a in self.anomalies_detected]),
            'anomaly_timestamps': [a['timestamp'] for a in self.anomalies_detected[-10:]]
        }


# ============================================================================
# ONLINE LEARNING MANAGER
# ============================================================================

class OnlineLearningManager:
    """Manage continuous online learning with concept drift detection."""

    def __init__(self, mri_system, drift_threshold: float = 0.1,
                 window_size: int = 100):
        self.mri = mri_system
        self.drift_threshold = drift_threshold
        self.window_size = window_size

        # Concept drift tracking
        self.performance_window = deque(maxlen=window_size)
        self.drift_detected_count = 0
        self.last_drift_timestamp = None

        # Model versions
        self.model_versions = []
        self.current_version = 0

    def learn_online(self, data: np.ndarray, label: Optional[str] = None,
                    true_label: Optional[str] = None) -> Dict[str, Any]:
        """Learn from online data with drift detection."""
        # Make prediction first
        prediction_resonance = self.mri.measure_resonance(data, evolve=False)

        # Learn from data
        self.mri.inject_pattern(data, label=label or true_label)

        # Track performance
        if true_label and label:
            correct = (label == true_label)
            self.performance_window.append(correct)

        # Detect concept drift
        drift_detected = False
        if len(self.performance_window) == self.window_size:
            recent_performance = np.mean(list(self.performance_window)[-50:])
            older_performance = np.mean(list(self.performance_window)[:50])

            if abs(recent_performance - older_performance) > self.drift_threshold:
                drift_detected = True
                self.drift_detected_count += 1
                self.last_drift_timestamp = time.time()

                # Save current model version
                self._save_model_version()

        return {
            'prediction_resonance': prediction_resonance,
            'drift_detected': drift_detected,
            'current_performance': np.mean(self.performance_window) if self.performance_window else None,
            'model_version': self.current_version,
            'total_drifts_detected': self.drift_detected_count
        }

    def _save_model_version(self):
        """Save current model as a version."""
        version_info = {
            'version': self.current_version,
            'timestamp': time.time(),
            'field_state': self.mri.field.field.copy(),
            'performance': np.mean(self.performance_window) if self.performance_window else 0
        }

        self.model_versions.append(version_info)
        self.current_version += 1

    def rollback_to_version(self, version: int):
        """Rollback to a previous model version."""
        if 0 <= version < len(self.model_versions):
            version_info = self.model_versions[version]
            self.mri.field.field = version_info['field_state'].copy()
            self.current_version = version
            print(f"Rolled back to version {version}")
        else:
            raise ValueError(f"Version {version} not found")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_streaming_processing():
    """Example: Real-time streaming data processing."""
    from mri_production_complete import MorphicResonanceIntelligence, MRIConfig

    # Initialize system
    config = MRIConfig(field_size=(64, 64))
    mri = MorphicResonanceIntelligence(config)

    # Create streaming processor
    processor = StreamingDataProcessor(mri, buffer_size=1000, batch_size=10)

    # Simulate data stream
    print("Processing streaming data...")
    for i in range(100):
        # Generate synthetic streaming data
        data = np.random.rand(64, 64) + i * 0.01
        label = f"stream_sample_{i}"

        # Process
        result = processor.process_stream(data, label, online_validation=True)

        if i % 20 == 0:
            stats = processor.get_statistics()
            print(f"Sample {i}: Throughput={stats['throughput_samples_per_sec']:.2f} samples/sec")

    print("Streaming processing complete!")
    final_stats = processor.get_statistics()
    print(f"Total samples processed: {final_stats['samples_processed']}")
    print(f"Average processing time: {final_stats['avg_processing_time']*1000:.2f}ms")


def example_time_series_processing():
    """Example: Time series processing and forecasting."""
    from mri_production_complete import MorphicResonanceIntelligence, MRIConfig

    config = MRIConfig(field_size=(64, 64))
    mri = MorphicResonanceIntelligence(config)

    processor = TimeSeriesProcessor(mri, window_size=50, stride=10)

    # Generate synthetic time series
    t = np.linspace(0, 10, 500)
    time_series = np.sin(2 * np.pi * t) + np.random.randn(len(t)) * 0.1

    # Process time series
    print("Processing time series...")
    result = processor.process_time_series(time_series, online=True)
    print(f"Resonance: {result.get('resonance', 'N/A')}")

    # Forecast
    print("Forecasting future values...")
    forecast = processor.forecast(n_steps=20)
    print(f"Forecast (first 5 values): {forecast[:5]}")


def example_anomaly_detection():
    """Example: Streaming anomaly detection."""
    from mri_production_complete import MorphicResonanceIntelligence, MRIConfig

    config = MRIConfig(field_size=(64, 64))
    mri = MorphicResonanceIntelligence(config)

    detector = StreamingAnomalyDetector(mri, threshold=0.3)

    print("Running anomaly detection...")

    # Train on normal data
    for i in range(50):
        normal_data = np.random.rand(64, 64) * 0.5
        detector.detect(normal_data, label='normal', update_model=True)

    # Introduce anomalies
    for i in range(10):
        anomaly_data = np.random.rand(64, 64) * 2.0  # Different distribution
        result = detector.detect(anomaly_data, label='test', update_model=False)

        if result['is_anomaly']:
            print(f"Anomaly detected! Score: {result['anomaly_score']:.3f}")

    # Get report
    report = detector.get_anomaly_report()
    print(f"\nTotal anomalies detected: {report['total_anomalies']}")


if __name__ == "__main__":
    print("="*70)
    print("MRI STREAMING DATA PROCESSING EXAMPLES")
    print("="*70)

    print("\n1. Streaming Processing:")
    example_streaming_processing()

    print("\n2. Time Series Processing:")
    example_time_series_processing()

    print("\n3. Anomaly Detection:")
    example_anomaly_detection()
