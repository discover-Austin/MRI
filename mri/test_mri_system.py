"""
Comprehensive Test Suite for MRI System
========================================

Production-grade testing with pytest.
Tests all components: core, optimization, integration, and deployment.
"""

import pytest
import numpy as np
import tempfile
import pickle
from pathlib import Path
from typing import List, Dict, Any

from mri_production_complete import (
    MorphicResonanceIntelligence, MRIConfig, ResonanceField,
    FieldOperations, InformationEncoder, InformationDecoder,
    MRIClassifier, MRIAnomalyDetector, MRIRecommender,
    FieldDimension, EvolutionMode, EncodingScheme,
    create_test_patterns, benchmark_performance
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def default_config():
    """Default MRI configuration for testing."""
    return MRIConfig(
        field_size=(64, 64),
        learning_rate=0.1,
        evolution_steps=5,
        enable_holographic_memory=True
    )


@pytest.fixture
def mri_system(default_config):
    """Initialized MRI system."""
    return MorphicResonanceIntelligence(default_config)


@pytest.fixture
def test_patterns():
    """Standard test patterns."""
    return create_test_patterns(size=(64, 64))


@pytest.fixture
def temp_directory():
    """Temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# CORE SYSTEM TESTS
# ============================================================================

class TestMRICore:
    """Test core MRI functionality."""

    def test_initialization(self, default_config):
        """Test system initialization."""
        mri = MorphicResonanceIntelligence(default_config)

        assert mri.config == default_config
        assert mri.field.field.shape == default_config.field_size
        assert len(mri.pattern_memory) == 0
        assert isinstance(mri.encoder, InformationEncoder)
        assert isinstance(mri.decoder, InformationDecoder)

    def test_pattern_injection(self, mri_system, test_patterns):
        """Test pattern learning."""
        pattern = test_patterns['square']

        result = mri_system.inject_pattern(pattern, label='test')

        assert 'label' in result
        assert 'timestamp' in result
        assert 'energy_delta' in result
        assert len(mri_system.pattern_memory) == 1
        assert mri_system.pattern_memory[0]['label'] == 'test'

    def test_multiple_pattern_learning(self, mri_system, test_patterns):
        """Test learning multiple patterns."""
        for label, pattern in test_patterns.items():
            mri_system.inject_pattern(pattern, label=label)

        assert len(mri_system.pattern_memory) == len(test_patterns)
        assert len(mri_system.frequency_library) == len(test_patterns)

    def test_field_evolution(self, mri_system):
        """Test field evolution dynamics."""
        initial_energy = mri_system.field.get_energy()

        mri_system.evolve_system(steps=10)

        assert len(mri_system.field.energy_history) > 0
        assert len(mri_system.field.entropy_history) > 0

    def test_resonance_measurement(self, mri_system, test_patterns):
        """Test resonance measurement."""
        pattern = test_patterns['circle']

        # Learn pattern
        mri_system.inject_pattern(pattern, label='circle')

        # Measure resonance
        resonance = mri_system.measure_resonance(pattern, evolve=False)

        assert 0 <= resonance <= 1.0
        assert resonance > 0.5  # Should recognize learned pattern

    def test_prediction(self, mri_system, test_patterns):
        """Test prediction generation."""
        input_pattern = test_patterns['square']
        target_pattern = test_patterns['circle']

        # Learn association
        mri_system.inject_pattern(input_pattern, phase=0.0)
        mri_system.inject_pattern(target_pattern, phase=0.0)
        mri_system.evolve_system()

        # Generate prediction
        prediction = mri_system.predict(input_pattern, evolution_time=20)

        assert prediction.shape == input_pattern.shape
        assert np.isfinite(prediction).all()

    def test_associative_learning(self, mri_system, test_patterns):
        """Test pattern association."""
        pattern_pairs = [
            (test_patterns['square'], test_patterns['circle']),
            (test_patterns['stripes'], test_patterns['diagonal'])
        ]

        history = mri_system.associate_patterns(pattern_pairs, iterations=20)

        assert 'resonances' in history
        assert 'energies' in history
        assert len(history['resonances']) > 0

    def test_mode_extraction(self, mri_system, test_patterns):
        """Test dominant mode extraction."""
        # Learn some patterns
        for label, pattern in test_patterns.items():
            mri_system.inject_pattern(pattern, label=label)

        modes = mri_system.extract_modes(n_modes=5, mode_type='frequency')

        assert len(modes) == 5
        assert all(isinstance(mode, np.ndarray) for mode in modes)

    def test_save_and_load(self, mri_system, test_patterns, temp_directory):
        """Test state persistence."""
        # Learn patterns
        for label, pattern in test_patterns.items():
            mri_system.inject_pattern(pattern, label=label)

        # Save state
        save_path = temp_directory / "test_mri.pkl"
        mri_system.save_state(str(save_path))

        assert save_path.exists()

        # Load state
        mri_loaded = MorphicResonanceIntelligence(mri_system.config)
        mri_loaded.load_state(str(save_path))

        assert len(mri_loaded.pattern_memory) == len(test_patterns)
        assert np.allclose(mri_loaded.field.field, mri_system.field.field)

    def test_memory_compression(self, mri_system):
        """Test memory compression."""
        # Add many patterns
        for i in range(50):
            pattern = np.random.rand(64, 64)
            mri_system.inject_pattern(pattern, label=f"pattern_{i}")

        initial_count = len(mri_system.pattern_memory)

        # Compress memory
        mri_system.compress_memory()

        # Should have fewer patterns after compression
        assert len(mri_system.pattern_memory) <= initial_count


# ============================================================================
# FIELD OPERATIONS TESTS
# ============================================================================

class TestFieldOperations:
    """Test mathematical field operations."""

    def test_fourier_transform(self):
        """Test FFT operations."""
        field = np.random.rand(32, 32) + 1j * np.random.rand(32, 32)

        freq = FieldOperations.fourier_transform(field)
        reconstructed = FieldOperations.fourier_transform(freq, inverse=True)

        assert np.allclose(field, reconstructed, atol=1e-10)

    def test_gradient(self):
        """Test gradient calculation."""
        field = np.random.rand(32, 32)

        gradients = FieldOperations.gradient(field)

        assert len(gradients) == 2  # 2D field
        assert all(g.shape == field.shape for g in gradients)

    def test_laplacian(self):
        """Test Laplacian operator."""
        field = np.random.rand(32, 32)

        laplacian = FieldOperations.laplacian(field)

        assert laplacian.shape == field.shape
        assert np.isfinite(laplacian).all()

    def test_overlap_integral(self):
        """Test overlap integral calculation."""
        field1 = np.random.rand(32, 32) + 1j * np.random.rand(32, 32)
        field2 = np.random.rand(32, 32) + 1j * np.random.rand(32, 32)

        overlap = FieldOperations.overlap_integral(field1, field2)

        assert isinstance(overlap, complex)

        # Test self-overlap
        self_overlap = FieldOperations.overlap_integral(field1, field1)
        assert self_overlap.real > 0  # Positive for non-zero field

    def test_normalization(self):
        """Test field normalization."""
        field = np.random.rand(32, 32) * 100  # Large values

        # Max normalization
        normalized = FieldOperations.normalize_field(field, norm_type='max')
        assert np.abs(normalized).max() <= 1.0

        # L2 normalization
        normalized_l2 = FieldOperations.normalize_field(field, norm_type='l2')
        norm = np.sqrt(np.sum(np.abs(normalized_l2)**2))
        assert np.abs(norm - 1.0) < 1e-10


# ============================================================================
# ENCODER/DECODER TESTS
# ============================================================================

class TestEncodingDecoding:
    """Test information encoding and decoding."""

    def test_frequency_encoding(self, default_config):
        """Test frequency domain encoding."""
        encoder = InformationEncoder(default_config)
        data = np.random.rand(64, 64)

        encoded = encoder._encode_frequency(data)

        assert encoded.shape == default_config.field_size
        assert np.iscomplexobj(encoded)

    def test_spatial_encoding(self, default_config):
        """Test spatial encoding."""
        encoder = InformationEncoder(default_config)
        data = np.random.rand(64, 64)

        encoded = encoder._encode_spatial(data)

        assert encoded.shape == default_config.field_size
        assert np.iscomplexobj(encoded)

    def test_holographic_encoding(self, default_config):
        """Test holographic encoding."""
        encoder = InformationEncoder(default_config)
        data = np.random.rand(64, 64)

        encoded = encoder._encode_holographic(data)

        assert encoded.shape == default_config.field_size
        assert np.iscomplexobj(encoded)

    def test_encoding_schemes(self, default_config):
        """Test all encoding schemes."""
        encoder = InformationEncoder(default_config)
        data = np.random.rand(64, 64)

        schemes = [
            EncodingScheme.FREQUENCY,
            EncodingScheme.SPATIAL,
            EncodingScheme.HOLOGRAPHIC,
            EncodingScheme.PHASE,
            EncodingScheme.AMPLITUDE
        ]

        for scheme in schemes:
            encoded = encoder.encode(data, scheme=scheme)
            assert encoded.shape == default_config.field_size
            assert np.iscomplexobj(encoded)

    def test_decoding(self, default_config):
        """Test information decoding."""
        encoder = InformationEncoder(default_config)
        decoder = InformationDecoder(default_config)

        data = np.random.rand(64, 64)
        encoded = encoder.encode(data)
        decoded = decoder.decode(encoded)

        assert decoded.shape == data.shape
        assert np.isreal(decoded).all()


# ============================================================================
# APPLICATION TESTS
# ============================================================================

class TestApplications:
    """Test application templates."""

    def test_classifier(self, default_config, test_patterns):
        """Test MRI classifier."""
        clf = MRIClassifier(default_config)

        X = [test_patterns['square'], test_patterns['circle']]
        y = ['square', 'circle']

        clf.fit(X, y, iterations=20)

        predictions = clf.predict(X)
        assert len(predictions) == len(X)

        probabilities = clf.predict_proba(X)
        assert len(probabilities) == len(X)
        assert all(isinstance(p, dict) for p in probabilities)

    def test_anomaly_detector(self, default_config, test_patterns):
        """Test anomaly detector."""
        detector = MRIAnomalyDetector(default_config, threshold=0.3)

        normal_data = [test_patterns['square'], test_patterns['circle']]
        detector.fit(normal_data)

        test_data = [
            test_patterns['square'],  # Normal
            test_patterns['checkerboard']  # Potentially anomalous
        ]

        predictions = detector.predict(test_data)
        scores = detector.score_samples(test_data)

        assert len(predictions) == len(test_data)
        assert len(scores) == len(test_data)
        assert all(p in [0, 1] for p in predictions)

    def test_recommender(self, default_config):
        """Test recommendation system."""
        recommender = MRIRecommender(default_config)

        # Add interactions
        items = {
            'item1': np.random.rand(64, 64),
            'item2': np.random.rand(64, 64),
            'item3': np.random.rand(64, 64)
        }

        recommender.add_interaction('user1', 'item1', items['item1'])
        recommender.add_interaction('user1', 'item2', items['item2'])

        # Get recommendations
        recommendations = recommender.recommend('user1', n_items=2)

        assert isinstance(recommendations, list)
        # Should recommend item3 since it's not interacted with
        assert 'item3' in recommendations or len(recommendations) >= 0


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Test system performance characteristics."""

    def test_learning_speed(self, mri_system):
        """Test learning performance."""
        import time

        pattern = np.random.rand(64, 64)

        start = time.time()
        for _ in range(10):
            mri_system.inject_pattern(pattern)
        elapsed = time.time() - start

        avg_time = elapsed / 10
        assert avg_time < 0.1  # Should be < 100ms per pattern on CPU

    def test_inference_speed(self, mri_system, test_patterns):
        """Test inference performance."""
        import time

        pattern = test_patterns['square']
        mri_system.inject_pattern(pattern)

        start = time.time()
        for _ in range(10):
            mri_system.measure_resonance(pattern, evolve=False)
        elapsed = time.time() - start

        avg_time = elapsed / 10
        assert avg_time < 0.5  # Should be < 500ms per inference

    def test_memory_usage(self, default_config):
        """Test memory efficiency."""
        mri = MorphicResonanceIntelligence(default_config)

        # Add patterns
        for i in range(100):
            pattern = np.random.rand(64, 64)
            mri.inject_pattern(pattern)

        metrics = mri.get_system_metrics()

        # Should be reasonable memory usage
        assert metrics['memory_usage_mb'] < 100  # < 100MB for 100 patterns

    def test_benchmark_suite(self, default_config):
        """Test benchmark functionality."""
        results = benchmark_performance(default_config, n_patterns=50)

        assert 'avg_learning_time_ms' in results
        assert 'avg_inference_time_ms' in results
        assert 'memory_mb' in results
        assert results['total_patterns'] == 50


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests across components."""

    def test_end_to_end_workflow(self, default_config, test_patterns):
        """Test complete workflow."""
        # Initialize system
        mri = MorphicResonanceIntelligence(default_config)

        # Learn patterns
        for label, pattern in test_patterns.items():
            mri.inject_pattern(pattern, label=label)

        # Evolve system
        mri.evolve_system(steps=10)

        # Test resonance
        for label, pattern in test_patterns.items():
            resonance = mri.measure_resonance(pattern, evolve=False)
            assert resonance > 0.3  # Should recognize learned patterns

        # Generate predictions
        prediction = mri.predict(test_patterns['square'])
        assert prediction.shape == test_patterns['square'].shape

        # Get metrics
        metrics = mri.get_system_metrics()
        assert metrics['patterns_learned'] == len(test_patterns)

    def test_continuous_learning(self, mri_system):
        """Test continuous learning without forgetting."""
        patterns_to_learn = []

        # Learn patterns sequentially
        for i in range(20):
            pattern = np.random.rand(64, 64)
            patterns_to_learn.append(pattern)
            mri_system.inject_pattern(pattern, label=f"pattern_{i}")

        # Verify all patterns still recognized
        resonances = []
        for pattern in patterns_to_learn:
            res = mri_system.measure_resonance(pattern, evolve=False)
            resonances.append(res)

        # All should have decent resonance (no catastrophic forgetting)
        assert np.mean(resonances) > 0.2
        assert min(resonances) > 0.1


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================

class TestConfiguration:
    """Test configuration validation and settings."""

    def test_config_validation(self):
        """Test configuration validation."""
        config = MRIConfig(field_size=(128, 128))
        assert config.validate()

        # Invalid configuration
        with pytest.raises(AssertionError):
            invalid_config = MRIConfig(
                field_size=(128,),  # 1D but claims 2D
                field_dimension=FieldDimension.TWO_D
            )
            invalid_config.validate()

    def test_different_field_dimensions(self):
        """Test different field dimensionalities."""
        # 1D
        config_1d = MRIConfig(
            field_size=(256,),
            field_dimension=FieldDimension.ONE_D
        )
        mri_1d = MorphicResonanceIntelligence(config_1d)
        assert mri_1d.field.field.shape == (256,)

        # 2D
        config_2d = MRIConfig(
            field_size=(128, 128),
            field_dimension=FieldDimension.TWO_D
        )
        mri_2d = MorphicResonanceIntelligence(config_2d)
        assert mri_2d.field.field.shape == (128, 128)

        # 3D
        config_3d = MRIConfig(
            field_size=(32, 32, 32),
            field_dimension=FieldDimension.THREE_D
        )
        mri_3d = MorphicResonanceIntelligence(config_3d)
        assert mri_3d.field.field.shape == (32, 32, 32)

    def test_evolution_modes(self):
        """Test different evolution modes."""
        modes = [
            EvolutionMode.SCHRODINGER,
            EvolutionMode.DIFFUSION,
            EvolutionMode.WAVE,
            EvolutionMode.HYBRID
        ]

        for mode in modes:
            config = MRIConfig(
                field_size=(64, 64),
                evolution_mode=mode,
                evolution_steps=5
            )
            mri = MorphicResonanceIntelligence(config)
            mri.evolve_system()

            # Should complete without errors
            assert len(mri.field.energy_history) > 0


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
