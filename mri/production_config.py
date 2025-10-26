from mri_production_complete import MRIConfig, FieldDimension, EvolutionMode, EncodingScheme

production_config = MRIConfig(
    # Core parameters
    field_size=(512, 512),
    field_dimension=FieldDimension.TWO_D,
    
    # Learning
    learning_rate=0.1,
    adaptive_learning=True,
    learning_momentum=0.95,
    
    # Evolution
    evolution_mode=EvolutionMode.HYBRID,
    evolution_steps=20,
    diffusion_coefficient=0.05,
    nonlinearity_strength=0.02,
    
    # Memory
    enable_holographic_memory=True,
    memory_compression=True,
    max_patterns=100000,
    
    # Resonance parameters
    resonance_threshold=0.2,
    resonance_bandwidth=0.05,
    coupling_strength=0.8,

    # Performance
    use_sparse_representation=True,
    use_gpu=False,  # Set to True if you have a CUDA-enabled GPU
    num_threads=8,
    
    # Advanced
    enable_quantum_effects=True,
    enable_topological_modes=True,
    enable_meta_learning=True
)
