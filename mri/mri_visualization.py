"""
MRI Advanced Visualization and Explainability
==============================================

Production-grade visualization and explainability tools for MRI systems.
Includes interactive dashboards, 3D visualizations, and interpretability analysis.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json


# ============================================================================
# FIELD VISUALIZATION
# ============================================================================

class FieldVisualizer:
    """Advanced visualization for resonance fields."""

    def __init__(self, mri_system):
        self.mri = mri_system
        self.field = mri_system.field

    def visualize_complete(self, save_path: Optional[str] = None, interactive: bool = False):
        """Create comprehensive field visualization."""
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Amplitude visualization
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_amplitude(ax1)

        # Phase visualization
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_phase(ax2)

        # Power spectrum
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_power_spectrum(ax3)

        # Energy evolution
        ax4 = fig.add_subplot(gs[0, 3])
        self._plot_energy_evolution(ax4)

        # Topological defects
        ax5 = fig.add_subplot(gs[1, 0])
        self._plot_topological_defects(ax5)

        # Phase coherence
        ax6 = fig.add_subplot(gs[1, 1])
        self._plot_phase_coherence(ax6)

        # Frequency modes
        ax7 = fig.add_subplot(gs[1, 2])
        self._plot_dominant_modes(ax7)

        # Learning curve
        ax8 = fig.add_subplot(gs[1, 3])
        self._plot_learning_curve(ax8)

        # 3D amplitude surface
        ax9 = fig.add_subplot(gs[2, :2], projection='3d')
        self._plot_3d_surface(ax9)

        # Pattern embeddings
        ax10 = fig.add_subplot(gs[2, 2:])
        self._plot_pattern_embeddings(ax10)

        plt.suptitle('MRI System Complete Visualization', fontsize=16, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")

        if not interactive:
            plt.close()
        else:
            plt.show()

        return fig

    def _plot_amplitude(self, ax):
        """Plot field amplitude."""
        amplitude = np.abs(self.field.field)
        im = ax.imshow(amplitude, cmap='viridis', interpolation='bilinear')
        ax.set_title('Field Amplitude')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _plot_phase(self, ax):
        """Plot field phase."""
        phase = np.angle(self.field.field)
        im = ax.imshow(phase, cmap='twilight', interpolation='bilinear')
        ax.set_title('Field Phase')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _plot_power_spectrum(self, ax):
        """Plot power spectrum."""
        from scipy.fft import fft2, fftshift

        freq_field = fftshift(fft2(self.field.field))
        power = np.log(np.abs(freq_field)**2 + 1)

        im = ax.imshow(power, cmap='hot', interpolation='nearest')
        ax.set_title('Power Spectrum (log scale)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _plot_energy_evolution(self, ax):
        """Plot energy evolution over time."""
        if self.field.energy_history:
            ax.plot(self.field.energy_history, linewidth=2, color='#2E86AB')
            ax.fill_between(range(len(self.field.energy_history)),
                           self.field.energy_history, alpha=0.3, color='#2E86AB')
            ax.set_title('Field Energy Evolution')
            ax.set_xlabel('Evolution Step')
            ax.set_ylabel('Energy')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No evolution history', ha='center', va='center')
            ax.set_title('Field Energy Evolution')

    def _plot_topological_defects(self, ax):
        """Plot topological defects."""
        amplitude = np.abs(self.field.field)
        phase = np.angle(self.field.field)

        # Calculate phase gradient for defect detection
        grad_y, grad_x = np.gradient(phase)
        curl = np.abs(grad_x[1:, :] - grad_x[:-1, :] +
                     grad_y[:, 1:] - grad_y[:, :-1])

        # Detect vortices
        threshold = np.percentile(curl, 95)
        defects = curl > threshold

        im = ax.imshow(amplitude, cmap='gray', alpha=0.5)
        ax.contour(defects[:-1, :-1], colors='red', linewidths=2, levels=[0.5])
        ax.set_title('Topological Defects (Vortices)')
        ax.axis('off')

    def _plot_phase_coherence(self, ax):
        """Plot phase coherence map."""
        phase = np.angle(self.field.field)

        # Local phase coherence
        from scipy.ndimage import uniform_filter

        coherence = np.abs(
            uniform_filter(np.cos(phase), size=5) +
            1j * uniform_filter(np.sin(phase), size=5)
        )

        im = ax.imshow(coherence, cmap='RdYlGn', interpolation='bilinear')
        ax.set_title('Phase Coherence')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _plot_dominant_modes(self, ax):
        """Plot dominant frequency modes."""
        modes = self.mri.extract_modes(n_modes=10, mode_type='frequency')

        if modes:
            # Show first 3 modes
            combined = np.zeros_like(modes[0])
            for i, mode in enumerate(modes[:3]):
                combined += (i+1) * np.abs(mode)

            im = ax.imshow(combined, cmap='plasma', interpolation='bilinear')
            ax.set_title('Dominant Frequency Modes (Top 3)')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.text(0.5, 0.5, 'No modes extracted', ha='center', va='center')
            ax.set_title('Dominant Frequency Modes')

    def _plot_learning_curve(self, ax):
        """Plot learning curve."""
        times = [lc['time'] for lc in self.mri.learning_curve]
        energies = [lc['energy'] for lc in self.mri.learning_curve]

        if times:
            ax2 = ax.twinx()

            line1 = ax.plot(times, linewidth=2, color='#A23B72', label='Learning Time')
            ax.set_ylabel('Time (s)', color='#A23B72')
            ax.tick_params(axis='y', labelcolor='#A23B72')

            line2 = ax2.plot(energies, linewidth=2, color='#F18F01', label='Energy')
            ax2.set_ylabel('Field Energy', color='#F18F01')
            ax2.tick_params(axis='y', labelcolor='#F18F01')

            ax.set_xlabel('Pattern Number')
            ax.set_title('Learning Curve')
            ax.grid(True, alpha=0.3)

            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper right')
        else:
            ax.text(0.5, 0.5, 'No learning data', ha='center', va='center')
            ax.set_title('Learning Curve')

    def _plot_3d_surface(self, ax):
        """Plot 3D amplitude surface."""
        from mpl_toolkits.mplot3d import Axes3D

        amplitude = np.abs(self.field.field)

        # Downsample for performance
        stride = max(1, amplitude.shape[0] // 50)
        x = np.arange(0, amplitude.shape[1], stride)
        y = np.arange(0, amplitude.shape[0], stride)
        X, Y = np.meshgrid(x, y)
        Z = amplitude[::stride, ::stride]

        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8,
                              linewidth=0, antialiased=True)

        ax.set_title('3D Amplitude Surface')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Amplitude')
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    def _plot_pattern_embeddings(self, ax):
        """Plot pattern embeddings in 2D."""
        if len(self.mri.pattern_memory) < 2:
            ax.text(0.5, 0.5, 'Insufficient patterns for embedding',
                   ha='center', va='center')
            ax.set_title('Pattern Embeddings (t-SNE)')
            return

        # Extract pattern representations
        patterns = []
        labels = []

        for pattern_info in self.mri.pattern_memory:
            if pattern_info.get('encoded_pattern') is not None:
                patterns.append(pattern_info['encoded_pattern'].flatten())
                labels.append(pattern_info.get('label', 'unknown'))

        if len(patterns) < 2:
            ax.text(0.5, 0.5, 'Insufficient encoded patterns',
                   ha='center', va='center')
            ax.set_title('Pattern Embeddings (t-SNE)')
            return

        patterns = np.array([np.real(p) for p in patterns])

        # Reduce to 2D using PCA (simplified t-SNE)
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            embeddings = pca.fit_transform(patterns)

            # Plot
            scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1],
                               c=range(len(embeddings)), cmap='tab20',
                               s=100, alpha=0.6, edgecolors='black')

            # Add labels
            for i, label in enumerate(labels):
                ax.annotate(str(label)[:10], (embeddings[i, 0], embeddings[i, 1]),
                          fontsize=8, alpha=0.7)

            ax.set_title('Pattern Embeddings (PCA)')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.grid(True, alpha=0.3)

        except ImportError:
            ax.text(0.5, 0.5, 'sklearn required for embeddings',
                   ha='center', va='center')
            ax.set_title('Pattern Embeddings')

    def create_animation(self, evolution_steps: int = 100, save_path: str = "field_evolution.gif"):
        """Create animated visualization of field evolution."""
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Store initial state
        initial_field = self.field.field.copy()

        frames = []
        for step in range(evolution_steps):
            # Evolve field
            self.mri.evolve_system(steps=1)

            # Capture frame
            frame_data = {
                'amplitude': np.abs(self.field.field),
                'phase': np.angle(self.field.field),
                'energy': self.field.get_energy()
            }
            frames.append(frame_data)

        # Restore initial state
        self.field.field = initial_field

        # Create animation
        def update(frame_idx):
            frame = frames[frame_idx]

            # Update plots
            axes[0].clear()
            axes[0].imshow(frame['amplitude'], cmap='viridis')
            axes[0].set_title(f'Amplitude (Step {frame_idx})')
            axes[0].axis('off')

            axes[1].clear()
            axes[1].imshow(frame['phase'], cmap='twilight')
            axes[1].set_title(f'Phase (Step {frame_idx})')
            axes[1].axis('off')

            axes[2].clear()
            energies = [f['energy'] for f in frames[:frame_idx+1]]
            axes[2].plot(energies, color='#2E86AB', linewidth=2)
            axes[2].set_title('Energy Evolution')
            axes[2].set_xlabel('Step')
            axes[2].set_ylabel('Energy')
            axes[2].grid(True, alpha=0.3)

        anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                      interval=50, repeat=True)

        anim.save(save_path, writer='pillow', fps=20)
        print(f"Animation saved to {save_path}")
        plt.close()

        return anim


# ============================================================================
# EXPLAINABILITY ANALYZER
# ============================================================================

class ExplainabilityAnalyzer:
    """Analyze and explain MRI system decisions."""

    def __init__(self, mri_system):
        self.mri = mri_system

    def explain_resonance(self, pattern: np.ndarray, label: Optional[str] = None) -> Dict[str, Any]:
        """Explain why a pattern has specific resonance."""
        # Measure overall resonance
        resonance_details = self.mri.measure_resonance(pattern, evolve=False, return_details=True)

        # Analyze frequency contributions
        freq_analysis = self._analyze_frequency_contributions(pattern)

        # Spatial importance map
        importance_map = self._compute_importance_map(pattern)

        # Similar patterns
        similar_patterns = self._find_similar_patterns(pattern, top_k=5)

        explanation = {
            'resonance_score': resonance_details['resonance'],
            'phase_difference': resonance_details['phase_difference'],
            'field_energy': resonance_details['field_energy'],
            'frequency_analysis': freq_analysis,
            'importance_map': importance_map,
            'similar_patterns': similar_patterns,
            'interpretation': self._generate_interpretation(resonance_details, freq_analysis)
        }

        return explanation

    def _analyze_frequency_contributions(self, pattern: np.ndarray) -> Dict[str, Any]:
        """Analyze which frequencies contribute to resonance."""
        from scipy.fft import fft2, fftshift

        # Encode pattern
        encoded = self.mri.encoder.encode(pattern)

        # Get frequency representation
        freq_pattern = fftshift(fft2(encoded))
        freq_field = fftshift(fft2(self.mri.field.field))

        # Calculate contribution of each frequency
        contributions = np.abs(np.conj(freq_pattern) * freq_field)

        # Find dominant frequencies
        flat_contrib = contributions.flatten()
        top_indices = np.argsort(flat_contrib)[-10:]

        return {
            'total_contribution': float(np.sum(contributions)),
            'max_contribution': float(np.max(contributions)),
            'mean_contribution': float(np.mean(contributions)),
            'dominant_frequencies': len(top_indices),
            'contribution_map': contributions
        }

    def _compute_importance_map(self, pattern: np.ndarray) -> np.ndarray:
        """Compute spatial importance map using perturbation analysis."""
        import matplotlib.pyplot as plt

        baseline_resonance = self.mri.measure_resonance(pattern, evolve=False)

        importance = np.zeros_like(pattern)

        # Perturbation analysis (simplified - sample only subset for speed)
        patch_size = max(8, pattern.shape[0] // 10)
        stride = patch_size // 2

        for i in range(0, pattern.shape[0] - patch_size, stride):
            for j in range(0, pattern.shape[1] - patch_size, stride):
                # Perturb region
                perturbed = pattern.copy()
                perturbed[i:i+patch_size, j:j+patch_size] = 0

                # Measure change in resonance
                perturbed_resonance = self.mri.measure_resonance(perturbed, evolve=False)
                importance[i:i+patch_size, j:j+patch_size] += abs(baseline_resonance - perturbed_resonance)

        # Normalize
        importance = importance / (importance.max() + 1e-10)

        return importance

    def _find_similar_patterns(self, pattern: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find patterns similar to given pattern."""
        similarities = []

        encoded_pattern = self.mri.encoder.encode(pattern)

        for pattern_info in self.mri.pattern_memory:
            if pattern_info.get('encoded_pattern') is not None:
                # Calculate similarity
                from mri_production_complete import FieldOperations
                overlap = FieldOperations.overlap_integral(
                    encoded_pattern,
                    pattern_info['encoded_pattern']
                )

                similarity = np.abs(overlap) / (
                    np.sqrt(np.sum(np.abs(encoded_pattern)**2)) *
                    np.sqrt(np.sum(np.abs(pattern_info['encoded_pattern'])**2)) + 1e-10
                )

                similarities.append({
                    'label': pattern_info.get('label', 'unknown'),
                    'similarity': float(similarity),
                    'timestamp': pattern_info.get('timestamp', 0)
                })

        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)

        return similarities[:top_k]

    def _generate_interpretation(self, resonance_details: Dict, freq_analysis: Dict) -> str:
        """Generate human-readable interpretation."""
        resonance = resonance_details['resonance']
        freq_contrib = freq_analysis['mean_contribution']

        interpretation = []

        if resonance > 0.7:
            interpretation.append("Strong resonance - pattern is well-recognized by the system.")
        elif resonance > 0.4:
            interpretation.append("Moderate resonance - pattern has some similarity to learned patterns.")
        else:
            interpretation.append("Weak resonance - pattern is novel or dissimilar to learned patterns.")

        if freq_contrib > 1000:
            interpretation.append("High frequency contribution indicates strong feature matching.")
        else:
            interpretation.append("Low frequency contribution suggests weak feature alignment.")

        return " ".join(interpretation)

    def generate_report(self, pattern: np.ndarray, label: Optional[str] = None,
                       save_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive explainability report."""
        import matplotlib.pyplot as plt

        explanation = self.explain_resonance(pattern, label)

        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Original pattern
        axes[0, 0].imshow(pattern, cmap='gray')
        axes[0, 0].set_title('Input Pattern')
        axes[0, 0].axis('off')

        # Importance map
        im = axes[0, 1].imshow(explanation['importance_map'], cmap='hot')
        axes[0, 1].set_title('Importance Map')
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

        # Frequency contributions
        contrib_map = explanation['frequency_analysis']['contribution_map']
        im = axes[0, 2].imshow(np.log(contrib_map + 1), cmap='viridis')
        axes[0, 2].set_title('Frequency Contributions')
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

        # Similar patterns
        axes[1, 0].axis('off')
        if explanation['similar_patterns']:
            text = "Top Similar Patterns:\n\n"
            for i, sim in enumerate(explanation['similar_patterns'][:5]):
                text += f"{i+1}. {sim['label']}: {sim['similarity']:.3f}\n"
            axes[1, 0].text(0.1, 0.5, text, fontsize=10, family='monospace', va='center')
            axes[1, 0].set_title('Similar Patterns')

        # Metrics
        metrics_text = f"""
Resonance Metrics:

Score: {explanation['resonance_score']:.4f}
Phase Diff: {explanation['phase_difference']:.4f}
Field Energy: {explanation['field_energy']:.2f}

Frequency Analysis:
Total Contrib: {explanation['frequency_analysis']['total_contribution']:.2f}
Mean Contrib: {explanation['frequency_analysis']['mean_contribution']:.2f}
        """
        axes[1, 1].axis('off')
        axes[1, 1].text(0.1, 0.5, metrics_text.strip(), fontsize=9,
                       family='monospace', va='center')
        axes[1, 1].set_title('Detailed Metrics')

        # Interpretation
        axes[1, 2].axis('off')
        axes[1, 2].text(0.1, 0.5, explanation['interpretation'],
                       fontsize=10, wrap=True, va='center')
        axes[1, 2].set_title('Interpretation')

        plt.suptitle(f'Explainability Analysis{" - " + label if label else ""}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Explainability report saved to {save_path}")

        plt.close()

        return explanation


# ============================================================================
# INTERACTIVE DASHBOARD
# ============================================================================

class InteractiveDashboard:
    """Interactive web-based dashboard for MRI system."""

    def __init__(self, mri_system, port: int = 8050):
        self.mri = mri_system
        self.port = port
        self.app = None

    def create_dashboard(self):
        """Create interactive dashboard using Plotly Dash."""
        try:
            import dash
            from dash import dcc, html
            from dash.dependencies import Input, Output
            import plotly.graph_objs as go

            self.app = dash.Dash(__name__)

            # Layout
            self.app.layout = html.Div([
                html.H1('MRI System Dashboard', style={'textAlign': 'center'}),

                html.Div([
                    html.Div([
                        html.H3('System Metrics'),
                        dcc.Graph(id='metrics-graph'),
                        dcc.Interval(id='interval-component', interval=2000, n_intervals=0)
                    ], className='six columns'),

                    html.Div([
                        html.H3('Field Visualization'),
                        dcc.Graph(id='field-viz'),
                    ], className='six columns'),
                ], className='row'),

                html.Div([
                    html.H3('Pattern Analysis'),
                    dcc.Graph(id='pattern-analysis'),
                ])
            ])

            # Callbacks
            @self.app.callback(
                Output('metrics-graph', 'figure'),
                Input('interval-component', 'n_intervals')
            )
            def update_metrics(n):
                metrics = self.mri.get_system_metrics()

                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=metrics['patterns_learned'],
                    title={'text': "Patterns Learned"},
                    gauge={'axis': {'range': [None, 1000]}}
                ))

                return fig

            @self.app.callback(
                Output('field-viz', 'figure'),
                Input('interval-component', 'n_intervals')
            )
            def update_field(n):
                amplitude = np.abs(self.mri.field.field)

                fig = go.Figure(data=go.Heatmap(
                    z=amplitude,
                    colorscale='Viridis'
                ))

                fig.update_layout(title='Field Amplitude')
                return fig

        except ImportError:
            print("Dash not installed. Install with: pip install dash plotly")
            return None

    def run(self, debug: bool = False):
        """Run the dashboard server."""
        if self.app is None:
            self.create_dashboard()

        if self.app:
            print(f"Starting dashboard on http://localhost:{self.port}")
            self.app.run_server(debug=debug, port=self.port)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    from mri_production_complete import MorphicResonanceIntelligence, MRIConfig, create_test_patterns

    # Initialize system
    config = MRIConfig(field_size=(128, 128))
    mri = MorphicResonanceIntelligence(config)

    # Learn some patterns
    patterns = create_test_patterns()
    for label, pattern in patterns.items():
        mri.inject_pattern(pattern, label=label)
        mri.evolve_system(steps=10)

    # Create visualizations
    visualizer = FieldVisualizer(mri)
    visualizer.visualize_complete(save_path="mri_complete_visualization.png")

    # Explainability analysis
    analyzer = ExplainabilityAnalyzer(mri)
    explanation = analyzer.generate_report(
        patterns['circle'],
        label='circle',
        save_path="explainability_report.png"
    )

    print("Visualization and explainability tools demonstration complete!")
    print(f"Resonance interpretation: {explanation['interpretation']}")
