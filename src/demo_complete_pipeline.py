"""
Complete Demo of NNfit1DRes Methodology Adapted for Tip Timing Analysis
=========================================================================

This demonstration script showcases the full implementation of the deep neural network
methodology from the Raman spectroscopy research paper, adapted for tip timing analysis.

The demo includes:
1. Synthetic data generation mimicking real tip timing measurements
2. Data preparation (4D to 2D transformation)
3. Sequential training (NNFit + Unet1D)
4. Comprehensive evaluation
5. Results visualization and reporting

Usage:
    python demo_complete_pipeline.py

Requirements:
    - TensorFlow >= 2.8
    - NumPy
    - Matplotlib
    - SciPy
    - scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path for imports
sys.path.append('.')
sys.path.append('./src')

# Import our implemented modules
from data.data_preparation import TipTimingDataProcessor, create_synthetic_tip_timing_data
from training.sequential_training_pipeline import SequentialTipTimingPipeline
from evaluation.tip_timing_metrics import TipTimingMetrics


def run_complete_demo():
    """
    Run the complete demonstration of the adapted NNfit1DRes methodology
    for tip timing analysis
    """
    print("🚀 COMPLETE DEMO: NNfit1DRes Adapted for Tip Timing Analysis")
    print("=" * 70)
    print("Methodology: Deep neural network preprocessing pipeline")
    print("Original: Raman spectroscopy baseline correction + denoising")
    print("Adapted: Tip timing zero function estimation + vibration denoising")
    print("=" * 70)
    print()
    
    # Configuration parameters
    config = {
        'n_aubes': 24,        # Number of blades
        'n_capteurs': 4,      # Number of sensors  
        'n_tours': 1000,      # Number of rotations
        'n_regimes': 5,       # Number of operating regimes
        'nnfit_epochs': 50,   # Reduced for demo
        'unet_epochs': 40,    # Reduced for demo
        'batch_size': 32,
        'save_dir': './demo_results'
    }
    
    print(f"📊 Demo Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Step 1: Generate synthetic tip timing data
    print("STEP 1: Generating Synthetic Tip Timing Data")
    print("-" * 50)
    
    deflexions_4d, baselines_4d, vibrations_4d = create_synthetic_tip_timing_data(
        n_aubes=config['n_aubes'],
        n_capteurs=config['n_capteurs'], 
        n_tours=config['n_tours'],
        n_regimes=config['n_regimes']
    )
    
    print(f"✓ Generated 4D data shape: {deflexions_4d.shape}")
    print(f"  (aubes × capteurs × tours × regimes)")
    print()
    
    # Step 2: Data preparation
    print("STEP 2: Data Preparation and Transformation")
    print("-" * 50)
    
    processor = TipTimingDataProcessor()
    prepared_data = processor.prepare_for_sequential_training(
        deflexions_4d=deflexions_4d,
        baselines_4d=baselines_4d,
        vibrations_4d=vibrations_4d,
        test_size=0.2,
        val_size=0.1,
        normalize=True
    )
    
    print(f"✓ Prepared data for neural network training:")
    print(f"  Training samples: {prepared_data['info']['n_train_samples']}")
    print(f"  Validation samples: {prepared_data['info']['n_val_samples']}")
    print(f"  Test samples: {prepared_data['info']['n_test_samples']}")
    print(f"  Sequence length: {prepared_data['info']['n_tours']} rotations")
    print()
    
    # Step 3: Sequential training pipeline
    print("STEP 3: Sequential Neural Network Training")
    print("-" * 50)
    
    # Initialize the pipeline
    pipeline = SequentialTipTimingPipeline(
        n_tours=config['n_tours'],
        save_dir=config['save_dir']
    )
    
    # Run complete training
    pipeline_results = pipeline.complete_pipeline_training(
        prepared_data=prepared_data,
        nnfit_epochs=config['nnfit_epochs'],
        unet_epochs=config['unet_epochs'],
        batch_size=config['batch_size']
    )
    
    print("✓ Sequential training completed successfully!")
    print()
    
    # Step 4: Demonstration of pipeline inference
    print("STEP 4: Pipeline Inference Demonstration")
    print("-" * 50)
    
    # Use test data to demonstrate inference
    test_deflections = prepared_data['nnfit']['X_test'][:5]  # First 5 test samples
    
    print(f"Processing {len(test_deflections)} test samples...")
    
    inference_results = pipeline.predict_pipeline(test_deflections)
    
    print("✓ Pipeline inference completed!")
    print(f"  Input: Raw deflection measurements")
    print(f"  Stage 1 output: Estimated zero functions + corrected deflections")
    print(f"  Stage 2 output: Clean vibration signals")
    print()
    
    # Step 5: Comprehensive evaluation
    print("STEP 5: Comprehensive Evaluation")
    print("-" * 50)
    
    metrics_calculator = TipTimingMetrics()
    
    # Extract evaluation data
    evaluation_data = {
        'raw_deflections': prepared_data['nnfit']['X_test'],
        'true_baselines': prepared_data['nnfit']['y_test'],
        'true_vibrations': prepared_data['unet']['y_test'],
        'metadata': prepared_data['metadata']['test']
    }
    
    # Get pipeline predictions for test set
    test_pipeline_results = pipeline.predict_pipeline(evaluation_data['raw_deflections'])
    
    # End-to-end evaluation
    e2e_evaluation = metrics_calculator.evaluate_end_to_end_pipeline(
        raw_deflections=evaluation_data['raw_deflections'],
        true_vibrations=evaluation_data['true_vibrations'],
        pipeline_results=test_pipeline_results,
        sample_metadata=evaluation_data['metadata']
    )
    
    print("✓ Evaluation completed!")
    print(f"  End-to-end RMSE: {e2e_evaluation['overall']['rmse_vibration']:.6f}")
    print(f"  End-to-end SNR: {e2e_evaluation['overall']['snr_vibration_db']:.2f} dB")
    print(f"  Noise reduction: {e2e_evaluation.get('noise_reduction_db', 'N/A'):.2f} dB")
    print()
    
    # Step 6: Generate comprehensive report
    print("STEP 6: Report Generation")
    print("-" * 50)
    
    # Combine all evaluation results
    complete_evaluation = {
        'nnfit_performance': pipeline_results['evaluation_metrics']['nnfit_performance'],
        'unet_performance': pipeline_results['evaluation_metrics']['unet_performance'], 
        'end_to_end': e2e_evaluation
    }
    
    # Generate report
    report_path = os.path.join(config['save_dir'], 'demo_evaluation_report.txt')
    report_text = metrics_calculator.generate_comprehensive_report(
        complete_evaluation, 
        save_path=report_path
    )
    
    print("✓ Comprehensive evaluation report generated!")
    print()
    
    # Step 7: Create visualization
    print("STEP 7: Results Visualization")
    print("-" * 50)
    
    create_demo_visualizations(
        test_deflections[:3],  # First 3 samples
        inference_results,
        prepared_data,
        config['save_dir']
    )
    
    print("✓ Visualization plots created!")
    print()
    
    # Demo summary
    print("🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("Summary of Achieved Functionality:")
    print("✓ Synthetic tip timing data generation")
    print("✓ 4D to 2D data transformation for neural networks")
    print("✓ NNFit model for zero function estimation (baseline correction)")
    print("✓ Unet1D model for vibration signal denoising")
    print("✓ Sequential training pipeline implementation")
    print("✓ Comprehensive evaluation with tip timing-specific metrics")
    print("✓ End-to-end pipeline inference")
    print("✓ Results visualization and reporting")
    print()
    print("🔬 Methodology Successfully Adapted:")
    print("  Original: Raman spectroscopy NNfit1DRes preprocessing")
    print("  Adapted: Tip timing vibration analysis for turbomachine monitoring")
    print()
    print(f"📁 All results saved to: {config['save_dir']}")
    print("=" * 70)


def create_demo_visualizations(sample_deflections, inference_results, prepared_data, save_dir):
    """
    Create comprehensive visualizations of the pipeline results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a comprehensive figure showing the complete pipeline
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('NNfit1DRes Adapted for Tip Timing Analysis - Complete Pipeline Demo', 
                 fontsize=16, fontweight='bold')
    
    n_samples = len(sample_deflections)
    colors = ['blue', 'red', 'green']
    
    for i in range(min(n_samples, 3)):
        
        # Column 1: Raw deflections vs corrected deflections
        ax1 = axes[i, 0]
        ax1.plot(sample_deflections[i], color=colors[i], alpha=0.7, 
                label=f'Raw Deflections (Sample {i+1})')
        ax1.plot(inference_results['corrected_deflections'][i], 
                color=colors[i], linestyle='--', alpha=0.9,
                label=f'Corrected Deflections')
        ax1.set_title(f'Stage 1: Zero Function Correction (Sample {i+1})')
        ax1.set_xlabel('Rotation Number')
        ax1.set_ylabel('Deflection Amplitude')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Column 2: Estimated zero functions
        ax2 = axes[i, 1]
        ax2.plot(inference_results['estimated_baselines'][i], color=colors[i], linewidth=2)
        ax2.set_title(f'Estimated Zero Function (Sample {i+1})')
        ax2.set_xlabel('Rotation Number')
        ax2.set_ylabel('Zero Function Amplitude')
        ax2.grid(True, alpha=0.3)
        
        # Column 3: Final clean vibrations
        ax3 = axes[i, 2]
        ax3.plot(inference_results['corrected_deflections'][i], 
                color=colors[i], alpha=0.6, label='Corrected (Noisy)')
        ax3.plot(inference_results['clean_vibrations'][i], 
                color=colors[i], linewidth=2, label='Clean Vibrations')
        ax3.set_title(f'Stage 2: Vibration Denoising (Sample {i+1})')
        ax3.set_xlabel('Rotation Number')
        ax3.set_ylabel('Vibration Amplitude')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    viz_path = os.path.join(save_dir, 'demo_pipeline_visualization.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a second figure showing frequency domain analysis
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Frequency Domain Analysis - Pipeline Effectiveness', fontsize=14)
    
    # Analyze first sample
    sample_idx = 0
    
    # FFT analysis
    from scipy.fft import fft, fftfreq
    
    n_points = len(sample_deflections[sample_idx])
    freqs = fftfreq(n_points)[:n_points//2]
    
    # Raw deflections FFT
    fft_raw = np.abs(fft(sample_deflections[sample_idx]))[:n_points//2]
    axes[0].plot(freqs, fft_raw)
    axes[0].set_title('Raw Deflections - Frequency Domain')
    axes[0].set_xlabel('Normalized Frequency')
    axes[0].set_ylabel('Magnitude')
    axes[0].grid(True, alpha=0.3)
    
    # Corrected deflections FFT
    fft_corrected = np.abs(fft(inference_results['corrected_deflections'][sample_idx]))[:n_points//2]
    axes[1].plot(freqs, fft_corrected)
    axes[1].set_title('Corrected Deflections - Frequency Domain')
    axes[1].set_xlabel('Normalized Frequency')
    axes[1].set_ylabel('Magnitude')
    axes[1].grid(True, alpha=0.3)
    
    # Clean vibrations FFT
    fft_clean = np.abs(fft(inference_results['clean_vibrations'][sample_idx]))[:n_points//2]
    axes[2].plot(freqs, fft_clean)
    axes[2].set_title('Clean Vibrations - Frequency Domain')
    axes[2].set_xlabel('Normalized Frequency')
    axes[2].set_ylabel('Magnitude')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    freq_path = os.path.join(save_dir, 'demo_frequency_analysis.png')
    plt.savefig(freq_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  📊 Pipeline visualization saved: {viz_path}")
    print(f"  📊 Frequency analysis saved: {freq_path}")


def create_test_framework():
    """
    Create a testing framework to validate the implementation
    """
    print("🔧 TESTING FRAMEWORK")
    print("-" * 30)
    
    # Test 1: Data preparation
    print("Test 1: Data Preparation...")
    try:
        deflexions_4d, baselines_4d, vibrations_4d = create_synthetic_tip_timing_data(
            n_aubes=8, n_capteurs=2, n_tours=100, n_regimes=2
        )
        processor = TipTimingDataProcessor()
        prepared_data = processor.prepare_for_sequential_training(
            deflexions_4d, baselines_4d, vibrations_4d
        )
        print("✓ Data preparation test passed")
    except Exception as e:
        print(f"✗ Data preparation test failed: {e}")
    
    # Test 2: Model initialization
    print("Test 2: Model Initialization...")
    try:
        from models.nnfit_tip_timing import NNFitTipTiming
        from models.unet1d_tip_timing import Unet1DTipTiming
        
        nnfit = NNFitTipTiming(n_tours=100)
        nnfit.build_model()
        
        unet = Unet1DTipTiming(n_tours=100)
        unet.build_model()
        
        print("✓ Model initialization test passed")
    except Exception as e:
        print(f"✗ Model initialization test failed: {e}")
    
    # Test 3: Pipeline creation
    print("Test 3: Pipeline Creation...")
    try:
        pipeline = SequentialTipTimingPipeline(n_tours=100, save_dir='./test_models')
        print("✓ Pipeline creation test passed")
    except Exception as e:
        print(f"✗ Pipeline creation test failed: {e}")
    
    print("✓ Testing framework validation completed")
    print()


if __name__ == "__main__":
    print("NNfit1DRes Methodology Adapted for Tip Timing Analysis")
    print("=" * 60)
    print("🔬 Research Paper Implementation:")
    print("   'Deep neural network as the novel pipelines in multiple")
    print("    preprocessing for Raman Spectroscopy NNfit1DRes'")
    print("🔧 Adaptation Target:")
    print("   Turbomachine blade vibration analysis via tip timing")
    print("=" * 60)
    print()
    
    # Run testing framework first
    create_test_framework()
    
    # Run complete demonstration
    try:
        run_complete_demo()
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n👋 Demo script completed!")