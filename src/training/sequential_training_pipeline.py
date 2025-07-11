"""
Sequential Training Pipeline for Tip Timing Analysis
Adapted from Raman spectroscopy NNfit1DRes methodology

This module implements the complete two-stage sequential training pipeline:
1. NNFit training for zero function estimation (baseline correction)
2. Unet1D training for vibration denoising (using corrected deflections)

The pipeline follows the methodology described in the deep neural network
preprocessing paper, adapted for turbomachine blade vibration analysis.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json

# Import our custom models and data utilities
import sys
sys.path.append('..')
from models.nnfit_tip_timing import NNFitTipTiming
from models.unet1d_tip_timing import Unet1DTipTiming
from data.data_preparation import TipTimingDataProcessor


class SequentialTipTimingPipeline:
    """
    Complete sequential training pipeline for tip timing analysis
    
    This pipeline implements the adapted NNfit1DRes methodology for tip timing:
    1. First stage: Train NNFit to estimate zero functions (baseline correction)
    2. Second stage: Train Unet1D to denoise corrected vibrations
    
    The sequential approach ensures that vibration denoising is performed on
    properly baseline-corrected signals, maximizing the effectiveness of both models.
    """
    
    def __init__(self,
                 n_tours: int = 1000,
                 nnfit_params: Optional[Dict] = None,
                 unet_params: Optional[Dict] = None,
                 save_dir: str = "./models_trained"):
        """
        Initialize the sequential training pipeline
        
        Args:
            n_tours: Number of rotations in time series
            nnfit_params: Parameters for NNFit model
            unet_params: Parameters for Unet1D model
            save_dir: Directory to save trained models and results
        """
        self.n_tours = n_tours
        self.save_dir = save_dir
        
        # Default parameters for models
        self.nnfit_params = nnfit_params or {
            'learning_rate': 1e-4,
            'dropout_rate': 0.1,
            'use_batch_norm': True
        }
        
        self.unet_params = unet_params or {
            'learning_rate': 1e-6,
            'padding_size': 4,
            'use_batch_norm': True,
            'dropout_rate': 0.1
        }
        
        # Initialize models
        self.nnfit_model = None
        self.unet_model = None
        
        # Training histories
        self.nnfit_history = None
        self.unet_history = None
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training metadata
        self.training_metadata = {
            'pipeline_version': '1.0',
            'adapted_from': 'NNfit1DRes_Raman_spectroscopy',
            'application': 'tip_timing_analysis',
            'created_at': datetime.now().isoformat()
        }
    
    def train_nnfit_stage(self,
                         X_train: np.ndarray,
                         y_train: np.ndarray,
                         X_val: np.ndarray,
                         y_val: np.ndarray,
                         epochs: int = 100,
                         batch_size: int = 32,
                         verbose: int = 1) -> Dict[str, Any]:
        """
        Stage 1: Train NNFit model for zero function estimation
        
        This stage learns to estimate the systematic baseline errors (zero functions)
        introduced by the center-time approximation in tip timing measurements.
        
        Args:
            X_train: Training deflection measurements
            y_train: Training zero functions (true baselines)
            X_val: Validation deflection measurements
            y_val: Validation zero functions
            epochs: Maximum training epochs
            batch_size: Training batch size
            verbose: Verbosity level
            
        Returns:
            Training history dictionary
        """
        print("=" * 60)
        print("STAGE 1: NNFit Training - Zero Function Estimation")
        print("=" * 60)
        print("Objective: Learn systematic baseline errors in tip timing measurements")
        print("Physical meaning: Correct center-time approximation errors")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Sequence length: {X_train.shape[1]} rotations")
        print("")
        
        # Initialize NNFit model
        self.nnfit_model = NNFitTipTiming(
            n_tours=self.n_tours,
            **self.nnfit_params
        )
        
        # Train the model
        self.nnfit_history = self.nnfit_model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        
        # Save the trained model
        nnfit_save_path = os.path.join(self.save_dir, "nnfit_tip_timing_final.h5")
        self.nnfit_model.save_model(nnfit_save_path)
        
        print(f"✓ NNFit model saved to: {nnfit_save_path}")
        print("✓ Stage 1 completed successfully")
        print("")
        
        return self.nnfit_history
    
    def prepare_corrected_deflections(self,
                                    X_train: np.ndarray,
                                    X_val: np.ndarray,
                                    X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply trained NNFit model to correct deflection measurements
        
        This intermediate step between Stage 1 and Stage 2 applies the learned
        zero function corrections to prepare clean input for vibration denoising.
        
        Args:
            X_train: Training deflection measurements
            X_val: Validation deflection measurements
            X_test: Test deflection measurements
            
        Returns:
            Tuple of corrected deflections (train, val, test)
        """
        if self.nnfit_model is None:
            raise ValueError("NNFit model must be trained before correction")
        
        print("Applying zero function corrections to deflection measurements...")
        
        # Apply corrections using the trained NNFit model
        X_train_corrected, _ = self.nnfit_model.correct_deflections(X_train)
        X_val_corrected, _ = self.nnfit_model.correct_deflections(X_val)
        X_test_corrected, _ = self.nnfit_model.correct_deflections(X_test)
        
        print(f"✓ Corrected {X_train.shape[0]} training samples")
        print(f"✓ Corrected {X_val.shape[0]} validation samples")
        print(f"✓ Corrected {X_test.shape[0]} test samples")
        print("✓ Deflections ready for vibration denoising")
        print("")
        
        return X_train_corrected, X_val_corrected, X_test_corrected
    
    def train_unet_stage(self,
                        X_train_corrected: np.ndarray,
                        y_train_vibrations: np.ndarray,
                        X_val_corrected: np.ndarray,
                        y_val_vibrations: np.ndarray,
                        epochs: int = 80,
                        batch_size: int = 32,
                        verbose: int = 1) -> Dict[str, Any]:
        """
        Stage 2: Train Unet1D model for vibration denoising
        
        This stage learns to extract clean blade vibration signals from the
        baseline-corrected deflection measurements, removing measurement noise
        while preserving critical vibration characteristics.
        
        Args:
            X_train_corrected: Corrected training deflections (input)
            y_train_vibrations: Clean training vibrations (target)
            X_val_corrected: Corrected validation deflections
            y_val_vibrations: Clean validation vibrations
            epochs: Maximum training epochs
            batch_size: Training batch size
            verbose: Verbosity level
            
        Returns:
            Training history dictionary
        """
        print("=" * 60)
        print("STAGE 2: Unet1D Training - Vibration Denoising")
        print("=" * 60)
        print("Objective: Extract clean blade vibrations from corrected deflections")
        print("Physical meaning: Remove measurement noise while preserving resonances")
        print(f"Training samples: {X_train_corrected.shape[0]}")
        print(f"Sequence length: {X_train_corrected.shape[1]} rotations")
        print("")
        
        # Initialize Unet1D model
        self.unet_model = Unet1DTipTiming(
            n_tours=self.n_tours,
            **self.unet_params
        )
        
        # Train the model
        self.unet_history = self.unet_model.train(
            X_train=X_train_corrected,
            y_train=y_train_vibrations,
            X_val=X_val_corrected,
            y_val=y_val_vibrations,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        
        # Save the trained model
        unet_save_path = os.path.join(self.save_dir, "unet_tip_timing_final.h5")
        self.unet_model.save_model(unet_save_path)
        
        print(f"✓ Unet1D model saved to: {unet_save_path}")
        print("✓ Stage 2 completed successfully")
        print("")
        
        return self.unet_history
    
    def complete_pipeline_training(self,
                                 prepared_data: Dict[str, Any],
                                 nnfit_epochs: int = 100,
                                 unet_epochs: int = 80,
                                 batch_size: int = 32) -> Dict[str, Any]:
        """
        Execute the complete sequential training pipeline
        
        This is the main entry point that orchestrates both training stages
        and produces a complete tip timing analysis system.
        
        Args:
            prepared_data: Output from TipTimingDataProcessor.prepare_for_sequential_training()
            nnfit_epochs: Epochs for NNFit training
            unet_epochs: Epochs for Unet1D training
            batch_size: Batch size for both stages
            
        Returns:
            Complete training results and evaluation metrics
        """
        print("🚀 Starting Complete Sequential Training Pipeline for Tip Timing Analysis")
        print("📊 Methodology: Adapted NNfit1DRes from Raman spectroscopy preprocessing")
        print("")
        
        # Extract data for training
        nnfit_data = prepared_data['nnfit']
        unet_data = prepared_data['unet']
        
        # STAGE 1: NNFit Training
        nnfit_history = self.train_nnfit_stage(
            X_train=nnfit_data['X_train'],
            y_train=nnfit_data['y_train'],
            X_val=nnfit_data['X_val'],
            y_val=nnfit_data['y_val'],
            epochs=nnfit_epochs,
            batch_size=batch_size
        )
        
        # INTERMEDIATE: Apply zero function corrections
        X_train_corrected, X_val_corrected, X_test_corrected = self.prepare_corrected_deflections(
            X_train=nnfit_data['X_train'],
            X_val=nnfit_data['X_val'],
            X_test=nnfit_data['X_test']
        )
        
        # STAGE 2: Unet1D Training
        unet_history = self.train_unet_stage(
            X_train_corrected=X_train_corrected,
            y_train_vibrations=unet_data['y_train'],
            X_val_corrected=X_val_corrected,
            y_val_vibrations=unet_data['y_val'],
            epochs=unet_epochs,
            batch_size=batch_size
        )
        
        # EVALUATION: Comprehensive pipeline evaluation
        evaluation_results = self.evaluate_complete_pipeline(
            X_test_raw=nnfit_data['X_test'],
            X_test_corrected=X_test_corrected,
            y_test_baselines=nnfit_data['y_test'],
            y_test_vibrations=unet_data['y_test']
        )
        
        # SAVE: Complete pipeline results
        pipeline_results = self.save_pipeline_results(
            nnfit_history, unet_history, evaluation_results, prepared_data
        )
        
        print("🎉 Sequential Training Pipeline Completed Successfully!")
        print(f"📁 Results saved to: {self.save_dir}")
        print("")
        
        return pipeline_results
    
    def evaluate_complete_pipeline(self,
                                 X_test_raw: np.ndarray,
                                 X_test_corrected: np.ndarray,
                                 y_test_baselines: np.ndarray,
                                 y_test_vibrations: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive evaluation of the complete pipeline
        
        Args:
            X_test_raw: Raw test deflections
            X_test_corrected: Corrected test deflections
            y_test_baselines: True test baselines
            y_test_vibrations: True test vibrations
            
        Returns:
            Complete evaluation metrics
        """
        print("📊 Evaluating Complete Pipeline Performance...")
        
        evaluation = {}
        
        # Stage 1 Evaluation: Zero function estimation quality
        if self.nnfit_model is not None:
            nnfit_metrics = self.nnfit_model.evaluate_correction_quality(
                X_test_raw, y_test_baselines
            )
            evaluation['nnfit_performance'] = nnfit_metrics
            print(f"✓ NNFit RMSE: {nnfit_metrics['rmse_baseline']:.6f}")
            print(f"✓ NNFit Accuracy Rate: {nnfit_metrics['accuracy_rate_percent']:.2f}%")
        
        # Stage 2 Evaluation: Vibration denoising quality
        if self.unet_model is not None:
            unet_metrics = self.unet_model.evaluate_denoising_performance(
                X_test_corrected, y_test_vibrations
            )
            evaluation['unet_performance'] = unet_metrics
            print(f"✓ Unet RMSE: {unet_metrics['rmse']:.6f}")
            print(f"✓ Unet SNR: {unet_metrics['snr_db']:.2f} dB")
        
        # End-to-end evaluation: Full pipeline performance
        if self.nnfit_model is not None and self.unet_model is not None:
            # Get final clean vibrations from complete pipeline
            final_vibrations = self.unet_model.denoise_vibrations(X_test_corrected)
            
            # Calculate end-to-end metrics
            e2e_rmse = float(np.sqrt(np.mean((y_test_vibrations - final_vibrations)**2)))
            e2e_corr = float(np.corrcoef(
                y_test_vibrations.flatten(), 
                final_vibrations.flatten()
            )[0, 1])
            
            evaluation['end_to_end'] = {
                'rmse': e2e_rmse,
                'correlation': e2e_corr,
                'samples_processed': len(y_test_vibrations)
            }
            
            print(f"✓ End-to-end RMSE: {e2e_rmse:.6f}")
            print(f"✓ End-to-end Correlation: {e2e_corr:.4f}")
        
        print("✓ Pipeline evaluation completed")
        print("")
        
        return evaluation
    
    def save_pipeline_results(self,
                            nnfit_history: Dict,
                            unet_history: Dict,
                            evaluation: Dict,
                            prepared_data: Dict) -> Dict[str, Any]:
        """
        Save complete pipeline results and metadata
        
        Args:
            nnfit_history: NNFit training history
            unet_history: Unet1D training history
            evaluation: Evaluation metrics
            prepared_data: Original prepared data info
            
        Returns:
            Complete results dictionary
        """
        # Package all results
        complete_results = {
            'training_metadata': self.training_metadata,
            'model_parameters': {
                'nnfit_params': self.nnfit_params,
                'unet_params': self.unet_params,
                'n_tours': self.n_tours
            },
            'training_histories': {
                'nnfit': nnfit_history,
                'unet': unet_history
            },
            'evaluation_metrics': evaluation,
            'data_info': prepared_data['info']
        }
        
        # Save results to JSON
        results_path = os.path.join(self.save_dir, "pipeline_results.json")
        with open(results_path, 'w') as f:
            json.dump(complete_results, f, indent=2, default=str)
        
        print(f"📁 Complete results saved to: {results_path}")
        
        # Create training plots
        self.create_training_plots()
        
        return complete_results
    
    def create_training_plots(self):
        """Create and save training visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Sequential Training Pipeline Results - Tip Timing Analysis', fontsize=16)
        
        # NNFit training plots
        if self.nnfit_history is not None:
            # Loss plot
            axes[0, 0].plot(self.nnfit_history['loss'], label='Training Loss')
            axes[0, 0].plot(self.nnfit_history['val_loss'], label='Validation Loss')
            axes[0, 0].set_title('NNFit Training - Zero Function Estimation')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss (MSE)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # MAE plot
            axes[0, 1].plot(self.nnfit_history['mae'], label='Training MAE')
            axes[0, 1].plot(self.nnfit_history['val_mae'], label='Validation MAE')
            axes[0, 1].set_title('NNFit MAE Evolution')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Unet training plots
        if self.unet_history is not None:
            # Loss plot
            axes[1, 0].plot(self.unet_history['loss'], label='Training Loss')
            axes[1, 0].plot(self.unet_history['val_loss'], label='Validation Loss')
            axes[1, 0].set_title('Unet1D Training - Vibration Denoising')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss (Custom)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # MAE plot
            axes[1, 1].plot(self.unet_history['mae'], label='Training MAE')
            axes[1, 1].plot(self.unet_history['val_mae'], label='Validation MAE')
            axes[1, 1].set_title('Unet1D MAE Evolution')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('MAE')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, "training_plots.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training plots saved to: {plot_path}")
    
    def predict_pipeline(self, raw_deflections: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply the complete trained pipeline to new data
        
        Args:
            raw_deflections: Raw deflection measurements to process
            
        Returns:
            Dictionary with corrected deflections and clean vibrations
        """
        if self.nnfit_model is None or self.unet_model is None:
            raise ValueError("Both models must be trained before prediction")
        
        # Stage 1: Correct deflections
        corrected_deflections, estimated_baselines = self.nnfit_model.correct_deflections(
            raw_deflections
        )
        
        # Stage 2: Denoise vibrations
        clean_vibrations = self.unet_model.denoise_vibrations(corrected_deflections)
        
        return {
            'raw_deflections': raw_deflections,
            'estimated_baselines': estimated_baselines,
            'corrected_deflections': corrected_deflections,
            'clean_vibrations': clean_vibrations
        }


def create_training_pipeline(n_tours: int = 1000,
                           save_dir: str = "./models_trained",
                           **kwargs) -> SequentialTipTimingPipeline:
    """
    Factory function to create a sequential training pipeline
    
    Args:
        n_tours: Number of rotations in time series
        save_dir: Directory to save results
        **kwargs: Additional pipeline parameters
        
    Returns:
        Configured SequentialTipTimingPipeline instance
    """
    return SequentialTipTimingPipeline(n_tours=n_tours, save_dir=save_dir, **kwargs)