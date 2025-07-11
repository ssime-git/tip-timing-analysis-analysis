"""
Data Preparation Utilities for Tip Timing Analysis
Adapted from Raman spectroscopy data preprocessing to tip timing signal processing

This module provides utilities for preparing tip timing data for deep learning models,
including the critical 4D to 2D transformation and normalization procedures required
for the NNfit1DRes methodology adaptation.

Functions handle the multi-dimensional nature of tip timing data:
- Blades × Sensors × Rotations × Operating_Speeds → Training samples
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Any
from sklearn.model_selection import train_test_split
import warnings


class TipTimingDataProcessor:
    """
    Comprehensive data processor for tip timing measurements
    
    Handles the transformation from 4D tip timing data structure to the 2D format
    required by neural networks, while maintaining physical meaning and traceability.
    """
    
    def __init__(self):
        self.metadata_mapping = {}
        self.normalization_params = {}
        
    def reshape_tip_timing_data(self, 
                              deflexions_4d: np.ndarray,
                              baselines_4d: np.ndarray,
                              vibrations_4d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Transform 4D tip timing data to 2D format required for neural network training
        
        This is the core transformation that converts the natural tip timing data structure
        into a format suitable for the adapted NNfit1DRes methodology.
        
        Args:
            deflexions_4d: Raw center-time deflections (n_aubes, n_capteurs, n_tours, n_regimes)
            baselines_4d: True zero functions (n_aubes, n_capteurs, n_tours, n_regimes)
            vibrations_4d: Clean vibration signals (n_aubes, n_capteurs, n_tours, n_regimes)
            
        Returns:
            Tuple of:
            - X_deflexions: Reshaped deflections (n_samples, n_tours)
            - Y_baselines: Reshaped baselines (n_samples, n_tours)
            - Y_vibrations: Reshaped vibrations (n_samples, n_tours)
            - metadata: List of dictionaries with sample traceability
        """
        n_aubes, n_capteurs, n_tours, n_regimes = deflexions_4d.shape
        n_samples = n_aubes * n_capteurs * n_regimes
        
        print(f"Reshaping tip timing data from 4D to 2D:")
        print(f"  Input shape: ({n_aubes} aubes, {n_capteurs} capteurs, {n_tours} tours, {n_regimes} regimes)")
        print(f"  Output shape: ({n_samples} samples, {n_tours} tours)")
        
        # Validate input shapes
        if not (deflexions_4d.shape == baselines_4d.shape == vibrations_4d.shape):
            raise ValueError("All input arrays must have the same shape")
        
        # Reshape: (aubes, capteurs, tours, regimes) -> (samples, tours)
        # Transpose to put tours last, then reshape
        X_deflexions = deflexions_4d.transpose(0, 1, 3, 2).reshape(n_samples, n_tours)
        Y_baselines = baselines_4d.transpose(0, 1, 3, 2).reshape(n_samples, n_tours)
        Y_vibrations = vibrations_4d.transpose(0, 1, 3, 2).reshape(n_samples, n_tours)
        
        # Create metadata for traceability
        metadata = []
        sample_idx = 0
        for aube in range(n_aubes):
            for capteur in range(n_capteurs):
                for regime in range(n_regimes):
                    metadata.append({
                        'sample_id': sample_idx,
                        'aube_id': aube,
                        'capteur_id': capteur,
                        'regime_id': regime,
                        'physical_meaning': f'Aube_{aube}_Capteur_{capteur}_Regime_{regime}'
                    })
                    sample_idx += 1
        
        # Store metadata mapping
        self.metadata_mapping = {i: meta for i, meta in enumerate(metadata)}
        
        print(f"  Created {len(metadata)} training samples with full traceability")
        
        return X_deflexions, Y_baselines, Y_vibrations, metadata
    
    def normalize_aube_signal(self, signal_aube: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Normalize individual blade signal according to tip timing requirements
        
        This normalization is crucial for the adapted NNfit1DRes methodology as it allows
        the models to focus on signal shapes rather than absolute amplitudes, essential
        for generalization across different blades and operating conditions.
        
        Args:
            signal_aube: Single blade signal sequence (n_tours,)
            
        Returns:
            Tuple of (normalized_signal, normalization_params)
        """
        mean_val = np.mean(signal_aube)
        std_val = np.std(signal_aube)
        
        # Avoid division by zero
        if std_val == 0:
            warnings.warn("Zero standard deviation detected. Using signal as-is.")
            return signal_aube, {'mean': mean_val, 'std': 1.0}
        
        normalized = (signal_aube - mean_val) / std_val
        
        return normalized, {'mean': mean_val, 'std': std_val}
    
    def normalize_dataset(self, 
                         X_deflexions: np.ndarray,
                         Y_baselines: np.ndarray,
                         Y_vibrations: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply per-sample normalization to the entire dataset
        
        Critical for tip timing analysis where:
        - Vibration amplitudes vary dramatically with operating conditions
        - Different blades have different response characteristics
        - Sensors may have different sensitivities
        
        Args:
            X_deflexions: Raw deflection measurements (n_samples, n_tours)
            Y_baselines: Baseline functions (n_samples, n_tours)
            Y_vibrations: Clean vibrations (n_samples, n_tours)
            
        Returns:
            Tuple of normalized arrays
        """
        print("Applying per-sample normalization for tip timing analysis...")
        
        n_samples = X_deflexions.shape[0]
        
        # Initialize output arrays
        X_norm = np.zeros_like(X_deflexions)
        Y_baselines_norm = np.zeros_like(Y_baselines)
        Y_vibrations_norm = np.zeros_like(Y_vibrations)
        
        # Store normalization parameters for each sample
        normalization_params = {}
        
        for i in range(n_samples):
            # Normalize each sample independently
            X_norm[i], params_x = self.normalize_aube_signal(X_deflexions[i])
            Y_baselines_norm[i], params_y_base = self.normalize_aube_signal(Y_baselines[i])
            Y_vibrations_norm[i], params_y_vib = self.normalize_aube_signal(Y_vibrations[i])
            
            # Store parameters for potential denormalization
            normalization_params[i] = {
                'deflexions': params_x,
                'baselines': params_y_base,
                'vibrations': params_y_vib
            }
        
        # Store normalization parameters
        self.normalization_params = normalization_params
        
        print(f"  Normalized {n_samples} samples individually")
        print(f"  This preserves relative signal shapes while enabling cross-blade learning")
        
        return X_norm, Y_baselines_norm, Y_vibrations_norm
    
    def create_stratified_split(self,
                               X: np.ndarray,
                               Y_baseline: np.ndarray,
                               Y_vibration: np.ndarray,
                               metadata: List[Dict],
                               test_size: float = 0.2,
                               val_size: float = 0.1,
                               random_state: int = 42) -> Tuple:
        """
        Create stratified train/validation/test split preserving blade/sensor diversity
        
        For tip timing analysis, it's crucial that the split maintains representation of:
        - All blades across training/validation/test sets
        - All sensors across training/validation/test sets
        - All operating regimes across training/validation/test sets
        
        Args:
            X: Input deflection data
            Y_baseline: Baseline target data
            Y_vibration: Vibration target data
            metadata: Sample metadata for stratification
            test_size: Proportion for test set
            val_size: Proportion for validation set
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_val, X_test, Y_base_train, Y_base_val, Y_base_test,
                     Y_vib_train, Y_vib_val, Y_vib_test, meta_train, meta_val, meta_test)
        """
        print("Creating stratified split for tip timing data...")
        
        # Create stratification labels based on blade and sensor combination
        strat_labels = []
        for meta in metadata:
            # Combine blade and sensor ID for stratification
            strat_label = f"{meta['aube_id']}_{meta['capteur_id']}"
            strat_labels.append(strat_label)
        
        strat_labels = np.array(strat_labels)
        
        # First split: train+val vs test
        train_val_size = 1.0 - test_size
        
        X_train_val, X_test, Y_base_train_val, Y_base_test, Y_vib_train_val, Y_vib_test, \
        strat_train_val, strat_test, meta_train_val, meta_test = train_test_split(
            X, Y_baseline, Y_vibration, strat_labels, metadata,
            test_size=test_size,
            stratify=strat_labels,
            random_state=random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / train_val_size
        
        X_train, X_val, Y_base_train, Y_base_val, Y_vib_train, Y_vib_val, \
        strat_train, strat_val, meta_train, meta_val = train_test_split(
            X_train_val, Y_base_train_val, Y_vib_train_val, strat_train_val, meta_train_val,
            test_size=val_size_adjusted,
            stratify=strat_train_val,
            random_state=random_state
        )
        
        print(f"  Training set: {len(X_train)} samples")
        print(f"  Validation set: {len(X_val)} samples")
        print(f"  Test set: {len(X_test)} samples")
        
        # Verify stratification worked
        self._verify_stratification(meta_train, meta_val, meta_test)
        
        return (X_train, X_val, X_test, 
                Y_base_train, Y_base_val, Y_base_test,
                Y_vib_train, Y_vib_val, Y_vib_test,
                meta_train, meta_val, meta_test)
    
    def _verify_stratification(self, meta_train: List[Dict], meta_val: List[Dict], meta_test: List[Dict]):
        """Verify that stratification preserved diversity across splits"""
        
        def get_unique_values(metadata, key):
            return set(meta[key] for meta in metadata)
        
        train_aubes = get_unique_values(meta_train, 'aube_id')
        val_aubes = get_unique_values(meta_val, 'aube_id')
        test_aubes = get_unique_values(meta_test, 'aube_id')
        
        train_capteurs = get_unique_values(meta_train, 'capteur_id')
        val_capteurs = get_unique_values(meta_val, 'capteur_id')
        test_capteurs = get_unique_values(meta_test, 'capteur_id')
        
        print(f"  Blade diversity - Train: {len(train_aubes)}, Val: {len(val_aubes)}, Test: {len(test_aubes)}")
        print(f"  Sensor diversity - Train: {len(train_capteurs)}, Val: {len(val_capteurs)}, Test: {len(test_capteurs)}")
        
        # Check if all sets have reasonable diversity
        total_aubes = train_aubes.union(val_aubes).union(test_aubes)
        total_capteurs = train_capteurs.union(val_capteurs).union(test_capteurs)
        
        if len(train_aubes) < len(total_aubes) * 0.5:
            warnings.warn("Training set may not have sufficient blade diversity")
        if len(train_capteurs) < len(total_capteurs) * 0.5:
            warnings.warn("Training set may not have sufficient sensor diversity")
    
    def prepare_for_sequential_training(self,
                                      deflexions_4d: np.ndarray,
                                      baselines_4d: np.ndarray,
                                      vibrations_4d: np.ndarray,
                                      test_size: float = 0.2,
                                      val_size: float = 0.1,
                                      normalize: bool = True) -> Dict[str, Any]:
        """
        Complete data preparation pipeline for sequential NNfit + Unet training
        
        This is the main entry point that performs all necessary data transformations
        for the adapted NNfit1DRes methodology in tip timing analysis.
        
        Args:
            deflexions_4d: Raw 4D deflection data
            baselines_4d: True 4D baseline data
            vibrations_4d: Clean 4D vibration data
            test_size: Test set proportion
            val_size: Validation set proportion
            normalize: Whether to apply normalization
            
        Returns:
            Dictionary containing all prepared datasets and metadata
        """
        print("=== Starting Complete Tip Timing Data Preparation Pipeline ===")
        
        # Step 1: 4D to 2D transformation
        X_deflexions, Y_baselines, Y_vibrations, metadata = self.reshape_tip_timing_data(
            deflexions_4d, baselines_4d, vibrations_4d
        )
        
        # Step 2: Normalization (if requested)
        if normalize:
            X_deflexions, Y_baselines, Y_vibrations = self.normalize_dataset(
                X_deflexions, Y_baselines, Y_vibrations
            )
        
        # Step 3: Stratified splitting
        data_splits = self.create_stratified_split(
            X_deflexions, Y_baselines, Y_vibrations, metadata,
            test_size=test_size, val_size=val_size
        )
        
        (X_train, X_val, X_test, 
         Y_base_train, Y_base_val, Y_base_test,
         Y_vib_train, Y_vib_val, Y_vib_test,
         meta_train, meta_val, meta_test) = data_splits
        
        # Package everything for easy access
        prepared_data = {
            # Training data for NNFit (baseline estimation)
            'nnfit': {
                'X_train': X_train,
                'y_train': Y_base_train,
                'X_val': X_val,
                'y_val': Y_base_val,
                'X_test': X_test,
                'y_test': Y_base_test
            },
            
            # Training data for Unet (vibration denoising)
            # Note: X will be corrected deflections after NNFit processing
            'unet': {
                'y_train': Y_vib_train,  # Clean vibrations (target)
                'y_val': Y_vib_val,
                'y_test': Y_vib_test
            },
            
            # Metadata and traceability
            'metadata': {
                'train': meta_train,
                'val': meta_val,
                'test': meta_test,
                'normalization_params': self.normalization_params if normalize else None
            },
            
            # Data characteristics
            'info': {
                'n_tours': X_train.shape[1],
                'n_train_samples': len(X_train),
                'n_val_samples': len(X_val),
                'n_test_samples': len(X_test),
                'normalized': normalize,
                'original_4d_shape': deflexions_4d.shape
            }
        }
        
        print("=== Data Preparation Pipeline Completed ===")
        print(f"Ready for sequential training:")
        print(f"  1. NNFit: {prepared_data['info']['n_train_samples']} training samples")
        print(f"  2. Unet: Will use NNFit-corrected deflections")
        print(f"  Sequence length: {prepared_data['info']['n_tours']} rotations")
        
        return prepared_data
    
    def denormalize_signal(self, 
                          normalized_signal: np.ndarray,
                          sample_id: int,
                          signal_type: str = 'deflexions') -> np.ndarray:
        """
        Denormalize a signal using stored normalization parameters
        
        Args:
            normalized_signal: The normalized signal to denormalize
            sample_id: The sample ID to get normalization parameters for
            signal_type: Type of signal ('deflexions', 'baselines', 'vibrations')
            
        Returns:
            Denormalized signal
        """
        if sample_id not in self.normalization_params:
            raise ValueError(f"No normalization parameters found for sample {sample_id}")
        
        params = self.normalization_params[sample_id][signal_type]
        mean_val = params['mean']
        std_val = params['std']
        
        return normalized_signal * std_val + mean_val


def create_synthetic_tip_timing_data(n_aubes: int = 24,
                                   n_capteurs: int = 4,
                                   n_tours: int = 1000,
                                   n_regimes: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create synthetic tip timing data for testing the adapted NNfit1DRes methodology
    
    This function generates realistic synthetic data that mimics real tip timing
    measurements with proper physical characteristics.
    
    Args:
        n_aubes: Number of blades
        n_capteurs: Number of sensors
        n_tours: Number of rotations
        n_regimes: Number of operating regimes
        
    Returns:
        Tuple of (deflexions_4d, baselines_4d, vibrations_4d)
    """
    print(f"Generating synthetic tip timing data:")
    print(f"  {n_aubes} aubes, {n_capteurs} capteurs, {n_tours} tours, {n_regimes} regimes")
    
    # Initialize arrays
    shape = (n_aubes, n_capteurs, n_tours, n_regimes)
    vibrations_4d = np.zeros(shape)
    baselines_4d = np.zeros(shape)
    
    # Generate time axis
    t = np.linspace(0, 2*np.pi, n_tours)
    
    for aube in range(n_aubes):
        for capteur in range(n_capteurs):
            for regime in range(n_regimes):
                
                # Generate blade-specific vibration characteristics
                blade_freq = 1.0 + 0.1 * np.random.randn()  # Natural frequency variation
                regime_amplitude = 0.5 + regime * 0.3  # Amplitude depends on regime
                
                # Multi-modal vibration (several harmonic components)
                vibration = (
                    regime_amplitude * np.sin(blade_freq * t) +
                    0.3 * regime_amplitude * np.sin(2 * blade_freq * t) +
                    0.1 * regime_amplitude * np.sin(3 * blade_freq * t) +
                    0.05 * np.random.randn(n_tours)  # Small amount of noise
                )
                
                # Generate baseline (zero function) - systematic error
                baseline_trend = 0.2 * np.sin(0.5 * t) + 0.1 * np.cos(0.3 * t)
                baseline_offset = 0.1 * (aube - n_aubes/2) / n_aubes  # Blade-dependent offset
                baseline = baseline_trend + baseline_offset
                
                vibrations_4d[aube, capteur, :, regime] = vibration
                baselines_4d[aube, capteur, :, regime] = baseline
    
    # Deflections = vibrations + baselines + noise
    noise = 0.02 * np.random.randn(*shape)
    deflexions_4d = vibrations_4d + baselines_4d + noise
    
    print(f"  Generated realistic synthetic data with:")
    print(f"    - Multi-modal blade vibrations")
    print(f"    - Regime-dependent amplitudes")
    print(f"    - Systematic baseline errors")
    print(f"    - Measurement noise")
    
    return deflexions_4d, baselines_4d, vibrations_4d