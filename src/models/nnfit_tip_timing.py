"""
NNFit Model for Tip Timing Analysis
Adapted from Raman spectroscopy baseline correction to tip timing zero function estimation

This module implements the NNFit architecture specifically designed for estimating
the "zero function" in tip timing measurements. The zero function represents the
systematic error introduced by the center-time approximation in tip timing analysis.

Original inspiration: Deep neural network pipelines for Raman spectroscopy preprocessing
Adaptation: Zero function estimation for turbomachine blade deflection measurements
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
from typing import Tuple, Optional, Dict, Any


class NNFitTipTiming:
    """
    Neural Network for Zero Function Estimation in Tip Timing Analysis
    
    This model estimates the systematic baseline error (zero function) that occurs
    in tip timing measurements due to the center-time approximation. The zero function
    depends on:
    - Angular position of blades relative to sensors
    - Rotational speed variations
    - Thermal and centrifugal effects
    - Shaft torsion and mounting dispersions
    
    Architecture inspired by NNfit1DRes from Raman spectroscopy research.
    """
    
    def __init__(self, 
                 n_tours: int = 1000,
                 learning_rate: float = 1e-4,
                 dropout_rate: float = 0.1,
                 use_batch_norm: bool = True):
        """
        Initialize NNFit model for tip timing zero function estimation
        
        Args:
            n_tours: Number of rotations in the time series (equivalent to spectral points)
            learning_rate: Learning rate for Adam optimizer
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
        """
        self.n_tours = n_tours
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.model = None
        self.history = None
        
    def build_model(self) -> tf.keras.Model:
        """
        Build the NNFit architecture optimized for tip timing zero function estimation
        
        Architecture design principles:
        - Dense layers to capture complex non-linear relationships
        - Progressive feature extraction from rotation-dependent patterns
        - Output dimensionality matches input for pointwise zero function estimation
        - Regularization to prevent overfitting on blade-specific patterns
        
        Returns:
            Compiled Keras model
        """
        model = Sequential(name='NNFit_TipTiming')
        
        # Input layer for center-time deflection measurements
        model.add(Input(shape=(self.n_tours,), name='deflexion_center_time'))
        
        # First feature extraction block
        model.add(Dense(1024, activation='relu', name='extraction_features_1'))
        if self.use_batch_norm:
            model.add(BatchNormalization(name='bn_1'))
        model.add(Dropout(self.dropout_rate, name='dropout_1'))
        
        # Complex analysis block - captures inter-rotation dependencies
        model.add(Dense(2048, activation='relu', name='analyse_complexe'))
        if self.use_batch_norm:
            model.add(BatchNormalization(name='bn_2'))
        model.add(Dropout(self.dropout_rate, name='dropout_2'))
        
        # Second feature extraction block
        model.add(Dense(1024, activation='relu', name='extraction_features_2'))
        if self.use_batch_norm:
            model.add(BatchNormalization(name='bn_3'))
        model.add(Dropout(self.dropout_rate, name='dropout_3'))
        
        # Additional deep feature extraction for complex zero function patterns
        model.add(Dense(512, activation='relu', name='deep_features'))
        if self.use_batch_norm:
            model.add(BatchNormalization(name='bn_4'))
        model.add(Dropout(self.dropout_rate / 2, name='dropout_4'))
        
        # Output layer - predicts zero function for each rotation
        model.add(Dense(self.n_tours, activation='linear', name='fonction_zero_estimee'))
        
        # Compile with optimizer adapted for vibrational signals
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        self.model = model
        return model
    
    def prepare_training_callbacks(self, 
                                 save_path: str = 'nnfit_tip_timing_best.h5') -> list:
        """
        Prepare training callbacks adapted for tip timing signal characteristics
        
        Args:
            save_path: Path to save the best model
            
        Returns:
            List of Keras callbacks
        """
        callbacks = [
            EarlyStopping(
                patience=15,
                restore_best_weights=True,
                monitor='val_loss',
                verbose=1,
                min_delta=1e-6
            ),
            ReduceLROnPlateau(
                patience=8,
                factor=0.5,
                min_lr=1e-7,
                verbose=1,
                monitor='val_loss'
            ),
            ModelCheckpoint(
                save_path,
                save_best_only=True,
                monitor='val_loss',
                verbose=1
            )
        ]
        return callbacks
    
    def train(self, 
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 100,
              batch_size: int = 32,
              verbose: int = 1) -> Dict[str, Any]:
        """
        Train the NNFit model for zero function estimation
        
        Args:
            X_train: Training deflection measurements (n_samples, n_tours)
            y_train: Training zero functions (n_samples, n_tours)
            X_val: Validation deflection measurements
            y_val: Validation zero functions
            epochs: Maximum number of training epochs
            batch_size: Training batch size
            verbose: Verbosity level
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.build_model()
            
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            
        # Prepare callbacks
        callbacks = self.prepare_training_callbacks()
        
        print("=== Starting NNFit Training for Zero Function Estimation ===")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Sequence length: {X_train.shape[1]} rotations")
        if validation_data:
            print(f"Validation samples: {X_val.shape[0]}")
        print("Objective: Learn systematic baseline errors in tip timing measurements")
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        print("=== NNFit Training Completed ===")
        return self.history.history
    
    def predict_zero_function(self, X: np.ndarray) -> np.ndarray:
        """
        Predict zero functions for given deflection measurements
        
        Args:
            X: Input deflection measurements (n_samples, n_tours)
            
        Returns:
            Predicted zero functions (n_samples, n_tours)
        """
        if self.model is None:
            raise ValueError("Model must be built and trained before prediction")
            
        return self.model.predict(X, verbose=0)
    
    def correct_deflections(self, 
                          deflections_center_time: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply zero function correction to center-time deflection measurements
        
        This is the core functionality that removes systematic errors from tip timing
        measurements, providing corrected deflections ready for vibration analysis.
        
        Args:
            deflections_center_time: Raw center-time deflection measurements
            
        Returns:
            Tuple of (corrected_deflections, estimated_zero_functions)
        """
        # Predict zero functions
        zero_functions = self.predict_zero_function(deflections_center_time)
        
        # Subtract zero function to get corrected deflections
        corrected_deflections = deflections_center_time - zero_functions
        
        return corrected_deflections, zero_functions
    
    def evaluate_correction_quality(self, 
                                  X_test: np.ndarray,
                                  y_true: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the quality of zero function estimation
        
        Args:
            X_test: Test deflection measurements
            y_true: True zero functions
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
            
        # Predict zero functions
        y_pred = self.predict_zero_function(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Accuracy rate (adapted from Raman spectroscopy paper)
        ac_rate = 1 - (np.sqrt(np.mean((y_true - y_pred)**2)) / 
                      np.sqrt(np.mean(y_true**2))) if np.mean(y_true**2) > 0 else 0
        
        # Extrema error (critical for resonance passages)
        max_true = np.max(np.abs(y_true))
        max_pred = np.max(np.abs(y_pred))
        extrema_error = np.abs(max_true - max_pred) / max_true * 100 if max_true > 0 else 0
        
        # Stability of estimation (rotation-to-rotation variation)
        stability_true = np.std(np.diff(y_true, axis=1))
        stability_pred = np.std(np.diff(y_pred, axis=1))
        stability_ratio = stability_pred / stability_true if stability_true > 0 else 1
        
        return {
            'rmse_baseline': float(rmse),
            'mae_baseline': float(mae),
            'accuracy_rate_percent': float(ac_rate * 100),
            'extrema_error_percent': float(extrema_error),
            'stability_ratio': float(stability_ratio)
        }
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(filepath)
        print(f"NNFit model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a pre-trained model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"NNFit model loaded from {filepath}")
    
    def get_model_summary(self):
        """Print model architecture summary"""
        if self.model is None:
            self.build_model()
        return self.model.summary()


def normalize_aube_signal(signal_aube: np.ndarray) -> np.ndarray:
    """
    Normalize blade signal according to tip timing requirements
    
    This normalization allows the model to focus on signal shapes rather than
    absolute amplitudes, which is crucial for generalization across different
    blades and operating conditions.
    
    Args:
        signal_aube: Single blade signal sequence
        
    Returns:
        Normalized signal
    """
    return (signal_aube - np.mean(signal_aube)) / np.std(signal_aube)


def create_nnfit_tip_timing(n_tours: int = 1000, 
                           learning_rate: float = 1e-4,
                           **kwargs) -> NNFitTipTiming:
    """
    Factory function to create NNFit model for tip timing
    
    Args:
        n_tours: Number of rotations in sequence
        learning_rate: Learning rate for optimization
        **kwargs: Additional model parameters
        
    Returns:
        Configured NNFitTipTiming instance
    """
    return NNFitTipTiming(n_tours=n_tours, learning_rate=learning_rate, **kwargs)