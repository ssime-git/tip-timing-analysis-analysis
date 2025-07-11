"""
Unet1DRes Model for Tip Timing Vibration Denoising
Adapted from Raman spectroscopy denoising to tip timing vibration signal cleaning

This module implements a 1D U-Net architecture with residual connections specifically
designed for extracting clean blade vibration signals from noisy tip timing measurements.
The model preserves critical vibration characteristics essential for fatigue monitoring.

Original inspiration: 1D U-Net for spectral denoising in Raman spectroscopy
Adaptation: Blade vibration denoising for turbomachine health monitoring
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, UpSampling1D, Concatenate, 
    Input, Lambda, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
from typing import Tuple, Optional, Dict, Any, Callable


class Unet1DTipTiming:
    """
    1D U-Net with Residual Connections for Blade Vibration Denoising
    
    This model extracts clean blade vibration signals from corrected deflection
    measurements (after zero function removal). It's designed to:
    - Preserve resonance peaks critical for fatigue detection
    - Handle multi-modal vibration patterns
    - Respect the periodic nature of blade passages
    - Maintain amplitude accuracy across different vibration levels
    
    Architecture adapted from Unet1DRes for Raman spectroscopy denoising.
    """
    
    def __init__(self,
                 n_tours: int = 1000,
                 learning_rate: float = 1e-6,
                 padding_size: int = 4,
                 use_batch_norm: bool = True,
                 dropout_rate: float = 0.1):
        """
        Initialize Unet1D model for tip timing vibration denoising
        
        Args:
            n_tours: Number of rotations in the time series
            learning_rate: Learning rate for Adam optimizer (lower for fine-tuning)
            padding_size: Size of periodic padding to handle boundary effects
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate for regularization
        """
        self.n_tours = n_tours
        self.learning_rate = learning_rate
        self.padding_size = padding_size
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        
    def periodic_padding(self, x: tf.Tensor, padding: int = 4) -> tf.Tensor:
        """
        Apply periodic padding to respect the cyclical nature of blade passages
        
        In tip timing, blade passages are inherently periodic. This padding helps
        the convolution operations understand the cyclical boundary conditions.
        
        Args:
            x: Input tensor (batch_size, sequence_length, channels)
            padding: Padding size on each side
            
        Returns:
            Padded tensor
        """
        return tf.concat([x[:, -padding:, :], x, x[:, :padding, :]], axis=1)
    
    def create_loss_function(self) -> Callable:
        """
        Create custom loss function optimized for vibration signal preservation
        
        This hybrid loss function:
        - Standard MSE for overall signal fidelity
        - Weighted MSE for preserving high-amplitude regions (resonances)
        - Ensures critical vibration characteristics are maintained
        
        Returns:
            Custom loss function
        """
        def loss_preservation_resonances(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            """
            Hybrid loss function for preserving resonance characteristics
            """
            # Standard MSE component
            mse_standard = tf.reduce_mean(tf.square(y_true - y_pred))
            
            # Amplitude-weighted MSE (emphasizes resonance preservation)
            amplitude_weights = tf.abs(y_true) + 1.0
            mse_weighted = tf.reduce_mean(amplitude_weights * tf.square(y_true - y_pred))
            
            # Frequency domain preservation (optional for resonance peaks)
            fft_true = tf.signal.fft(tf.cast(tf.squeeze(y_true), tf.complex64))
            fft_pred = tf.signal.fft(tf.cast(tf.squeeze(y_pred), tf.complex64))
            spectral_loss = tf.reduce_mean(tf.square(tf.abs(fft_true) - tf.abs(fft_pred)))
            
            # Combine losses
            total_loss = 0.6 * mse_standard + 0.3 * mse_weighted + 0.1 * spectral_loss
            
            return total_loss
        
        return loss_preservation_resonances
    
    def build_model(self) -> tf.keras.Model:
        """
        Build the 1D U-Net architecture optimized for blade vibration denoising
        
        Architecture design principles:
        - Encoder-decoder structure for multi-scale feature extraction
        - Skip connections to preserve fine-grained vibration details
        - Periodic padding for cyclical blade passage patterns
        - Progressive downsampling and upsampling for noise separation
        - Residual learning approach (predict noise to subtract)
        
        Returns:
            Compiled Keras model
        """
        # Input layer for corrected vibration signals (with noise)
        inputs = Input(shape=(self.n_tours, 1), name='vibration_avec_bruit')
        
        # Apply periodic padding to handle boundary effects
        padded = Lambda(
            lambda x: self.periodic_padding(x, self.padding_size), 
            name='padding_cyclique'
        )(inputs)
        
        # =================================================================
        # ENCODER (Downsampling) - Multi-scale feature extraction
        # =================================================================
        
        # First convolution block - capture local vibration patterns
        conv1 = Conv1D(32, 5, activation='relu', padding='same', name='conv1a')(padded)
        if self.use_batch_norm:
            conv1 = BatchNormalization(name='bn1a')(conv1)
        conv1 = Conv1D(32, 5, activation='relu', padding='same', name='conv1b')(conv1)
        if self.use_batch_norm:
            conv1 = BatchNormalization(name='bn1b')(conv1)
        conv1 = Dropout(self.dropout_rate, name='dropout1')(conv1)
        pool1 = MaxPooling1D(pool_size=2, name='pool1')(conv1)
        
        # Second convolution block - medium-scale patterns
        conv2 = Conv1D(64, 5, activation='relu', padding='same', name='conv2a')(pool1)
        if self.use_batch_norm:
            conv2 = BatchNormalization(name='bn2a')(conv2)
        conv2 = Conv1D(64, 5, activation='relu', padding='same', name='conv2b')(conv2)
        if self.use_batch_norm:
            conv2 = BatchNormalization(name='bn2b')(conv2)
        conv2 = Dropout(self.dropout_rate, name='dropout2')(conv2)
        pool2 = MaxPooling1D(pool_size=2, name='pool2')(conv2)
        
        # Third convolution block - large-scale patterns
        conv3 = Conv1D(128, 3, activation='relu', padding='same', name='conv3a')(pool2)
        if self.use_batch_norm:
            conv3 = BatchNormalization(name='bn3a')(conv3)
        conv3 = Conv1D(128, 3, activation='relu', padding='same', name='conv3b')(conv3)
        if self.use_batch_norm:
            conv3 = BatchNormalization(name='bn3b')(conv3)
        conv3 = Dropout(self.dropout_rate, name='dropout3')(conv3)
        pool3 = MaxPooling1D(pool_size=2, name='pool3')(conv3)
        
        # Fourth convolution block - very large-scale patterns
        conv4 = Conv1D(256, 3, activation='relu', padding='same', name='conv4a')(pool3)
        if self.use_batch_norm:
            conv4 = BatchNormalization(name='bn4a')(conv4)
        conv4 = Conv1D(256, 3, activation='relu', padding='same', name='conv4b')(conv4)
        if self.use_batch_norm:
            conv4 = BatchNormalization(name='bn4b')(conv4)
        conv4 = Dropout(self.dropout_rate, name='dropout4')(conv4)
        pool4 = MaxPooling1D(pool_size=2, name='pool4')(conv4)
        
        # =================================================================
        # BOTTLENECK - Deepest feature analysis
        # =================================================================
        bottleneck = Conv1D(512, 3, activation='relu', padding='same', name='bottleneck_a')(pool4)
        if self.use_batch_norm:
            bottleneck = BatchNormalization(name='bn_bottleneck_a')(bottleneck)
        bottleneck = Conv1D(512, 3, activation='relu', padding='same', name='bottleneck_b')(bottleneck)
        if self.use_batch_norm:
            bottleneck = BatchNormalization(name='bn_bottleneck_b')(bottleneck)
        bottleneck = Dropout(self.dropout_rate * 2, name='dropout_bottleneck')(bottleneck)
        
        # =================================================================
        # DECODER (Upsampling) - Progressive reconstruction
        # =================================================================
        
        # First upsampling block
        up5 = UpSampling1D(size=2, name='up5')(bottleneck)
        merge5 = Concatenate(name='merge5')([up5, conv4])
        conv5 = Conv1D(256, 3, activation='relu', padding='same', name='conv5a')(merge5)
        if self.use_batch_norm:
            conv5 = BatchNormalization(name='bn5a')(conv5)
        conv5 = Conv1D(256, 3, activation='relu', padding='same', name='conv5b')(conv5)
        if self.use_batch_norm:
            conv5 = BatchNormalization(name='bn5b')(conv5)
        conv5 = Dropout(self.dropout_rate, name='dropout5')(conv5)
        
        # Second upsampling block
        up6 = UpSampling1D(size=2, name='up6')(conv5)
        merge6 = Concatenate(name='merge6')([up6, conv3])
        conv6 = Conv1D(128, 3, activation='relu', padding='same', name='conv6a')(merge6)
        if self.use_batch_norm:
            conv6 = BatchNormalization(name='bn6a')(conv6)
        conv6 = Conv1D(128, 3, activation='relu', padding='same', name='conv6b')(conv6)
        if self.use_batch_norm:
            conv6 = BatchNormalization(name='bn6b')(conv6)
        conv6 = Dropout(self.dropout_rate, name='dropout6')(conv6)
        
        # Third upsampling block
        up7 = UpSampling1D(size=2, name='up7')(conv6)
        merge7 = Concatenate(name='merge7')([up7, conv2])
        conv7 = Conv1D(64, 5, activation='relu', padding='same', name='conv7a')(merge7)
        if self.use_batch_norm:
            conv7 = BatchNormalization(name='bn7a')(conv7)
        conv7 = Conv1D(64, 5, activation='relu', padding='same', name='conv7b')(conv7)
        if self.use_batch_norm:
            conv7 = BatchNormalization(name='bn7b')(conv7)
        conv7 = Dropout(self.dropout_rate, name='dropout7')(conv7)
        
        # Fourth upsampling block
        up8 = UpSampling1D(size=2, name='up8')(conv7)
        merge8 = Concatenate(name='merge8')([up8, conv1])
        conv8 = Conv1D(32, 5, activation='relu', padding='same', name='conv8a')(merge8)
        if self.use_batch_norm:
            conv8 = BatchNormalization(name='bn8a')(conv8)
        conv8 = Conv1D(32, 5, activation='relu', padding='same', name='conv8b')(conv8)
        if self.use_batch_norm:
            conv8 = BatchNormalization(name='bn8b')(conv8)
        
        # Remove the padding that was added initially
        conv8_cropped = Lambda(
            lambda x: x[:, self.padding_size:-self.padding_size, :], 
            name='suppression_padding'
        )(conv8)
        
        # =================================================================
        # OUTPUT - Residual learning approach
        # =================================================================
        # Predict the noise component to be subtracted (residual learning)
        noise_estimate = Conv1D(1, 1, activation='linear', name='bruit_estime')(conv8_cropped)
        
        # Create the model
        model = Model(inputs=inputs, outputs=noise_estimate, name='Unet1D_TipTiming')
        
        # Compile with custom loss function
        custom_loss = self.create_loss_function()
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=custom_loss,
            metrics=['mae', 'mse']
        )
        
        self.model = model
        return model
    
    def prepare_training_callbacks(self, 
                                 save_path: str = 'unet_tip_timing_best.h5') -> list:
        """
        Prepare training callbacks optimized for vibration denoising
        
        Args:
            save_path: Path to save the best model
            
        Returns:
            List of Keras callbacks
        """
        callbacks = [
            EarlyStopping(
                patience=12,
                restore_best_weights=True,
                monitor='val_loss',
                verbose=1,
                min_delta=1e-7
            ),
            ReduceLROnPlateau(
                patience=6,
                factor=0.3,
                min_lr=1e-8,
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
              epochs: int = 80,
              batch_size: int = 32,
              verbose: int = 1) -> Dict[str, Any]:
        """
        Train the Unet1D model for vibration denoising
        
        Args:
            X_train: Training corrected deflections (n_samples, n_tours, 1)
            y_train: Training clean vibrations (n_samples, n_tours, 1)
            X_val: Validation corrected deflections
            y_val: Validation clean vibrations
            epochs: Maximum number of training epochs
            batch_size: Training batch size
            verbose: Verbosity level
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.build_model()
            
        # Ensure proper input shape
        if len(X_train.shape) == 2:
            X_train = X_train[..., np.newaxis]
        if len(y_train.shape) == 2:
            y_train = y_train[..., np.newaxis]
            
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            if len(X_val.shape) == 2:
                X_val = X_val[..., np.newaxis]
            if len(y_val.shape) == 2:
                y_val = y_val[..., np.newaxis]
            validation_data = (X_val, y_val)
            
        # For residual learning, we predict the noise (difference between input and clean signal)
        noise_train = X_train - y_train
        noise_val = None
        if validation_data is not None:
            noise_val = X_val - y_val
            validation_data = (X_train, noise_train) if validation_data is None else (X_val, noise_val)
        
        # Prepare callbacks
        callbacks = self.prepare_training_callbacks()
        
        print("=== Starting Unet1D Training for Vibration Denoising ===")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Sequence length: {X_train.shape[1]} rotations")
        if validation_data:
            print(f"Validation samples: {X_val.shape[0]}")
        print("Objective: Extract clean blade vibrations from corrected deflections")
        print("Approach: Residual learning (predict noise to subtract)")
        
        # Train the model
        self.history = self.model.fit(
            X_train, noise_train,
            validation_data=(X_val, noise_val) if noise_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        print("=== Unet1D Training Completed ===")
        return self.history.history
    
    def denoise_vibrations(self, corrected_deflections: np.ndarray) -> np.ndarray:
        """
        Denoise corrected deflections to extract clean vibration signals
        
        Args:
            corrected_deflections: Baseline-corrected deflection measurements
            
        Returns:
            Clean vibration signals
        """
        if self.model is None:
            raise ValueError("Model must be built and trained before prediction")
            
        # Ensure proper input shape
        if len(corrected_deflections.shape) == 2:
            corrected_deflections = corrected_deflections[..., np.newaxis]
            
        # Predict noise component
        predicted_noise = self.model.predict(corrected_deflections, verbose=0)
        
        # Subtract predicted noise to get clean vibrations
        clean_vibrations = corrected_deflections - predicted_noise
        
        # Remove channel dimension if it was added
        if clean_vibrations.shape[-1] == 1:
            clean_vibrations = clean_vibrations.squeeze(-1)
            
        return clean_vibrations
    
    def evaluate_denoising_performance(self,
                                     X_test: np.ndarray,
                                     y_true: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the quality of vibration denoising
        
        Args:
            X_test: Test corrected deflections
            y_true: True clean vibrations
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
            
        # Get denoised vibrations
        y_pred = self.denoise_vibrations(X_test)
        
        # Calculate comprehensive metrics
        metrics = {}
        
        # Basic error metrics
        metrics['rmse'] = float(np.sqrt(np.mean((y_true - y_pred)**2)))
        metrics['mae'] = float(np.mean(np.abs(y_true - y_pred)))
        
        # Signal-to-noise ratio
        signal_power = np.mean(y_true**2)
        noise_power = np.mean((y_true - y_pred)**2)
        metrics['snr_db'] = float(10 * np.log10(signal_power / noise_power)) if noise_power > 0 else np.inf
        
        # Cosine similarity (shape preservation)
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        cos_sim = np.dot(y_true_flat, y_pred_flat) / (
            np.linalg.norm(y_true_flat) * np.linalg.norm(y_pred_flat)
        )
        metrics['cosine_similarity'] = float(cos_sim)
        
        # Amplitude preservation (critical for resonance detection)
        amplitude_true = np.max(np.abs(y_true))
        amplitude_pred = np.max(np.abs(y_pred))
        metrics['amplitude_error_percent'] = float(
            np.abs(amplitude_true - amplitude_pred) / amplitude_true * 100
        ) if amplitude_true > 0 else 0
        
        # Frequency domain analysis
        try:
            from scipy.fft import fft
            
            # FFT for all samples and average
            fft_true = np.mean([np.abs(fft(signal)) for signal in y_true], axis=0)
            fft_pred = np.mean([np.abs(fft(signal)) for signal in y_pred], axis=0)
            
            # Dominant frequency preservation
            freq_true = np.argmax(fft_true[:len(fft_true)//2])
            freq_pred = np.argmax(fft_pred[:len(fft_pred)//2])
            metrics['freq_error_bins'] = float(abs(freq_true - freq_pred))
            
            # Spectral correlation
            spectral_corr = np.corrcoef(
                fft_true[:len(fft_true)//2], 
                fft_pred[:len(fft_pred)//2]
            )[0, 1]
            metrics['spectral_correlation'] = float(spectral_corr)
            
        except ImportError:
            # scipy not available, skip frequency analysis
            pass
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(filepath)
        print(f"Unet1D model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a pre-trained model"""
        # Note: Custom loss function needs to be provided when loading
        custom_loss = self.create_loss_function()
        self.model = tf.keras.models.load_model(
            filepath, 
            custom_objects={'loss_preservation_resonances': custom_loss}
        )
        print(f"Unet1D model loaded from {filepath}")
    
    def get_model_summary(self):
        """Print model architecture summary"""
        if self.model is None:
            self.build_model()
        return self.model.summary()


def create_unet1d_tip_timing(n_tours: int = 1000,
                            learning_rate: float = 1e-6,
                            **kwargs) -> Unet1DTipTiming:
    """
    Factory function to create Unet1D model for tip timing
    
    Args:
        n_tours: Number of rotations in sequence
        learning_rate: Learning rate for optimization
        **kwargs: Additional model parameters
        
    Returns:
        Configured Unet1DTipTiming instance
    """
    return Unet1DTipTiming(n_tours=n_tours, learning_rate=learning_rate, **kwargs)