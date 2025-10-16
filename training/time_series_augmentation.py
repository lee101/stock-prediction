#!/usr/bin/env python3
"""
Comprehensive Time Series Data Augmentation for Financial Data
Advanced augmentation techniques specifically designed for trading systems
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from scipy import signal
from scipy.interpolate import interp1d, CubicSpline
from sklearn.preprocessing import StandardScaler
import torch
import warnings
warnings.filterwarnings('ignore')


class FinancialTimeSeriesAugmenter:
    """
    Comprehensive augmentation system for financial time series data
    Implements multiple modern augmentation techniques suitable for trading data
    """
    
    def __init__(
        self, 
        preserve_price_relationships=True,
        preserve_volume_patterns=True,
        augmentation_strength=0.5
    ):
        self.preserve_price_relationships = preserve_price_relationships
        self.preserve_volume_patterns = preserve_volume_patterns  
        self.augmentation_strength = augmentation_strength
        
        # Cache for trend patterns
        self._trend_cache = {}
        
    def augment_batch(
        self, 
        data: np.ndarray, 
        labels: Optional[np.ndarray] = None,
        augmentation_types: List[str] = None,
        num_augmentations: int = 1
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply multiple augmentations to a batch of time series data
        
        Args:
            data: Input data of shape (batch_size, seq_len, features)
            labels: Optional labels (batch_size,)
            augmentation_types: List of augmentation types to apply
            num_augmentations: Number of augmented versions per sample
            
        Returns:
            Augmented data and labels
        """
        if augmentation_types is None:
            augmentation_types = [
                'gaussian_noise', 'time_warp', 'magnitude_warp',
                'window_slice', 'channel_shuffle', 'mixup',
                'cutmix', 'frequency_mask', 'trend_injection'
            ]
        
        augmented_data = []
        augmented_labels = []
        
        for sample_idx in range(data.shape[0]):
            sample = data[sample_idx]
            sample_label = labels[sample_idx] if labels is not None else None
            
            # Original sample
            augmented_data.append(sample)
            if labels is not None:
                augmented_labels.append(sample_label)
            
            # Generate augmentations
            for _ in range(num_augmentations):
                # Randomly select augmentation techniques
                selected_augs = np.random.choice(
                    augmentation_types, 
                    size=np.random.randint(1, 4),  # Apply 1-3 augmentations
                    replace=False
                )
                
                aug_sample = sample.copy()
                
                for aug_type in selected_augs:
                    aug_sample = self._apply_augmentation(aug_sample, aug_type)
                
                augmented_data.append(aug_sample)
                if labels is not None:
                    augmented_labels.append(sample_label)
        
        augmented_data = np.array(augmented_data)
        augmented_labels = np.array(augmented_labels) if labels is not None else None
        
        return augmented_data, augmented_labels
    
    def _apply_augmentation(self, data: np.ndarray, aug_type: str) -> np.ndarray:
        """Apply specific augmentation type"""
        
        if aug_type == 'gaussian_noise':
            return self.add_gaussian_noise(data)
        elif aug_type == 'time_warp':
            return self.time_warp(data)
        elif aug_type == 'magnitude_warp':
            return self.magnitude_warp(data)
        elif aug_type == 'window_slice':
            return self.window_slice(data)
        elif aug_type == 'channel_shuffle':
            return self.channel_shuffle(data)
        elif aug_type == 'frequency_mask':
            return self.frequency_mask(data)
        elif aug_type == 'trend_injection':
            return self.trend_injection(data)
        elif aug_type == 'volatility_scaling':
            return self.volatility_scaling(data)
        elif aug_type == 'regime_shift':
            return self.regime_shift(data)
        else:
            return data
    
    def add_gaussian_noise(
        self, 
        data: np.ndarray, 
        noise_factor: Optional[float] = None
    ) -> np.ndarray:
        """
        Add Gaussian noise scaled by feature volatility
        Preserves price relationships if enabled
        """
        if noise_factor is None:
            noise_factor = 0.01 * self.augmentation_strength
        
        augmented = data.copy()
        
        for feature_idx in range(data.shape[1]):
            feature_data = data[:, feature_idx]
            
            # Scale noise by feature standard deviation
            feature_std = np.std(feature_data)
            if feature_std > 0:
                noise = np.random.normal(0, feature_std * noise_factor, len(feature_data))
                
                # For price features, ensure relationships are preserved
                if self.preserve_price_relationships and feature_idx < 4:  # OHLC
                    # Add proportional noise instead of absolute
                    augmented[:, feature_idx] = feature_data * (1 + noise)
                else:
                    augmented[:, feature_idx] = feature_data + noise
        
        return augmented
    
    def time_warp(
        self, 
        data: np.ndarray, 
        sigma: Optional[float] = None,
        knot_count: int = 4
    ) -> np.ndarray:
        """
        Apply smooth time warping using cubic splines
        More sophisticated than simple interpolation
        """
        if sigma is None:
            sigma = 0.2 * self.augmentation_strength
        
        seq_len = len(data)
        
        # Create random warping points
        orig_steps = np.linspace(0, seq_len - 1, knot_count)
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=knot_count)
        
        # Ensure monotonicity (time should still flow forward)
        random_warps = np.cumsum(random_warps)
        random_warps = random_warps / random_warps[-1] * (seq_len - 1)
        
        # Apply warping to each feature
        warped_data = np.zeros_like(data)
        
        for feature_idx in range(data.shape[1]):
            try:
                # Create cubic spline interpolator
                cs = CubicSpline(orig_steps, data[orig_steps.astype(int), feature_idx])
                
                # Sample at warped points
                new_steps = np.linspace(0, seq_len - 1, seq_len)
                warped_values = cs(random_warps)
                
                # Interpolate back to original length
                final_interp = interp1d(
                    random_warps, warped_values, 
                    kind='linear', fill_value='extrapolate'
                )
                warped_data[:, feature_idx] = final_interp(new_steps)
                
            except Exception:
                # Fallback to original data if interpolation fails
                warped_data[:, feature_idx] = data[:, feature_idx]
        
        return warped_data
    
    def magnitude_warp(
        self, 
        data: np.ndarray, 
        sigma: Optional[float] = None,
        knot_count: int = 4
    ) -> np.ndarray:
        """
        Apply random magnitude scaling along the time axis
        """
        if sigma is None:
            sigma = 0.2 * self.augmentation_strength
        
        seq_len = len(data)
        
        # Create warping curve
        warp_steps = np.linspace(0, seq_len - 1, knot_count)
        warp_values = np.random.normal(loc=1.0, scale=sigma, size=knot_count)
        
        # Interpolate to full sequence
        cs = CubicSpline(warp_steps, warp_values)
        full_warp = cs(np.arange(seq_len))
        
        # Apply magnitude warping
        warped_data = data.copy()
        
        for feature_idx in range(data.shape[1]):
            if self.preserve_price_relationships and feature_idx < 4:  # OHLC prices
                # Scale prices together to maintain relationships
                warped_data[:, feature_idx] = data[:, feature_idx] * full_warp
            elif not self.preserve_volume_patterns or feature_idx != 4:  # Not volume
                warped_data[:, feature_idx] = data[:, feature_idx] * full_warp
        
        return warped_data
    
    def window_slice(
        self, 
        data: np.ndarray, 
        slice_ratio: Optional[float] = None
    ) -> np.ndarray:
        """
        Randomly slice a window from the data and pad/repeat to maintain length
        """
        if slice_ratio is None:
            slice_ratio = 0.7 + 0.2 * self.augmentation_strength
        
        seq_len = len(data)
        slice_len = int(seq_len * slice_ratio)
        
        if slice_len >= seq_len:
            return data
        
        # Random start position
        start_pos = np.random.randint(0, seq_len - slice_len + 1)
        sliced_data = data[start_pos:start_pos + slice_len]
        
        # Pad by repeating edge values
        pad_before = start_pos
        pad_after = seq_len - start_pos - slice_len
        
        if pad_before > 0:
            before_pad = np.repeat(sliced_data[0:1], pad_before, axis=0)
            sliced_data = np.concatenate([before_pad, sliced_data], axis=0)
        
        if pad_after > 0:
            after_pad = np.repeat(sliced_data[-1:], pad_after, axis=0)
            sliced_data = np.concatenate([sliced_data, after_pad], axis=0)
        
        return sliced_data
    
    def channel_shuffle(self, data: np.ndarray) -> np.ndarray:
        """
        Shuffle non-price features to reduce overfitting to feature order
        Preserves price relationships (OHLC)
        """
        augmented = data.copy()
        
        if data.shape[1] > 5:  # If we have more than OHLC + Volume
            # Shuffle technical indicators but keep OHLC + Volume in place
            tech_features = augmented[:, 5:]  # Features beyond OHLC + Volume
            
            # Randomly permute technical features
            perm_indices = np.random.permutation(tech_features.shape[1])
            augmented[:, 5:] = tech_features[:, perm_indices]
        
        return augmented
    
    def frequency_mask(
        self, 
        data: np.ndarray, 
        mask_ratio: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply frequency domain masking to reduce high-frequency noise
        """
        if mask_ratio is None:
            mask_ratio = 0.1 * self.augmentation_strength
        
        augmented = data.copy()
        
        for feature_idx in range(data.shape[1]):
            feature_data = data[:, feature_idx]
            
            # Apply FFT
            fft_data = np.fft.fft(feature_data)
            freqs = np.fft.fftfreq(len(feature_data))
            
            # Mask high frequencies
            high_freq_cutoff = np.percentile(np.abs(freqs), (1 - mask_ratio) * 100)
            mask = np.abs(freqs) < high_freq_cutoff
            
            masked_fft = fft_data * mask
            
            # Inverse FFT
            filtered_data = np.real(np.fft.ifft(masked_fft))
            augmented[:, feature_idx] = filtered_data
        
        return augmented
    
    def trend_injection(
        self, 
        data: np.ndarray, 
        trend_strength: Optional[float] = None
    ) -> np.ndarray:
        """
        Inject synthetic trends to improve generalization
        """
        if trend_strength is None:
            trend_strength = 0.05 * self.augmentation_strength
        
        seq_len = len(data)
        augmented = data.copy()
        
        # Generate trend types
        trend_types = ['linear', 'exponential', 'sinusoidal', 'step']
        trend_type = np.random.choice(trend_types)
        
        if trend_type == 'linear':
            trend = np.linspace(0, trend_strength, seq_len)
        elif trend_type == 'exponential':
            trend = np.exp(np.linspace(0, trend_strength, seq_len)) - 1
        elif trend_type == 'sinusoidal':
            trend = trend_strength * np.sin(np.linspace(0, 4 * np.pi, seq_len))
        else:  # step
            step_point = seq_len // 2
            trend = np.concatenate([
                np.zeros(step_point),
                np.full(seq_len - step_point, trend_strength)
            ])
        
        # Apply trend to price features
        if self.preserve_price_relationships:
            # Apply same trend to all price features
            for price_idx in range(min(4, data.shape[1])):  # OHLC
                augmented[:, price_idx] = data[:, price_idx] * (1 + trend)
        else:
            # Apply random trends to different features
            for feature_idx in range(data.shape[1]):
                if np.random.random() < 0.3:  # 30% chance per feature
                    augmented[:, feature_idx] = data[:, feature_idx] * (1 + trend)
        
        return augmented
    
    def volatility_scaling(
        self, 
        data: np.ndarray, 
        scale_factor: Optional[float] = None
    ) -> np.ndarray:
        """
        Scale the volatility of the time series
        """
        if scale_factor is None:
            scale_factor = np.random.uniform(0.5, 2.0) * self.augmentation_strength + (1 - self.augmentation_strength)
        
        augmented = data.copy()
        
        for feature_idx in range(data.shape[1]):
            feature_data = data[:, feature_idx]
            feature_mean = np.mean(feature_data)
            
            # Scale deviations from mean
            scaled_data = feature_mean + (feature_data - feature_mean) * scale_factor
            augmented[:, feature_idx] = scaled_data
        
        return augmented
    
    def regime_shift(
        self, 
        data: np.ndarray, 
        shift_point: Optional[int] = None,
        shift_magnitude: Optional[float] = None
    ) -> np.ndarray:
        """
        Simulate market regime changes
        """
        if shift_point is None:
            shift_point = np.random.randint(len(data) // 4, 3 * len(data) // 4)
        
        if shift_magnitude is None:
            shift_magnitude = 0.1 * self.augmentation_strength
        
        augmented = data.copy()
        
        # Apply regime shift to price-based features
        regime_multiplier = 1 + shift_magnitude * np.random.choice([-1, 1])
        
        for feature_idx in range(min(4, data.shape[1])):  # OHLC
            augmented[shift_point:, feature_idx] *= regime_multiplier
        
        return augmented
    
    @staticmethod
    def mixup(
        data1: np.ndarray, 
        data2: np.ndarray, 
        alpha: float = 0.4
    ) -> Tuple[np.ndarray, float]:
        """
        Mixup augmentation between two samples
        """
        lam = np.random.beta(alpha, alpha)
        mixed_data = lam * data1 + (1 - lam) * data2
        return mixed_data, lam
    
    @staticmethod
    def cutmix(
        data1: np.ndarray, 
        data2: np.ndarray, 
        alpha: float = 1.0
    ) -> Tuple[np.ndarray, float]:
        """
        CutMix augmentation - replace random segments
        """
        lam = np.random.beta(alpha, alpha)
        seq_len = len(data1)
        
        cut_len = int(seq_len * (1 - lam))
        cut_start = np.random.randint(0, seq_len - cut_len)
        
        mixed_data = data1.copy()
        mixed_data[cut_start:cut_start + cut_len] = data2[cut_start:cut_start + cut_len]
        
        return mixed_data, lam


class AdaptiveAugmentationScheduler:
    """
    Adaptive scheduler for augmentation strength based on training progress
    Reduces augmentation as model improves to prevent over-regularization
    """
    
    def __init__(
        self, 
        initial_strength: float = 1.0,
        final_strength: float = 0.3,
        adaptation_steps: int = 1000
    ):
        self.initial_strength = initial_strength
        self.final_strength = final_strength
        self.adaptation_steps = adaptation_steps
        self.current_step = 0
        
    def get_current_strength(self) -> float:
        """Get current augmentation strength"""
        if self.current_step >= self.adaptation_steps:
            return self.final_strength
        
        # Linear decay from initial to final strength
        progress = self.current_step / self.adaptation_steps
        return self.initial_strength + (self.final_strength - self.initial_strength) * progress
    
    def step(self):
        """Update the scheduler"""
        self.current_step += 1
    
    def reset(self):
        """Reset the scheduler"""
        self.current_step = 0


def create_augmented_dataset(
    original_data: np.ndarray,
    augmentation_factor: int = 2,
    augmentation_types: List[str] = None,
    preserve_relationships: bool = True
) -> np.ndarray:
    """
    Create an augmented dataset with specified factor
    
    Args:
        original_data: Original dataset (samples, seq_len, features)
        augmentation_factor: How many augmented versions per sample
        augmentation_types: Which augmentations to use
        preserve_relationships: Whether to preserve financial relationships
        
    Returns:
        Augmented dataset
    """
    
    augmenter = FinancialTimeSeriesAugmenter(
        preserve_price_relationships=preserve_relationships,
        preserve_volume_patterns=preserve_relationships
    )
    
    augmented_data, _ = augmenter.augment_batch(
        original_data,
        augmentation_types=augmentation_types,
        num_augmentations=augmentation_factor
    )
    
    return augmented_data


if __name__ == '__main__':
    print("\n" + "="*80)
    print("🔄 COMPREHENSIVE TIME SERIES AUGMENTATION SYSTEM")
    print("="*80)
    
    # Test the augmentation system
    print("\n🧪 Testing augmentation system...")
    
    # Create sample financial data (batch_size=2, seq_len=100, features=10)
    np.random.seed(42)
    sample_data = np.random.randn(2, 100, 10)
    
    # Make it look more like financial data
    sample_data[:, :, 0] = 100 + np.cumsum(np.random.randn(2, 100) * 0.01, axis=1)  # Price
    sample_data[:, :, 4] = np.abs(np.random.randn(2, 100)) * 1000  # Volume
    
    # Create augmenter
    augmenter = FinancialTimeSeriesAugmenter(
        preserve_price_relationships=True,
        augmentation_strength=0.5
    )
    
    # Test different augmentations
    aug_types = [
        'gaussian_noise', 'time_warp', 'magnitude_warp',
        'window_slice', 'frequency_mask', 'trend_injection'
    ]
    
    print(f"📊 Original data shape: {sample_data.shape}")
    
    for aug_type in aug_types:
        try:
            augmented = augmenter._apply_augmentation(sample_data[0], aug_type)
            print(f"✅ {aug_type}: {augmented.shape}")
        except Exception as e:
            print(f"❌ {aug_type}: Failed - {str(e)}")
    
    # Test batch augmentation
    augmented_batch, _ = augmenter.augment_batch(
        sample_data,
        num_augmentations=3
    )
    
    print(f"\n📈 Batch augmentation:")
    print(f"   Original: {sample_data.shape}")
    print(f"   Augmented: {augmented_batch.shape}")
    print(f"   Augmentation factor: {augmented_batch.shape[0] / sample_data.shape[0]:.1f}x")
    
    # Test adaptive scheduler
    scheduler = AdaptiveAugmentationScheduler()
    print(f"\n⚡ Adaptive scheduling:")
    for step in [0, 250, 500, 750, 1000, 1500]:
        scheduler.current_step = step
        strength = scheduler.get_current_strength()
        print(f"   Step {step:4d}: Strength = {strength:.3f}")
    
    print("\n" + "="*80)
    print("AUGMENTATION TECHNIQUES IMPLEMENTED:")
    print("="*80)
    print("✅ Gaussian Noise (volatility-scaled)")
    print("✅ Time Warping (cubic spline)")
    print("✅ Magnitude Warping")
    print("✅ Window Slicing")
    print("✅ Channel Shuffling")
    print("✅ Frequency Masking")
    print("✅ Trend Injection")
    print("✅ Volatility Scaling")
    print("✅ Regime Shifts")
    print("✅ Mixup & CutMix")
    print("✅ Adaptive Scheduling")
    print("="*80)