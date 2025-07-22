#!/usr/bin/env python3
"""
Pre-process Module
==================

Handles preprocessing operations like downsampling.
"""

import logging
import numpy as np
import mne
from typing import Tuple, Optional

def downsample_data(raw: mne.io.Raw, data: np.ndarray, target_sfreq: float, 
                   subject_id: str) -> Tuple[Optional[mne.io.Raw], Optional[np.ndarray]]:
    """
    Downsample EEG data to target sampling frequency.
    
    Parameters
    ----------
    raw : mne.io.Raw
        MNE raw object
    data : np.ndarray
        EEG data array (in microvolts)
    target_sfreq : float
        Target sampling frequency in Hz
    subject_id : str
        Subject identifier for logging
    
    Returns
    -------
    Tuple[Optional[mne.io.Raw], Optional[np.ndarray]]
        Tuple of (downsampled_raw, downsampled_data_in_microvolts) or (None, None) if fails
    """
    logger = logging.getLogger(__name__)
    
    try:
        current_sfreq = raw.info['sfreq']
        
        if target_sfreq >= current_sfreq:
            logger.warning(f"[{subject_id}] Target sampling rate ({target_sfreq} Hz) >= current rate ({current_sfreq} Hz), skipping downsampling")
            return raw, data
        
        logger.info(f"[{subject_id}] Downsampling from {current_sfreq} Hz to {target_sfreq} Hz")
        
        # Create a copy to avoid modifying original
        raw_downsampled = raw.copy()
        
        # Use MNE's resample method which includes anti-aliasing filter
        raw_downsampled.resample(target_sfreq)
        
        # Get the downsampled data in microvolts
        data_downsampled = raw_downsampled.get_data(units="uV")
        
        # Log the results
        original_duration = data.shape[1] / current_sfreq
        new_duration = data_downsampled.shape[1] / target_sfreq
        compression_ratio = data.shape[1] / data_downsampled.shape[1]
        
        logger.info(f"[{subject_id}] ✓ Downsampling completed:")
        logger.info(f"[{subject_id}]   Original: {data.shape[1]:,} samples at {current_sfreq} Hz ({original_duration:.1f}s)")
        logger.info(f"[{subject_id}]   Downsampled: {data_downsampled.shape[1]:,} samples at {target_sfreq} Hz ({new_duration:.1f}s)")
        logger.info(f"[{subject_id}]   Compression ratio: {compression_ratio:.2f}x")
        logger.debug(f"[{subject_id}]   Downsampled data range: {np.min(data_downsampled):.1f} to {np.max(data_downsampled):.1f} μV")
        logger.debug(f"[{subject_id}]   Downsampled data std: {np.std(data_downsampled):.1f} μV")
        
        # Check for any flat channels after downsampling
        channel_stds = np.std(data_downsampled, axis=1)
        flat_channels = np.sum(channel_stds < 0.1)  # Channels with very low std
        if flat_channels > 0:
            logger.warning(f"[{subject_id}] Found {flat_channels} channels with very low signal variance after downsampling")
        
        return raw_downsampled, data_downsampled
        
    except Exception as e:
        logger.error(f"[{subject_id}] Error during downsampling: {str(e)}")
        return None, None

def apply_preprocessing(raw: mne.io.Raw, data: np.ndarray, subject_id: str, 
                       downsample_freq: Optional[float] = None) -> Tuple[Optional[mne.io.Raw], Optional[np.ndarray]]:
    """
    Apply preprocessing steps to the EEG data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        MNE raw object
    data : np.ndarray
        EEG data array (in microvolts)
    subject_id : str
        Subject identifier for logging
    downsample_freq : Optional[float]
        Target downsampling frequency, if None no downsampling is applied
    
    Returns
    -------
    Tuple[Optional[mne.io.Raw], Optional[np.ndarray]]
        Tuple of (processed_raw, processed_data_in_microvolts) or (None, None) if fails
    """
    logger = logging.getLogger(__name__)
    
    try:
        processed_raw = raw
        processed_data = data
        
        logger.info(f"[{subject_id}] Starting preprocessing...")
        logger.debug(f"[{subject_id}] Input data shape: {data.shape}, dtype: {data.dtype}")
        logger.debug(f"[{subject_id}] Input data range: {np.min(data):.1f} to {np.max(data):.1f} μV")
        
        # Apply downsampling if requested
        if downsample_freq is not None:
            logger.info(f"[{subject_id}] Applying downsampling to {downsample_freq} Hz")
            processed_raw, processed_data = downsample_data(
                processed_raw, processed_data, downsample_freq, subject_id
            )
            
            if processed_raw is None or processed_data is None:
                logger.error(f"[{subject_id}] Preprocessing failed during downsampling")
                return None, None
        else:
            logger.info(f"[{subject_id}] No downsampling requested")
        
        # Add any additional preprocessing steps here in the future
        # For example: filtering, artifact removal, etc.
        # Important: Always use units="uV" when extracting data to maintain microvolts
        
        logger.info(f"[{subject_id}] ✓ Preprocessing completed successfully")
        logger.debug(f"[{subject_id}] Output data shape: {processed_data.shape}, dtype: {processed_data.dtype}")
        logger.debug(f"[{subject_id}] Output data range: {np.min(processed_data):.1f} to {np.max(processed_data):.1f} μV")
        
        return processed_raw, processed_data
        
    except Exception as e:
        logger.error(f"[{subject_id}] Error during preprocessing: {str(e)}")
        return None, None

def validate_preprocessing_parameters(downsample_freq: Optional[float], 
                                     current_sfreq: float, subject_id: str) -> bool:
    """
    Validate preprocessing parameters.
    
    Parameters
    ----------
    downsample_freq : Optional[float]
        Target downsampling frequency
    current_sfreq : float
        Current sampling frequency
    subject_id : str
        Subject identifier for logging
    
    Returns
    -------
    bool
        True if parameters are valid, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    if downsample_freq is None:
        return True
    
    if downsample_freq <= 0:
        logger.error(f"[{subject_id}] Invalid downsampling frequency: {downsample_freq} Hz (must be positive)")
        return False
    
    if downsample_freq > current_sfreq:
        logger.warning(f"[{subject_id}] Downsampling frequency ({downsample_freq} Hz) > current frequency ({current_sfreq} Hz)")
        return True  # Not an error, just won't downsample
    
    # Check if downsampling ratio is reasonable
    ratio = current_sfreq / downsample_freq
    if ratio > 100:
        logger.warning(f"[{subject_id}] Very high downsampling ratio: {ratio:.1f}x")
    
    return True 