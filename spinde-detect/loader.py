#!/usr/bin/env python3
"""
Loader Module
=============

Handles loading of EEG .set files and basic data validation.
"""

import logging
import numpy as np
import mne
from pathlib import Path
from typing import Tuple, Optional

def load_eeg_file(eeg_file_path: Path, subject_id: str) -> Tuple[Optional[mne.io.Raw], Optional[np.ndarray]]:
    """
    Load EEG data from .set file using MNE.
    
    Parameters
    ----------
    eeg_file_path : Path
        Path to the .set file
    subject_id : str
        Subject identifier for logging
    
    Returns
    -------
    Tuple[Optional[mne.io.Raw], Optional[np.ndarray]]
        Tuple of (raw_object, data_array_in_microvolts) or (None, None) if loading fails
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"[{subject_id}] Loading EEG data from: {eeg_file_path}")
        
        # Load the .set file with preload=True to keep data available
        raw = mne.io.read_raw_eeglab(str(eeg_file_path), preload=True)
        
        # Get data in microvolts (YASA expects microvolts) - only set units during loading
        data = raw.get_data(units="uV")
        
        # Log basic information
        n_channels, n_samples = data.shape
        duration = n_samples / raw.info['sfreq']
        
        logger.info(f"[{subject_id}] ✓ Data loaded: {n_channels} channels, {n_samples:,} samples")
        logger.info(f"[{subject_id}] ✓ Duration: {duration:.1f} seconds, Sampling rate: {raw.info['sfreq']} Hz")
        logger.debug(f"[{subject_id}] Data shape: {data.shape}, Data type: {data.dtype}")
        logger.debug(f"[{subject_id}] Data converted to microvolts (μV)")
        
        return raw, data
        
    except Exception as e:
        logger.error(f"[{subject_id}] Error loading EEG data: {str(e)}")
        return None, None

def validate_eeg_data(raw: mne.io.Raw, data: np.ndarray, subject_id: str) -> bool:
    """
    Validate loaded EEG data for basic quality checks.
    
    Parameters
    ----------
    raw : mne.io.Raw
        MNE raw object
    data : np.ndarray
        EEG data array (in microvolts)
    subject_id : str
        Subject identifier for logging
    
    Returns
    -------
    bool
        True if data passes validation, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Check for basic data integrity
        if data is None or data.size == 0:
            logger.error(f"[{subject_id}] Data is empty or None")
            return False
        
        # Check for reasonable data ranges (assuming μV units)
        data_range = np.ptp(data)
        if data_range == 0:
            logger.error(f"[{subject_id}] Data has no variation (all values identical)")
            return False
        
        if data_range > 10000:  # More than 10,000 μV range seems unrealistic
            logger.warning(f"[{subject_id}] Very large data range: {data_range:.1f} μV")
        
        # Check for excessive number of NaN or infinite values
        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()
        
        total_samples = data.size
        nan_percentage = (nan_count / total_samples) * 100
        inf_percentage = (inf_count / total_samples) * 100
        
        if nan_percentage > 5:  # More than 5% NaN values
            logger.error(f"[{subject_id}] Too many NaN values: {nan_percentage:.2f}%")
            return False
        
        if inf_percentage > 0:
            logger.error(f"[{subject_id}] Infinite values detected: {inf_percentage:.2f}%")
            return False
        
        # Check sampling rate
        sfreq = raw.info['sfreq']
        if sfreq < 50 or sfreq > 10000:
            logger.warning(f"[{subject_id}] Unusual sampling rate: {sfreq} Hz")
        
        # Check number of channels
        n_channels = len(raw.ch_names)
        if n_channels < 10:
            logger.warning(f"[{subject_id}] Low number of channels: {n_channels}")
        elif n_channels > 500:
            logger.warning(f"[{subject_id}] Very high number of channels: {n_channels}")
        
        # Check data range (should be in microvolts, typically -500 to 500)
        data_min, data_max = np.min(data), np.max(data)
        if abs(data_max) > 5000 or abs(data_min) > 5000:
            logger.warning(f"[{subject_id}] Data values seem unusually large for μV: "
                          f"min={data_min:.1f}μV, max={data_max:.1f}μV")
        
        logger.debug(f"[{subject_id}] Data validation passed")
        logger.debug(f"[{subject_id}] NaN values: {nan_count} ({nan_percentage:.3f}%)")
        logger.debug(f"[{subject_id}] Data range: {data_range:.1f} μV (min={data_min:.1f}μV, max={data_max:.1f}μV)")
        
        return True
        
    except Exception as e:
        logger.error(f"[{subject_id}] Error during data validation: {str(e)}")
        return False

def get_channel_info(raw: mne.io.Raw, subject_id: str) -> dict:
    """
    Extract channel information from raw object.
    
    Parameters
    ----------
    raw : mne.io.Raw
        MNE raw object
    subject_id : str
        Subject identifier for logging
    
    Returns
    -------
    dict
        Dictionary containing channel information
    """
    logger = logging.getLogger(__name__)
    
    try:
        channel_info = {
            'ch_names': raw.ch_names,
            'n_channels': len(raw.ch_names),
            'sfreq': raw.info['sfreq'],
            'duration': raw.times[-1],
            'n_samples': len(raw.times)
        }
        
        # Try to get channel types if available
        try:
            channel_info['ch_types'] = raw.get_channel_types()
        except:
            logger.debug(f"[{subject_id}] Could not determine channel types")
            channel_info['ch_types'] = ['eeg'] * len(raw.ch_names)
        
        logger.debug(f"[{subject_id}] Channel info extracted successfully")
        return channel_info
        
    except Exception as e:
        logger.error(f"[{subject_id}] Error extracting channel info: {str(e)}")
        return {}

def load_and_validate_subject_data(eeg_file_path: Path, subject_id: str) -> Tuple[Optional[mne.io.Raw], Optional[np.ndarray]]:
    """
    Load and validate EEG data for a subject.
    
    Parameters
    ----------
    eeg_file_path : Path
        Path to the .set file
    subject_id : str
        Subject identifier
    
    Returns
    -------
    Tuple[Optional[mne.io.Raw], Optional[np.ndarray]]
        Tuple of (raw_object, data_array_in_microvolts) or (None, None) if loading/validation fails
    """
    logger = logging.getLogger(__name__)
    
    # Load the data
    raw, data = load_eeg_file(eeg_file_path, subject_id)
    
    if raw is None or data is None:
        logger.error(f"[{subject_id}] Failed to load EEG data")
        return None, None
    
    # Validate the data
    if not validate_eeg_data(raw, data, subject_id):
        logger.error(f"[{subject_id}] Data validation failed")
        return None, None
    
    # Get and log channel information
    channel_info = get_channel_info(raw, subject_id)
    if channel_info:
        logger.debug(f"[{subject_id}] Channel information: {channel_info['n_channels']} channels, "
                    f"{channel_info['sfreq']} Hz, {channel_info['duration']:.1f}s")
    
    logger.info(f"[{subject_id}] ✓ EEG data loaded and validated successfully (in μV)")
    return raw, data 