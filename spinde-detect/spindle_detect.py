#!/usr/bin/env python3
"""
Spindle Detect Module
=====================

Handles spindle detection using YASA.
"""

import logging
import numpy as np
import pandas as pd
import yasa
from typing import Dict, Optional, Tuple

def detect_spindles(data: np.ndarray, sfreq: float, ch_names: list, 
                   subject_id: str, freq_range: Tuple[float, float] = (10.0, 16.0)) -> Optional[yasa.SpindlesResults]:
    """
    Detect spindles using YASA.
    
    Parameters
    ----------
    data : np.ndarray
        EEG data array (channels x samples)
    sfreq : float
        Sampling frequency
    ch_names : list
        Channel names
    subject_id : str
        Subject identifier for logging
    freq_range : Tuple[float, float]
        Frequency range for spindle detection
    
    Returns
    -------
    Optional[yasa.SpindlesResults]
        YASA spindle detection results, or None if failed
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"[{subject_id}] Detecting spindles...")
        logger.debug(f"[{subject_id}] Detection parameters: freq_range={freq_range}")
        logger.debug(f"[{subject_id}] Starting YASA spindle detection...")
        
        # Run YASA spindle detection
        sp = yasa.spindles_detect(
            data, 
            sf=sfreq, 
            ch_names=ch_names,
            freq_sp=freq_range,
            duration=(0.5, 2.0),  # Duration criteria
            min_distance=500,     # Minimum distance between spindles (ms)
            thresh={'rel_pow': 0.2, 'corr': 0.65, 'rms': 1.5}  # Detection thresholds
        )
        
        if sp is not None:
            logger.info(f"[{subject_id}] ✓ Spindle detection completed")
            logger.debug(f"[{subject_id}] Spindle detection results type: {type(sp)}")
        else:
            logger.warning(f"[{subject_id}] No spindles detected")
            
        return sp
        
    except Exception as e:
        logger.error(f"[{subject_id}] Error detecting spindles: {str(e)}")
        return None

def analyze_spindle_types(spindle_results: yasa.SpindlesResults, subject_id: str,
                         slow_range: Tuple[float, float] = (10.0, 12.0),
                         fast_range: Tuple[float, float] = (12.0, 16.0)) -> Optional[Dict]:
    """
    Analyze spindle types (slow vs fast).
    
    Parameters
    ----------
    spindle_results : yasa.SpindlesResults
        YASA spindle detection results
    subject_id : str
        Subject identifier for logging
    slow_range : Tuple[float, float]
        Frequency range for slow spindles
    fast_range : Tuple[float, float]
        Frequency range for fast spindles
    
    Returns
    -------
    Optional[Dict]
        Analysis results dictionary, or None if failed
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"[{subject_id}] Analyzing spindle types...")
        logger.debug(f"[{subject_id}] Analysis parameters: slow_range={slow_range}, fast_range={fast_range}")
        logger.debug(f"[{subject_id}] Starting spindle type analysis...")
        
        if spindle_results is None:
            logger.error(f"[{subject_id}] No spindle results to analyze")
            return None
        
        # Get the summary dataframe
        df_all = spindle_results.summary()
        
        if df_all is None or len(df_all) == 0:
            logger.warning(f"[{subject_id}] No spindles found in results")
            return _create_empty_analysis_results(slow_range, fast_range)
        
        # Classify spindles by frequency
        df_slow = df_all[(df_all['Frequency'] >= slow_range[0]) & 
                        (df_all['Frequency'] < slow_range[1])]
        df_fast = df_all[(df_all['Frequency'] >= fast_range[0]) & 
                        (df_all['Frequency'] < fast_range[1])]
        
        # Create channel summaries
        channel_summary_all = _create_channel_summary(df_all)
        channel_summary_slow = _create_channel_summary(df_slow)
        channel_summary_fast = _create_channel_summary(df_fast)
        
        # Calculate statistics
        stats = _calculate_spindle_statistics(df_all, df_slow, df_fast)
        
        # Create frequency ranges info
        frequency_ranges = {
            'total': list(fast_range),  # Use fast_range as total range for consistency
            'slow': list(slow_range),
            'fast': list(fast_range)
        }
        
        analysis_results = {
            'spindle_results': spindle_results,
            'df_all': df_all,
            'df_slow': df_slow,
            'df_fast': df_fast,
            'channel_summary_all': channel_summary_all,
            'channel_summary_slow': channel_summary_slow,
            'channel_summary_fast': channel_summary_fast,
            'statistics': stats,
            'frequency_ranges': frequency_ranges
        }
        
        logger.info(f"[{subject_id}] ✓ Analysis completed: {stats['total_spindles']} total spindles")
        logger.info(f"[{subject_id}] ✓ Slow spindles: {stats['slow_spindles']}, Fast spindles: {stats['fast_spindles']}")
        logger.debug(f"[{subject_id}] Analysis results keys: {list(analysis_results.keys())}")
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"[{subject_id}] Error analyzing spindle types: {str(e)}")
        return None

def _create_empty_analysis_results(slow_range: Tuple[float, float], 
                                  fast_range: Tuple[float, float]) -> Dict:
    """Create empty analysis results for cases with no spindles."""
    return {
        'spindle_results': None,
        'df_all': pd.DataFrame(),
        'df_slow': pd.DataFrame(),
        'df_fast': pd.DataFrame(),
        'channel_summary_all': pd.DataFrame(columns=['Count']),
        'channel_summary_slow': pd.DataFrame(columns=['Count']),
        'channel_summary_fast': pd.DataFrame(columns=['Count']),
        'statistics': {
            'total_spindles': 0,
            'slow_spindles': 0,
            'fast_spindles': 0,
            'slow_percentage': 0.0,
            'fast_percentage': 0.0
        },
        'frequency_ranges': {
            'total': list(fast_range),
            'slow': list(slow_range),
            'fast': list(fast_range)
        }
    }

def _create_channel_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create channel summary from spindle dataframe."""
    if len(df) == 0:
        return pd.DataFrame(columns=['Count'])
    
    # Count spindles per channel
    channel_counts = df['Channel'].value_counts().sort_index()
    channel_summary = pd.DataFrame({
        'Count': channel_counts
    })
    
    return channel_summary

def _calculate_spindle_statistics(df_all: pd.DataFrame, df_slow: pd.DataFrame, 
                                 df_fast: pd.DataFrame) -> Dict:
    """Calculate comprehensive spindle statistics."""
    stats = {
        'total_spindles': len(df_all),
        'slow_spindles': len(df_slow),
        'fast_spindles': len(df_fast),
    }
    
    # Calculate percentages
    if stats['total_spindles'] > 0:
        stats['slow_percentage'] = (stats['slow_spindles'] / stats['total_spindles']) * 100
        stats['fast_percentage'] = (stats['fast_spindles'] / stats['total_spindles']) * 100
    else:
        stats['slow_percentage'] = 0.0
        stats['fast_percentage'] = 0.0
    
    # Calculate frequency, duration, and amplitude statistics for all spindles
    if len(df_all) > 0:
        stats.update({
            'mean_frequency': float(df_all['Frequency'].mean()),
            'std_frequency': float(df_all['Frequency'].std()),
            'mean_duration': float(df_all['Duration'].mean()),
            'std_duration': float(df_all['Duration'].std()),
            'mean_amplitude': float(df_all['Amplitude'].mean()),
            'std_amplitude': float(df_all['Amplitude'].std())
        })
    
    # Calculate statistics for slow spindles
    if len(df_slow) > 0:
        stats.update({
            'slow_mean_frequency': float(df_slow['Frequency'].mean()),
            'slow_std_frequency': float(df_slow['Frequency'].std()),
            'slow_mean_duration': float(df_slow['Duration'].mean()),
            'slow_std_duration': float(df_slow['Duration'].std()),
            'slow_mean_amplitude': float(df_slow['Amplitude'].mean()),
            'slow_std_amplitude': float(df_slow['Amplitude'].std())
        })
    
    # Calculate statistics for fast spindles
    if len(df_fast) > 0:
        stats.update({
            'fast_mean_frequency': float(df_fast['Frequency'].mean()),
            'fast_std_frequency': float(df_fast['Frequency'].std()),
            'fast_mean_duration': float(df_fast['Duration'].mean()),
            'fast_std_duration': float(df_fast['Duration'].std()),
            'fast_mean_amplitude': float(df_fast['Amplitude'].mean()),
            'fast_std_amplitude': float(df_fast['Amplitude'].std())
        })
    
    return stats 