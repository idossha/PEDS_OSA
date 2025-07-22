#!/usr/bin/env python3
"""
DataFrame Create Module
=======================

Creates all dataframes from spindle analysis results.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

def create_spindle_dataframes(analysis_results: Dict, subject_id: str) -> Dict[str, pd.DataFrame]:
    """
    Create comprehensive dataframes from analysis results.
    
    Parameters
    ----------
    analysis_results : Dict
        Analysis results from spindle detection
    subject_id : str
        Subject identifier for logging
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary containing all dataframes
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"[{subject_id}] Creating dataframes...")
        
        dataframes = {}
        
        # Main spindle dataframes
        dataframes['all_spindles'] = analysis_results.get('df_all', pd.DataFrame())
        dataframes['slow_spindles'] = analysis_results.get('df_slow', pd.DataFrame())
        dataframes['fast_spindles'] = analysis_results.get('df_fast', pd.DataFrame())
        
        # Channel summary dataframes
        dataframes['channel_summary_all'] = analysis_results.get('channel_summary_all', pd.DataFrame())
        dataframes['channel_summary_slow'] = analysis_results.get('channel_summary_slow', pd.DataFrame())
        dataframes['channel_summary_fast'] = analysis_results.get('channel_summary_fast', pd.DataFrame())
        
        # Create summary statistics dataframe
        dataframes['summary_stats'] = create_summary_statistics_df(analysis_results, subject_id)
        
        logger.info(f"[{subject_id}] ✓ Created {len(dataframes)} dataframes")
        logger.debug(f"[{subject_id}] Dataframe keys: {list(dataframes.keys())}")
        
        return dataframes
        
    except Exception as e:
        logger.error(f"[{subject_id}] Error creating dataframes: {str(e)}")
        return {}

def create_summary_statistics_df(analysis_results: Dict, subject_id: str) -> pd.DataFrame:
    """
    Create a summary statistics dataframe.
    
    Parameters
    ----------
    analysis_results : Dict
        Analysis results
    subject_id : str
        Subject identifier
    
    Returns
    -------
    pd.DataFrame
        Summary statistics dataframe
    """
    stats = analysis_results.get('statistics', {})
    
    summary_data = {
        'Subject': [subject_id],
        'Total_Spindles': [stats.get('total_spindles', 0)],
        'Slow_Spindles': [stats.get('slow_spindles', 0)],
        'Fast_Spindles': [stats.get('fast_spindles', 0)],
        'Slow_Percentage': [stats.get('slow_percentage', 0.0)],
        'Fast_Percentage': [stats.get('fast_percentage', 0.0)],
        'Mean_Frequency_Hz': [stats.get('mean_frequency', np.nan)],
        'Std_Frequency_Hz': [stats.get('std_frequency', np.nan)],
        'Mean_Duration_s': [stats.get('mean_duration', np.nan)],
        'Std_Duration_s': [stats.get('std_duration', np.nan)],
        'Mean_Amplitude_uV': [stats.get('mean_amplitude', np.nan)],
        'Std_Amplitude_uV': [stats.get('std_amplitude', np.nan)],
        'Analysis_Timestamp': [datetime.now().isoformat()]
    }
    
    return pd.DataFrame(summary_data)

def save_dataframes(dataframes: Dict[str, pd.DataFrame], output_dir: Path, 
                   subject_id: str, overwrite: bool = False) -> Dict[str, Path]:
    """
    Save all dataframes to CSV files.
    
    Parameters
    ----------
    dataframes : Dict[str, pd.DataFrame]
        Dictionary of dataframes to save
    output_dir : Path
        Output directory
    subject_id : str
        Subject identifier for logging
    overwrite : bool
        Whether to overwrite existing files
    
    Returns
    -------
    Dict[str, Path]
        Dictionary mapping dataframe names to saved file paths
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"[{subject_id}] Saving dataframes...")
        
        saved_files = {}
        
        # Define file mappings
        file_mappings = {
            'all_spindles': 'all_spindles_detailed.csv',
            'slow_spindles': 'slow_spindles_detailed.csv',
            'fast_spindles': 'fast_spindles_detailed.csv',
            'channel_summary_all': 'channel_summary_all.csv',
            'channel_summary_slow': 'channel_summary_slow.csv',
            'channel_summary_fast': 'channel_summary_fast.csv',
            'summary_stats': 'spindle_summary.csv'
        }
        
        for df_name, filename in file_mappings.items():
            if df_name in dataframes and not dataframes[df_name].empty:
                file_path = output_dir / filename
                
                # Check if file exists and overwrite setting
                if file_path.exists() and not overwrite:
                    logger.warning(f"[{subject_id}] File exists, skipping: {filename}")
                    continue
                
                # Save dataframe
                dataframes[df_name].to_csv(file_path, index=True)
                saved_files[df_name] = file_path
                logger.debug(f"[{subject_id}] Saved {df_name}: {filename}")
            else:
                logger.debug(f"[{subject_id}] Skipping empty dataframe: {df_name}")
        
        logger.info(f"[{subject_id}] ✓ Saved {len(saved_files)} dataframe files")
        return saved_files
        
    except Exception as e:
        logger.error(f"[{subject_id}] Error saving dataframes: {str(e)}")
        return {}

def create_batch_summary_dataframe(processed_subjects: list, derivatives_dir: Path) -> Optional[pd.DataFrame]:
    """
    Create a batch summary dataframe from multiple subjects.
    
    Parameters
    ----------
    processed_subjects : list
        List of processed subject IDs
    derivatives_dir : Path
        Path to derivatives directory
    
    Returns
    -------
    Optional[pd.DataFrame]
        Batch summary dataframe, or None if failed
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Creating batch summary for {len(processed_subjects)} subjects...")
        
        batch_data = []
        
        for subject_id in processed_subjects:
            # Load subject summary
            summary_file = derivatives_dir / subject_id / 'spindles' / 'results' / 'spindle_summary.csv'
            
            if summary_file.exists():
                try:
                    subject_summary = pd.read_csv(summary_file, index_col=0)
                    batch_data.append(subject_summary)
                    logger.debug(f"Added {subject_id} to batch summary")
                except Exception as e:
                    logger.warning(f"Could not load summary for {subject_id}: {str(e)}")
            else:
                logger.warning(f"Summary file not found for {subject_id}")
        
        if not batch_data:
            logger.error("No valid subject summaries found for batch")
            return None
        
        # Combine all summaries
        batch_df = pd.concat(batch_data, ignore_index=True)
        
        logger.info(f"✓ Created batch summary with {len(batch_df)} subjects")
        return batch_df
        
    except Exception as e:
        logger.error(f"Error creating batch summary: {str(e)}")
        return None 