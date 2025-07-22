#!/usr/bin/env python3
"""
Pre-detection Module
====================

Extracts spectrogram, 2D channel layout images, and recording information 
before spindle detection.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import mne
import yasa
from pathlib import Path
from typing import Dict, Optional, Tuple

def create_spectrogram(raw: mne.io.Raw, subject_id: str, 
                      output_dir: Path) -> Optional[Path]:
    """
    Create spectrogram visualization following SW-detect pattern.
    
    Parameters
    ----------
    raw : mne.io.Raw
        MNE raw object
    subject_id : str
        Subject identifier for logging
    output_dir : Path
        Output directory for saving plots
    
    Returns
    -------
    Optional[Path]
        Path to saved spectrogram file, or None if failed
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"[{subject_id}] Creating spectrogram...")
        
        # Set matplotlib backend
        plt.switch_backend('Agg')
        
        # Use preferred channel indices like SW-detect, fallback to first channel
        preferred_channel_indices = [80, 89, 79, 44]
        channels = raw.info['ch_names']
        
        # Select the first available preferred channel
        selected_channel = None
        for idx in preferred_channel_indices:
            if idx < len(channels):
                selected_channel = channels[idx]
                break
        
        # If no preferred channel is found, use the first channel
        if selected_channel is None:
            selected_channel = channels[0]
        
        logger.info(f"[{subject_id}] Selected channel for spectrogram: {selected_channel}")
        
        # Extract data for selected channel using picks (SW-detect pattern)
        data = raw.get_data(picks=selected_channel).flatten()
        
        logger.debug(f"[{subject_id}] Spectrogram data shape: {data.shape}")
        logger.debug(f"[{subject_id}] Spectrogram data range: {np.min(data):.1f} to {np.max(data):.1f} μV")
        
        # Create spectrogram using YASA (let yasa create the figure like SW-detect)
        fig = yasa.plot_spectrogram(
            data,
            sf=raw.info['sfreq'],
            fmin=0.5, fmax=25,
            trimperc=2.5,
            cmap='RdBu_r',
            vmin=None, vmax=None,
            figsize=(15, 7)
        )
        
        # Access the spectrogram axes (assuming it's the first and only axes)
        ax = fig.axes[0]
        
        # Set title with channel name
        ax.set_title(f'Spectrogram - {selected_channel}', 
                    fontsize=14, fontweight='bold')
        
        # Add annotation overlays if available (following SW-detect pattern)
        annotations = raw.annotations
        found_stim = False
        if len(annotations):
            for annot in annotations:
                description = annot['description'].lower()
                if description in ["stim start", "stim end"]:
                    found_stim = True
                    # Convert onset from seconds to hours for the spectrogram's x-axis
                    onset_hr = annot['onset'] / 3600
                    ax.axvline(x=onset_hr, color='yellow', alpha=0.8, linestyle='--', linewidth=2)
                    ax.plot(onset_hr, ax.get_ylim()[1], marker='v', color='red', markersize=10)
                    logger.info(f"[{subject_id}] Added stim marker for annotation: '{annot['description']}' at {annot['onset']}s")

            if found_stim:
                ax.legend(['Stim events'], loc='upper right')
            else:
                logger.info(f"[{subject_id}] No 'stim start' or 'stim end' annotations found to overlay.")
        
        # Save spectrogram
        output_file = output_dir / 'spectrogram.png'
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"[{subject_id}] ✓ Spectrogram saved: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"[{subject_id}] Error creating spectrogram: {str(e)}")
        return None

def create_2d_sensor_layout(raw: mne.io.Raw, subject_id: str, 
                           output_dir: Path) -> Optional[Path]:
    """
    Create 2D sensor layout visualization.
    
    Parameters
    ----------
    raw : mne.io.Raw
        MNE raw object
    subject_id : str
        Subject identifier for logging
    output_dir : Path
        Output directory for saving plots
    
    Returns
    -------
    Optional[Path]
        Path to saved sensor layout file, or None if failed
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"[{subject_id}] Creating 2D sensor layout...")
        
        # Set matplotlib backend
        plt.switch_backend('Agg')
        
        try:
            # Try to use MNE's sensor plotting
            fig = raw.plot_sensors(kind='topomap', show_names=True, show=False)
            logger.debug(f"[{subject_id}] Used MNE sensor plotting")
            
        except Exception as e:
            logger.warning(f"[{subject_id}] MNE sensor plotting failed: {str(e)}")
            # Fallback: create a simple layout plot
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f'2D Sensor Layout\n{len(raw.info["ch_names"])} channels\n\n'
                              f'Unable to display layout:\n{str(e)[:100]}...', 
                   ha='center', va='center', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            ax.set_title('2D Sensor Layout', fontsize=16, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        # Save sensor layout
        output_file = output_dir / 'sensor_layout_2d.png'
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"[{subject_id}] ✓ 2D sensor layout saved: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"[{subject_id}] Error creating 2D sensor layout: {str(e)}")
        return None

def extract_recording_info(raw: mne.io.Raw, subject_id: str, 
                          output_dir: Path) -> Optional[Dict]:
    """
    Extract and save detailed recording information.
    
    Parameters
    ----------
    raw : mne.io.Raw
        MNE raw object
    subject_id : str
        Subject identifier for logging
    output_dir : Path
        Output directory for saving info
    
    Returns
    -------
    Optional[Dict]
        Recording information dictionary, or None if failed
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"[{subject_id}] Extracting recording information...")
        
        # Extract basic recording info
        recording_info = {
            'subject_id': subject_id,
            'n_channels': len(raw.ch_names),
            'channel_names': raw.ch_names,
            'sampling_frequency': raw.info['sfreq'],
            'duration_seconds': raw.times[-1],
            'duration_minutes': raw.times[-1] / 60,
            'duration_hours': raw.times[-1] / 3600,
            'n_samples': len(raw.times),
            'first_sample': raw.first_samp,
            'last_sample': raw.last_samp
        }
        
        # Try to get channel types
        try:
            recording_info['channel_types'] = raw.get_channel_types()
            channel_type_counts = {}
            for ch_type in recording_info['channel_types']:
                channel_type_counts[ch_type] = channel_type_counts.get(ch_type, 0) + 1
            recording_info['channel_type_counts'] = channel_type_counts
        except:
            logger.debug(f"[{subject_id}] Could not determine channel types")
            recording_info['channel_types'] = ['eeg'] * len(raw.ch_names)
            recording_info['channel_type_counts'] = {'eeg': len(raw.ch_names)}
        
        # Try to get additional info from raw.info
        try:
            if 'description' in raw.info:
                recording_info['description'] = raw.info['description']
            if 'experimenter' in raw.info:
                recording_info['experimenter'] = raw.info['experimenter']
            if 'subject_info' in raw.info and raw.info['subject_info']:
                recording_info['subject_info'] = raw.info['subject_info']
        except:
            logger.debug(f"[{subject_id}] Additional info not available")
        
        # Calculate some basic statistics if data is available
        try:
            # Get data in microvolts (like in loader.py)
            data = raw.get_data(units="uV")
            
            # Calculate statistics on the data
            recording_info['data_statistics'] = {
                'mean_amplitude': float(np.mean(data)),
                'std_amplitude': float(np.std(data)),
                'min_amplitude': float(np.min(data)),
                'max_amplitude': float(np.max(data)),
                'amplitude_range': float(np.ptp(data))
            }
            
            logger.debug(f"[{subject_id}] Data statistics calculated: "
                        f"mean={recording_info['data_statistics']['mean_amplitude']:.2f}μV, "
                        f"std={recording_info['data_statistics']['std_amplitude']:.2f}μV")
            
        except Exception as e:
            logger.warning(f"[{subject_id}] Could not calculate data statistics: {str(e)}")
            recording_info['data_statistics'] = {
                'mean_amplitude': 'N/A',
                'std_amplitude': 'N/A', 
                'min_amplitude': 'N/A',
                'max_amplitude': 'N/A',
                'amplitude_range': 'N/A'
            }
        
        # Save recording info to JSON file
        import json
        info_file = output_dir / 'recording_info.json'
        with open(info_file, 'w') as f:
            json.dump(recording_info, f, indent=2, default=str)
        
        # Also save as human-readable text
        info_text_file = output_dir / 'recording_info.txt'
        with open(info_text_file, 'w') as f:
            f.write(f"Recording Information for {subject_id}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Channels: {recording_info['n_channels']}\n")
            f.write(f"Sampling Rate: {recording_info['sampling_frequency']} Hz\n")
            f.write(f"Duration: {recording_info['duration_minutes']:.1f} minutes ({recording_info['duration_hours']:.2f} hours)\n")
            f.write(f"Total Samples: {recording_info['n_samples']:,}\n\n")
            
            f.write("Channel Types:\n")
            for ch_type, count in recording_info['channel_type_counts'].items():
                f.write(f"  {ch_type}: {count} channels\n")
            
            if 'data_statistics' in recording_info:
                stats = recording_info['data_statistics']
                f.write(f"\nData Statistics:\n")
                f.write(f"  Mean amplitude: {stats['mean_amplitude']:.2f} μV\n")
                f.write(f"  Std amplitude: {stats['std_amplitude']:.2f} μV\n")
                f.write(f"  Amplitude range: {stats['amplitude_range']:.2f} μV\n")
        
        logger.info(f"[{subject_id}] ✓ Recording info saved: {info_file}")
        logger.debug(f"[{subject_id}] {recording_info['n_channels']} channels, "
                    f"{recording_info['sampling_frequency']} Hz, "
                    f"{recording_info['duration_minutes']:.1f} min")
        
        return recording_info
        
    except Exception as e:
        logger.error(f"[{subject_id}] Error extracting recording info: {str(e)}")
        return None

def run_pre_detection_analysis(raw: mne.io.Raw, subject_id: str, 
                              output_dirs: Dict[str, Path]) -> Dict[str, Optional[Path]]:
    """
    Run all pre-detection analysis steps.
    
    Parameters
    ----------
    raw : mne.io.Raw
        MNE raw object
    subject_id : str
        Subject identifier
    output_dirs : Dict[str, Path]
        Dictionary of output directories
    
    Returns
    -------
    Dict[str, Optional[Path]]
        Dictionary with paths to generated files
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"[{subject_id}] Running pre-detection analysis...")
    
    results = {}
    
    # Create spectrogram
    results['spectrogram'] = create_spectrogram(
        raw, subject_id, output_dirs['visualizations']
    )
    
    # Create 2D sensor layout
    results['sensor_layout'] = create_2d_sensor_layout(
        raw, subject_id, output_dirs['visualizations']
    )
    
    # Extract recording info
    recording_info = extract_recording_info(
        raw, subject_id, output_dirs['data']
    )
    results['recording_info'] = recording_info
    
    # Count successful operations
    successful_ops = sum(1 for v in results.values() if v is not None)
    total_ops = len(results)
    
    logger.info(f"[{subject_id}] ✓ Pre-detection analysis completed: "
               f"{successful_ops}/{total_ops} operations successful")
    
    return results 