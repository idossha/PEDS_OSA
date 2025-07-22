#!/usr/bin/env python3
"""
Post-detection Module
=====================

Creates images and info regarding spindles: density, counts, topoplots, etc.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import mne
import yasa
from pathlib import Path
from typing import Dict, Optional

def create_comprehensive_visualization(analysis_results: Dict, raw: mne.io.Raw, 
                                     subject_id: str) -> Optional[plt.Figure]:
    """Create the main 2×3 comprehensive analysis plot."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"[{subject_id}] Creating comprehensive visualization...")
        plt.switch_backend('Agg')
        
        # Extract data
        channel_summary_all = analysis_results['channel_summary_all']
        channel_summary_slow = analysis_results['channel_summary_slow']
        channel_summary_fast = analysis_results['channel_summary_fast']
        stats = analysis_results['statistics']
        
        # Create topomap data
        topomap_all = _create_topomap_data(channel_summary_all, raw.info)
        topomap_slow = _create_topomap_data(channel_summary_slow, raw.info)
        topomap_fast = _create_topomap_data(channel_summary_fast, raw.info)
        
        # Create 2×3 figure
        fig = plt.figure(figsize=(18, 12))
        
        # Top row: Topographic maps
        _create_topomap_subplot(fig, 2, 3, 1, topomap_all, raw.info, 'General Spindle Count\n(10-16 Hz)', 'Reds')
        _create_topomap_subplot(fig, 2, 3, 2, topomap_slow, raw.info, 'Slow Spindle Count\n(10-12 Hz)', 'Reds')
        _create_topomap_subplot(fig, 2, 3, 3, topomap_fast, raw.info, 'Fast Spindle Count\n(12-16 Hz)', 'Reds')
        
        # Bottom row: Channel count bar plots
        _create_channel_count_subplot(fig, 2, 3, 4, channel_summary_all, 'General Spindle Counts by Channel', 'purple')
        _create_channel_count_subplot(fig, 2, 3, 5, channel_summary_slow, 'Slow Spindle Counts by Channel', 'blue')
        _create_channel_count_subplot(fig, 2, 3, 6, channel_summary_fast, 'Fast Spindle Counts by Channel', 'red')
        
        # Add title
        fig.suptitle(f'Comprehensive Spindle Analysis\n'
                    f'Total: {stats["total_spindles"]} spindles | '
                    f'Slow: {stats["slow_spindles"]} ({stats["slow_percentage"]:.1f}%) | '
                    f'Fast: {stats["fast_spindles"]} ({stats["fast_percentage"]:.1f}%)',
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        
        logger.info(f"[{subject_id}] ✓ Comprehensive visualization created")
        return fig
        
    except Exception as e:
        logger.error(f"[{subject_id}] Error creating comprehensive visualization: {str(e)}")
        return None

def create_spindle_density(analysis_results: Dict, raw: mne.io.Raw, 
                          subject_id: str) -> Optional[plt.Figure]:
    """Create spindle density visualization showing spindles/min over time."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"[{subject_id}] Creating spindle density plot...")
        plt.switch_backend('Agg')
        
        fig, ax = plt.subplots(figsize=(15, 6))
        
        df_all = analysis_results['df_all']
        
        if df_all is not None and len(df_all) > 0:
            # Calculate spindle density over time
            window_size = 60  # 1-minute windows
            step_size = 30    # 30-second steps (50% overlap)
            
            # Get total recording duration from raw data
            total_duration = raw.times[-1]  # Total duration in seconds
            
            # Create time windows
            windows = np.arange(0, total_duration - window_size, step_size)
            density = []
            time_points = []
            
            for start in windows:
                end = start + window_size
                # Count spindles in this time window
                window_spindles = df_all[(df_all['Start'] >= start) & 
                                       (df_all['Start'] < end)].shape[0]
                # Convert to spindles per minute
                spindles_per_min = window_spindles / (window_size / 60)
                density.append(spindles_per_min)
                time_points.append((start + end) / 2 / 60)  # Midpoint in minutes
            
            # Plot spindle density
            ax.plot(time_points, density, linewidth=2, color='purple', alpha=0.8)
            ax.fill_between(time_points, density, alpha=0.3, color='purple')
            
            # Customize plot
            ax.set_xlabel('Time (minutes)', fontsize=14)
            ax.set_ylabel('Spindle Density (spindles/min)', fontsize=14)
            ax.set_title('Spindle Density Over Time', fontsize=16, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3)
            
            # Add statistics as text
            mean_density = np.mean(density)
            max_density = np.max(density)
            total_spindles = len(df_all)
            
            stats_text = f'Total Spindles: {total_spindles:,}\nMean Density: {mean_density:.1f} spindles/min\nPeak Density: {max_density:.1f} spindles/min'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Set reasonable y-axis limits
            ax.set_ylim(0, max_density * 1.1)
            
            logger.info(f"[{subject_id}] Spindle density calculated: {len(time_points)} time points, mean={mean_density:.1f} spindles/min")
            
        else:
            ax.text(0.5, 0.5, 'No spindles detected', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=16)
            ax.set_xlabel('Time (minutes)', fontsize=14)
            ax.set_ylabel('Spindle Density (spindles/min)', fontsize=14)
            ax.set_title('Spindle Density Over Time', fontsize=16, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3)
            logger.warning(f"[{subject_id}] No spindles found for density analysis")
        
        plt.tight_layout()
        
        logger.info(f"[{subject_id}] ✓ Spindle density plot created")
        return fig
        
    except Exception as e:
        logger.error(f"[{subject_id}] Error creating spindle density plot: {str(e)}")
        return None

def save_all_visualizations(analysis_results: Dict, raw: mne.io.Raw, 
                           output_dir: Path, subject_id: str) -> Dict[str, Path]:
    """Create and save all post-detection visualizations."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"[{subject_id}] Creating all post-detection visualizations...")
        saved_files = {}
        
        # Comprehensive analysis
        comp_fig = create_comprehensive_visualization(analysis_results, raw, subject_id)
        if comp_fig:
            comp_path = output_dir / 'comprehensive_analysis.png'
            comp_fig.savefig(comp_path, dpi=300, bbox_inches='tight')
            plt.close(comp_fig)
            saved_files['comprehensive_analysis'] = comp_path
        
        # Spindle density analysis
        temp_fig = create_spindle_density(analysis_results, raw, subject_id)
        if temp_fig:
            temp_path = output_dir / 'spindle_density.png'
            temp_fig.savefig(temp_path, dpi=300, bbox_inches='tight')
            plt.close(temp_fig)
            saved_files['spindle_density'] = temp_path
        
        # Additional visualizations would go here...
        
        logger.info(f"[{subject_id}] ✓ Created {len(saved_files)} visualizations")
        return saved_files
        
    except Exception as e:
        logger.error(f"[{subject_id}] Error in post-detection visualization: {str(e)}")
        return {}

# Helper functions (simplified versions)
def _create_topomap_data(channel_summary, raw_info):
    """Create data array for topographic mapping."""
    data_for_topomap = np.zeros(len(raw_info['ch_names']))
    ch_names = raw_info['ch_names']
    
    for ch in channel_summary.index:
        if ch in ch_names:
            idx = ch_names.index(ch)
            data_for_topomap[idx] = channel_summary.loc[ch, 'Count']
    return data_for_topomap

def _create_topomap_subplot(fig, nrows, ncols, position, data, raw_info, title, cmap):
    """Create a topographic map subplot."""
    ax = plt.subplot(nrows, ncols, position)
    try:
        im, cm = mne.viz.plot_topomap(data, raw_info, cmap=cmap, sensors=True, show=False, axes=ax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Count', fontsize=12)
    except Exception as e:
        ax.text(0.5, 0.5, f'Topomap Error\n{str(e)[:50]}...', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=14, fontweight='bold')

def _create_channel_count_subplot(fig, nrows, ncols, position, channel_summary, title, color):
    """Create a channel count bar plot subplot."""
    ax = plt.subplot(nrows, ncols, position)
    channels_with_data = channel_summary[channel_summary['Count'] > 0]
    
    if len(channels_with_data) > 0:
        ax.bar(range(len(channels_with_data)), channels_with_data['Count'], color=color, alpha=0.7)
        total_count = int(channels_with_data['Count'].sum())
        n_channels = len(channels_with_data)
        ax.text(0.02, 0.98, f'Total: {total_count} spindles\nChannels: {n_channels}', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'No spindles detected', ha='center', va='center', transform=ax.transAxes)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Channel Index', fontsize=12)
    ax.set_ylabel('Spindle Count', fontsize=12)
    ax.grid(True, alpha=0.3) 