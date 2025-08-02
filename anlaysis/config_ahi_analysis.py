# =============================================================================
# PEDS OSA: SPINDLE DENSITY VS AHI ANALYSIS - CONFIGURATION
# =============================================================================
# 
# This file contains all configuration parameters for the spindle density analysis.
# 
# Usage:
# 1. Copy this file to config.py
# 2. Modify the paths and parameters below
# 3. Run: python run_ahi_analysis.py --config config.py
# 
# Analysis Overview:
# - Examines relationship between spindle density and AHI
# - Tests age × AHI interactions
# - Compares Control vs OSA groups
# - Uses log-transformed AHI as recommended in reference

import os

# =============================================================================
# PROJECT SETTINGS
# =============================================================================

# Path to your project directory (MODIFY THIS)
PROJECT_DIR = "/path/to/your/002_PEDS_OSA"

# Output directory for results
OUTPUT_DIR = os.path.join(PROJECT_DIR, "analysis_2")

# =============================================================================
# SPINDLE DETECTION PARAMETERS
# =============================================================================

# Frequency threshold for slow vs fast spindle classification
SPINDLE_FREQUENCY_THRESHOLD = 12  # Hz

# Duration limits for spindle detection
MIN_SPINDLE_DURATION = 0.5  # seconds
MAX_SPINDLE_DURATION = 3.0  # seconds

# =============================================================================
# AHI ANALYSIS PARAMETERS
# =============================================================================

# AHI threshold for OSA diagnosis
AHI_THRESHOLD = 1.0

# AHI severity classification bins
AHI_SEVERITY_BINS = [0, 1, 5, 10, float('inf')]
AHI_SEVERITY_LABELS = ['Normal', 'Mild', 'Moderate', 'Severe']

# =============================================================================
# STATISTICAL ANALYSIS PARAMETERS
# =============================================================================

# Significance level for hypothesis testing
ALPHA_LEVEL = 0.05

# Random seed for reproducible results
RANDOM_SEED = 42

# =============================================================================
# DATA PROCESSING PARAMETERS
# =============================================================================

# Minimum recording duration to include in analysis
MIN_RECORDING_DURATION = 100  # minutes

# Maximum proportion of missing data allowed
MAX_MISSING_DATA = 0.2  # 20%

# =============================================================================
# REGIONAL ANALYSIS SETTINGS
# =============================================================================

# EEG channel definitions for regional analysis
FRONTAL_CHANNELS = ['F3', 'F4', 'Fz']
PARIETAL_CHANNELS = ['P3', 'P4', 'Pz']

# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================

# Plot styling and appearance
PLOT_STYLE = 'default'
COLOR_PALETTE = 'husl'
FIGURE_SIZE = (18, 12)
DPI = 300  # Resolution for saved plots

# =============================================================================
# OUTPUT SETTINGS
# =============================================================================

# Whether to save all results
SAVE_RESULTS = True 