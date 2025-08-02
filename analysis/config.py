# =============================================================================
# PEDS OSA ANALYSIS - UNIFIED CONFIGURATION
# =============================================================================
# 
# This file contains all configuration parameters for both analysis pipelines:
# 1. AHI vs Spindle Density Analysis
# 2. Spindle Density vs TOVA Score Analysis
#
# Usage:
# 1. Modify the PROJECT_DIR path below to point to your data
# 2. Run: python run_ahi_analysis.py or python run_tova_analysis.py

import os

# =============================================================================
# PROJECT SETTINGS
# =============================================================================

# Path to your project directory (MODIFY THIS)
PROJECT_DIR = "/Volumes/Ido/002_PEDS_OSA"

# Output directories for results (will be created in PROJECT_DIR)
AHI_OUTPUT_DIR = os.path.join(PROJECT_DIR, "ahi_analysis_results")
TOVA_OUTPUT_DIR = os.path.join(PROJECT_DIR, "tova_analysis_results")

# =============================================================================
# DATA FILE LOCATIONS
# =============================================================================

# Expected data structure:
# PROJECT_DIR/
# ├── derivatives/sub-XXX/spindles/results/all_spindles_detailed.csv
# ├── demographics.csv
# └── TOVA.csv

# Reference files (for development/testing)
DEMOGRAPHICS_REFERENCE = "reference/demographics.csv"
TOVA_REFERENCE = "reference/TOVA.csv"
ANTERIOR_CHANNELS_FILE = "reference/segment_net.txt"

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

# AHI severity classification
AHI_SEVERITY_BINS = [0, 1, 5, 10, float('inf')]
AHI_SEVERITY_LABELS = ['Normal', 'Mild', 'Moderate', 'Severe']

# Regional analysis - analyze both anterior and posterior regions
ANALYZE_ANTERIOR_REGION_AHI = True
ANALYZE_POSTERIOR_REGION_AHI = True

# Analysis type for AHI study
ANALYSIS_TYPE_AHI = "regional_ahi_vs_spindle_density"
PREDICTOR_VARIABLE = "log_AHI"  # Using log-transformed AHI as predictor

# Regional outcome variables
ANTERIOR_OUTCOME = "anterior_spindle_density"
POSTERIOR_OUTCOME = "posterior_spindle_density"

# Output directory for regional AHI analysis
REGIONAL_AHI_OUTPUT_DIR = os.path.join(PROJECT_DIR, "regional_ahi_results")

# =============================================================================
# TOVA ANALYSIS PARAMETERS
# =============================================================================

# Regional analysis - analyze both anterior and posterior regions
ANALYZE_ANTERIOR_REGION = True
ANALYZE_POSTERIOR_REGION = True

# Analysis type for TOVA study
ANALYSIS_TYPE = "regional_spindle_density_vs_cohens_d"
OUTCOME_VARIABLE = "cohens_d"  # Using Cohen's d as outcome variable

# Regional predictor variables
ANTERIOR_PREDICTOR = "anterior_spindle_density"
POSTERIOR_PREDICTOR = "posterior_spindle_density"

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

# File naming conventions
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"