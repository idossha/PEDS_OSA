# =============================================================================
# CONFIGURATION TEMPLATE FOR PEDS OSA SPINDLE ANALYSIS
# =============================================================================
# 
# Copy this file to config.py and modify the paths below to point to your actual data.
# Then run: python peds_osa_spindle_analysis.py

import os

# Path to your project directory
PROJECT_DIR = "/path/to/your/002_PEDS_OSA"

# Analysis parameters
SPINDLE_FREQUENCY_THRESHOLD = 12  # Hz - threshold for slow vs fast spindles
MIN_SPINDLE_DURATION = 0.5  # seconds
MAX_SPINDLE_DURATION = 3.0  # seconds

# Statistical analysis parameters
ALPHA_LEVEL = 0.05  # Significance level
RANDOM_SEED = 42  # For reproducible results

# Output settings
SAVE_RESULTS = True
OUTPUT_DIR = os.path.join(PROJECT_DIR, "analysis")

# Data processing
MIN_RECORDING_DURATION = 100  # minutes - minimum recording time to include
MAX_MISSING_DATA = 0.2  # maximum proportion of missing data allowed 