#!/usr/bin/env python3
"""
PEDS OSA: Spindle Density vs AHI Analysis - Execution Script
============================================================

This script provides a command-line interface for running the spindle density vs AHI analysis.
It handles dependency checking, project validation, and execution of the analysis pipeline.

Usage:
    python run_ahi_analysis.py [--config CONFIG_FILE] [--project-dir PROJECT_DIR]
    python run_ahi_analysis.py --check-only  # Validate setup without running analysis
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def check_dependencies():
    """
    Verify that all required Python packages are installed.
    
    Returns
    -------
    bool
        True if all dependencies are available, False otherwise
    """
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'scipy', 'statsmodels', 'pathlib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def load_config(config_file=None):
    """
    Load analysis configuration from file or use default settings.
    
    Parameters
    ----------
    config_file : str, optional
        Path to configuration file. If None, uses built-in defaults.
    
    Returns
    -------
    object
        Configuration object with analysis parameters
    """
    if config_file and os.path.exists(config_file):
        # Load custom config
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_file)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        return config
    else:
        # Use default configuration
        class DefaultConfig:
            PROJECT_DIR = "/Volumes/Ido/002_PEDS_OSA"
            OUTPUT_DIR = os.path.join(PROJECT_DIR, "analysis_2")
            SPINDLE_FREQUENCY_THRESHOLD = 12
            MIN_SPINDLE_DURATION = 0.5
            MAX_SPINDLE_DURATION = 3.0
            AHI_THRESHOLD = 1.0
            ALPHA_LEVEL = 0.05
            RANDOM_SEED = 42
            SAVE_RESULTS = True
            MIN_RECORDING_DURATION = 100
            MAX_MISSING_DATA = 0.2
        
        return DefaultConfig()

def validate_project_structure(project_dir):
    """
    Validate that the project directory has the expected structure.
    
    Parameters
    ----------
    project_dir : str
        Path to the project directory to validate
    
    Returns
    -------
    bool
        True if structure is valid, False if issues found
    """
    required_paths = [
        os.path.join(project_dir, "derivatives"),
        os.path.join(project_dir, "analysis", "reference"),
    ]
    
    missing_paths = []
    for path in required_paths:
        if not os.path.exists(path):
            missing_paths.append(path)
    
    if missing_paths:
        print("Warning: Some expected project paths are missing:")
        for path in missing_paths:
            print(f"  - {path}")
        print("The analysis may fail if required data files are not found.")
        return False
    
    return True

def find_spindle_files(project_dir):
    """
    Locate all spindle data files in the project directory.
    
    Parameters
    ----------
    project_dir : str
        Path to the project directory
    
    Returns
    -------
    list
        List of Path objects pointing to spindle data files
    """
    pattern = os.path.join(project_dir, "derivatives", "sub-*", "spindles", "results", "all_spindles_detailed.csv")
    spindle_files = list(Path(project_dir).glob("derivatives/sub-*/spindles/results/all_spindles_detailed.csv"))
    
    print(f"Found {len(spindle_files)} spindle data files")
    return spindle_files

def find_demographics_file(project_dir):
    """
    Locate the demographics file in the project directory.
    
    Parameters
    ----------
    project_dir : str
        Path to the project directory
    
    Returns
    -------
    str or None
        Path to demographics file if found, None otherwise
    """
    possible_locations = [
        os.path.join(project_dir, "analysis", "reference", "demographics.csv"),
        os.path.join(project_dir, "demographics.csv"),
        os.path.join(project_dir, "analysis", "demographics.csv"),
    ]
    
    for location in possible_locations:
        if os.path.exists(location):
            print(f"Found demographics file: {location}")
            return location
    
    print("Warning: Demographics file not found in expected locations:")
    for location in possible_locations:
        print(f"  - {location}")
    return None

def run_analysis(config):
    """
    Execute the complete spindle density analysis pipeline.
    
    Parameters
    ----------
    config : object
        Configuration object with analysis parameters
    
    Returns
    -------
    bool
        True if analysis completed successfully, False otherwise
    """
    print("PEDS OSA: Spindle Density vs AHI Analysis")
    print("=" * 40)
    
    # Check dependencies
    print("\n1. Checking dependencies...")
    if not check_dependencies():
        return False
    
    # Validate project structure
    print("\n2. Validating project structure...")
    if not validate_project_structure(config.PROJECT_DIR):
        print("Continuing with analysis (some paths may be missing)...")
    
    # Find data files
    print("\n3. Locating data files...")
    spindle_files = find_spindle_files(config.PROJECT_DIR)
    demographics_file = find_demographics_file(config.PROJECT_DIR)
    
    if not spindle_files:
        print("Error: No spindle data files found!")
        return False
    
    # Create output directory
    print(f"\n4. Creating output directory: {config.OUTPUT_DIR}")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Run the analysis
    print("\n5. Running spindle density analysis...")
    try:
        # Import and run the analysis
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from analyze_spindle_density_vs_ahi import main
        
        # Run the analysis with the specified project directory
        main(config.PROJECT_DIR)
        
        print("\n✓ Analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function for command-line execution.
    
    This function parses command-line arguments and orchestrates the analysis workflow.
    """
    parser = argparse.ArgumentParser(
        description="Run PEDS OSA: Spindle Density vs AHI Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_ahi_analysis.py
  python run_ahi_analysis.py --project-dir /path/to/002_PEDS_OSA
  python run_ahi_analysis.py --config my_config.py
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        help='Path to configuration file (default: use built-in defaults)'
    )
    
    parser.add_argument(
        '--project-dir', 
        type=str, 
        help='Path to project directory (overrides config file)'
    )
    
    parser.add_argument(
        '--check-only', 
        action='store_true', 
        help='Only check dependencies and project structure, don\'t run analysis'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override project directory if specified
    if args.project_dir:
        config.PROJECT_DIR = args.project_dir
        config.OUTPUT_DIR = os.path.join(args.project_dir, "analysis_2")
    
    print(f"Project directory: {config.PROJECT_DIR}")
    print(f"Output directory: {config.OUTPUT_DIR}")
    
    if args.check_only:
        # Only perform checks
        print("\nPerforming checks only...")
        check_dependencies()
        validate_project_structure(config.PROJECT_DIR)
        find_spindle_files(config.PROJECT_DIR)
        find_demographics_file(config.PROJECT_DIR)
        print("\nChecks completed.")
    else:
        # Run full analysis
        success = run_analysis(config)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 