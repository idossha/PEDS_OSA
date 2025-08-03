#!/usr/bin/env python3
"""
PEDS OSA Regional Spindle Analysis - Main Entry Point
====================================================

Simple wrapper script to run the complete regional spindle analysis pipeline.
Generates two publication-ready figures:
1. main_results_figure.png (3 panels)
2. demographics_dashboard.png (6 panels)

Usage:
    python run_regional_analysis.py [--project-dir PROJECT_DIR]
    python run_regional_analysis.py --check-only  # Validate setup only
"""

import os
import sys
import argparse
from pathlib import Path

def check_dependencies():
    """Verify that all required Python packages are installed."""
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'scipy', 'statsmodels'
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

def validate_project_structure(project_dir):
    """Validate that the project directory has the expected structure."""
    print(f"Validating project structure in: {project_dir}")
    
    required_files = [
        os.path.join(project_dir, "demographics.csv"),
        os.path.join(project_dir, "TOVA.csv"),
        os.path.join(project_dir, "derivatives")
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Warning: Some required files/directories are missing:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    # Check for spindle files
    spindle_files = list(Path(project_dir).glob("derivatives/sub-*/spindles/results/all_spindles_detailed.csv"))
    print(f"Found {len(spindle_files)} spindle data files")
    
    if len(spindle_files) == 0:
        print("Error: No spindle data files found!")
        return False
    
    return True

def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Run PEDS OSA Regional Spindle Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_regional_analysis.py
  python run_regional_analysis.py --project-dir /path/to/002_PEDS_OSA
  python run_regional_analysis.py --check-only
        """
    )
    
    parser.add_argument(
        '--project-dir', 
        type=str, 
        help='Path to project directory (default: from config.py)'
    )
    
    parser.add_argument(
        '--check-only', 
        action='store_true', 
        help='Only check dependencies and project structure, don\'t run analysis'
    )
    
    args = parser.parse_args()
    
    # Determine project directory
    if args.project_dir:
        project_dir = args.project_dir
    else:
        try:
            from config import PROJECT_DIR
            project_dir = PROJECT_DIR
        except ImportError:
            print("Error: No project directory specified and config.py not found.")
            print("Please specify --project-dir or ensure config.py exists.")
            sys.exit(1)
    
    print("PEDS OSA Regional Spindle Analysis")
    print("=" * 40)
    print(f"Project directory: {project_dir}")
    
    if args.check_only:
        # Only perform validation
        print("\nPerforming validation checks only...")
        deps_ok = check_dependencies()
        struct_ok = validate_project_structure(project_dir)
        
        if deps_ok and struct_ok:
            print("\n✓ All validation checks passed!")
        else:
            print("\n✗ Some validation checks failed.")
            sys.exit(1)
    else:
        # Run full analysis
        print("\n1. Checking dependencies...")
        if not check_dependencies():
            sys.exit(1)
        
        print("\n2. Validating project structure...")
        if not validate_project_structure(project_dir):
            print("Warning: Some validation checks failed. Continuing...")
        
        print("\n3. Running complete regional analysis...")
        try:
            # Import and run the complete analysis
            from analysis.regional_tova_analysis import RegionalSpindleAnalysis
            
            output_dir = os.path.join(project_dir, "regional_analysis_results")
            analysis = RegionalSpindleAnalysis(project_dir, output_dir)
            df, results = analysis.run_complete_analysis()
            
            print(f"\n✓ ANALYSIS COMPLETED SUCCESSFULLY!")
            print(f"✓ Results saved to: {output_dir}")
            
        except Exception as e:
            print(f"\n✗ Analysis failed with error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()