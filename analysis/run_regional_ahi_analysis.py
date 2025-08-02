#!/usr/bin/env python3
"""
PEDS OSA: Regional AHI Analysis Runner
=====================================

Streamlined entry point for the complete regional AHI vs spindle density analysis.

This analysis examines:
1. Anterior spindle density vs log(AHI)
2. Posterior spindle density vs log(AHI)  
3. Regional comparison of sleep apnea effects

Key Features:
- Single script with all analysis logic
- Analyzes both anterior and posterior regions simultaneously
- Direct statistical comparison between regions (Steiger's Z-test)
- Generates only the essential publication figures
- Controls for age and sex in all models

Usage:
    python run_regional_ahi_analysis.py --project-dir /path/to/your/002_PEDS_OSA
    python run_regional_ahi_analysis.py --check-only  # Validate setup without running analysis

Output: PROJECT_DIR/regional_ahi_results/
- main_results_figure.png (3 panels)
- demographics_dashboard.png (6 panels)
- analysis_summary.txt (comprehensive technical summary)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def check_dependencies():
    """Verify that all required Python packages are installed."""
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

def validate_project_structure(project_dir):
    """Validate that the project directory has the expected structure."""
    print(f"Validating project structure in: {project_dir}")
    
    # Check for derivatives directory
    derivatives_dir = os.path.join(project_dir, "derivatives")
    if not os.path.exists(derivatives_dir):
        print(f"Warning: Derivatives directory not found: {derivatives_dir}")
        return False
    
    # Check for demographics file
    demographics_locations = [
        os.path.join(project_dir, "demographics.csv"),
        os.path.join(project_dir, "analysis", "reference", "demographics.csv"),
    ]
    
    demographics_found = False
    for location in demographics_locations:
        if os.path.exists(location):
            print(f"Found demographics file: {location}")
            demographics_found = True
            break
    
    if not demographics_found:
        print("Warning: Demographics file not found in expected locations:")
        for location in demographics_locations:
            print(f"  - {location}")
        return False
    
    return True

def find_data_files(project_dir):
    """Locate spindle data files in the project directory."""
    spindle_files = list(Path(project_dir).glob("derivatives/sub-*/spindles/results/all_spindles_detailed.csv"))
    print(f"Found {len(spindle_files)} spindle data files")
    
    if not spindle_files:
        print("Error: No spindle data files found!")
        print("Expected location: derivatives/sub-*/spindles/results/all_spindles_detailed.csv")
        return False
    
    return True

def run_analysis(project_dir):
    """Execute the regional AHI analysis pipeline."""
    print("PEDS OSA Regional AHI Analysis")
    print("========================================")
    print(f"Project directory: {project_dir}")
    
    # Check dependencies
    print("\n1. Checking dependencies...")
    if not check_dependencies():
        return False
    
    # Validate project structure
    print("\n2. Validating project structure...")
    if not validate_project_structure(project_dir):
        print("Error: Project structure validation failed!")
        return False
    
    # Find data files
    if not find_data_files(project_dir):
        return False
    
    # Run the analysis
    print("\n3. Running complete regional analysis...")
    try:
        # Import and run the analysis
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from regional_ahi_analysis import main
        
        # Run the analysis with the specified project directory
        success = main(project_dir)
        
        if success:
            print("\n✓ ANALYSIS COMPLETED SUCCESSFULLY!")
            print(f"✓ Results saved to: {project_dir}/regional_ahi_results")
            return True
        else:
            print("\n✗ Analysis failed!")
            return False
        
    except Exception as e:
        print(f"\n✗ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Run PEDS OSA Regional AHI Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_regional_ahi_analysis.py --project-dir /path/to/002_PEDS_OSA
  python run_regional_ahi_analysis.py --check-only
        """
    )
    
    parser.add_argument(
        '--project-dir', 
        type=str, 
        required=True,
        help='Path to project directory containing derivatives and demographics.csv'
    )
    
    parser.add_argument(
        '--check-only', 
        action='store_true', 
        help='Only check dependencies and project structure, don\'t run analysis'
    )
    
    args = parser.parse_args()
    
    # Validate project directory exists
    if not os.path.exists(args.project_dir):
        print(f"Error: Project directory does not exist: {args.project_dir}")
        sys.exit(1)
    
    if args.check_only:
        # Only perform checks
        print("Performing validation checks only...")
        deps_ok = check_dependencies()
        struct_ok = validate_project_structure(args.project_dir)
        files_ok = find_data_files(args.project_dir)
        
        if deps_ok and struct_ok and files_ok:
            print("\n✓ All validation checks passed!")
            sys.exit(0)
        else:
            print("\n✗ Some validation checks failed!")
            sys.exit(1)
    else:
        # Run full analysis
        success = run_analysis(args.project_dir)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()